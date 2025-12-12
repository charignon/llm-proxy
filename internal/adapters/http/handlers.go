// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// ChatHandler handles chat completion requests.
type ChatHandler struct {
	Router        *app.Router
	Providers     map[string]ports.ChatProvider
	Cache         ports.Cache
	Logger        ports.RequestLogger
	Metrics       MetricsRecorder
	GenerateKey   func(req *domain.ChatCompletionRequest, route *domain.RouteConfig) string
	CalculateCost func(model string, inputTokens, outputTokens int) float64

	// Pending request tracking
	AddPending    func(req *domain.ChatCompletionRequest, route *domain.RouteConfig, startTime time.Time) string
	RemovePending func(id string)

	// Model availability check (returns true if model is disabled)
	IsModelDisabled func(model string) bool
}

// MetricsRecorder interface for recording request metrics.
type MetricsRecorder interface {
	RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool)
}

// getClientIP extracts the client IP address from the request.
// It checks X-Forwarded-For and X-Real-IP headers first (for proxied requests),
// then falls back to RemoteAddr.
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (may contain multiple IPs, take the first)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr (strip port if present)
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr // Return as-is if no port
	}
	return ip
}

// ServeHTTP handles the chat completion request.
func (h *ChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req domain.ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Allow usecase from header (for clients like aider that can't add custom body fields)
	if req.Usecase == "" {
		req.Usecase = r.Header.Get("X-Usecase")
	}

	// Validate usecase is provided
	if req.Usecase == "" {
		http.Error(w, "Missing required field: usecase. Please provide a usecase to identify the caller (body field or X-Usecase header).", http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestedModel: req.Model,
		Precision:      req.Precision,
		Usecase:        req.Usecase,
		HasImages:      req.HasImages(),
		RequestBody:    body,
		ClientIP:       getClientIP(r),
	}
	if req.Sensitive != nil {
		logEntry.Sensitive = *req.Sensitive
	}
	// Check if this is a replay request
	if r.Header.Get("X-LLM-Proxy-Replay") == "true" {
		logEntry.IsReplay = true
	}

	// Resolve routing
	route, err := h.Router.ResolveRoute(&req)
	if err != nil {
		logEntry.Provider = "routing_failed"
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)

		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Check if model is disabled
	if h.IsModelDisabled != nil && h.IsModelDisabled(route.Model) {
		logEntry.Provider = route.Provider
		logEntry.Model = route.Model
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("model %s is disabled", route.Model)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)

		http.Error(w, fmt.Sprintf("Model %s is currently disabled", route.Model), http.StatusServiceUnavailable)
		return
	}

	logEntry.Provider = route.Provider
	logEntry.Model = route.Model

	// Handle streaming requests
	if req.Stream {
		h.handleStreaming(w, r, &req, route, body, logEntry, startTime)
		return
	}

	// Check cache (only for non-streaming)
	cacheKey := h.GenerateKey(&req, route)
	logEntry.CacheKey = cacheKey

	if !req.NoCache {
		if cached, ok := h.Cache.Get(cacheKey); ok {
			logEntry.Cached = true
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			logEntry.Success = true
			logEntry.ResponseBody = cached
			requestID := h.Logger.LogRequest(logEntry)

			// Record cache hit metrics
			h.Metrics.RecordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, 0, 0, 0, true)

			// Add cached flag to response
			var resp domain.ChatCompletionResponse
			json.Unmarshal(cached, &resp)
			resp.Cached = true

			w.Header().Set("Content-Type", "application/json")
			if requestID > 0 {
				w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
			}
			w.Header().Set("X-LLM-Proxy-Cached", "true")
			json.NewEncoder(w).Encode(resp)
			return
		}
	}

	// Track pending request
	pendingID := h.AddPending(&req, route, startTime)
	defer h.RemovePending(pendingID)

	// Make the actual call
	var resp *domain.ChatCompletionResponse
	provider, ok := h.Providers[route.Provider]
	if !ok {
		err = fmt.Errorf("unknown provider: %s", route.Provider)
	} else {
		resp, err = provider.Chat(&req, route.Model)
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()

	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()

		h.Logger.LogRequest(logEntry)
		h.Metrics.RecordRequest(route.Provider, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)

		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Update log entry
	if resp.Usage != nil {
		logEntry.InputTokens = resp.Usage.PromptTokens
		logEntry.OutputTokens = resp.Usage.CompletionTokens
	}
	logEntry.CostUSD = h.CalculateCost(route.Model, logEntry.InputTokens, logEntry.OutputTokens)
	logEntry.Success = true

	// Cache the response
	respBytes, _ := json.Marshal(resp)
	h.Cache.Set(cacheKey, body, respBytes)

	logEntry.ResponseBody = respBytes
	requestID := h.Logger.LogRequest(logEntry)

	// Record metrics
	h.Metrics.RecordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, logEntry.InputTokens, logEntry.OutputTokens, logEntry.CostUSD, false)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", logEntry.LatencyMs))
	w.Header().Set("X-LLM-Proxy-Cost-USD", fmt.Sprintf("%.6f", logEntry.CostUSD))
	if requestID > 0 {
		w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
	}
	json.NewEncoder(w).Encode(resp)
}

// handleStreaming handles streaming chat completion requests by forwarding to OpenAI with stream=true.
func (h *ChatHandler) handleStreaming(w http.ResponseWriter, r *http.Request, req *domain.ChatCompletionRequest, route *domain.RouteConfig, body []byte, logEntry *domain.RequestLog, startTime time.Time) {
	// Only OpenAI streaming is supported for now
	if route.Provider != "openai" {
		http.Error(w, fmt.Sprintf("Streaming not supported for provider: %s", route.Provider), http.StatusBadRequest)
		return
	}

	// Get OpenAI provider to access API key
	provider, ok := h.Providers[route.Provider]
	if !ok {
		http.Error(w, "OpenAI provider not configured", http.StatusInternalServerError)
		return
	}
	openaiProvider, ok := provider.(interface{ GetAPIKey() string })
	if !ok {
		http.Error(w, "Provider does not support streaming", http.StatusInternalServerError)
		return
	}
	apiKey := openaiProvider.GetAPIKey()

	// Build OpenAI streaming request
	openaiReq := map[string]interface{}{
		"model":    route.Model,
		"messages": req.Messages,
		"stream":   true,
	}
	if req.MaxTokens > 0 {
		if strings.HasPrefix(route.Model, "gpt-4o") || strings.HasPrefix(route.Model, "gpt-5") || strings.HasPrefix(route.Model, "o1") {
			openaiReq["max_completion_tokens"] = req.MaxTokens
		} else {
			openaiReq["max_tokens"] = req.MaxTokens
		}
	}
	if req.Temperature > 0 {
		openaiReq["temperature"] = req.Temperature
	}
	if len(req.Tools) > 0 {
		openaiReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		openaiReq["tool_choice"] = req.ToolChoice
	}

	reqBody, _ := json.Marshal(openaiReq)

	// Make streaming request to OpenAI
	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", strings.NewReader(string(reqBody)))
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		http.Error(w, "OpenAI request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("OpenAI error %d: %s", resp.StatusCode, string(respBody))
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		http.Error(w, string(respBody), resp.StatusCode)
		return
	}

	// Set up SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Stream the response
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			flusher.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	// Log success
	logEntry.Success = true
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.RequestType = "stream"
	h.Logger.LogRequest(logEntry)
	h.Metrics.RecordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, 0, 0, 0, false)
}
