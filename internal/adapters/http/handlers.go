// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bufio"
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

	// Check cache first (unless no_cache is set)
	// DISABLED: Cache causing issues - model responses seem "dumb"
	cacheKey := h.GenerateKey(req, route)
	logEntry.CacheKey = cacheKey

	if false && !req.NoCache { // Cache disabled
		if cached, ok := h.Cache.Get(cacheKey); ok {
			// Cache hit - fake stream the cached response
			h.fakeStreamCachedResponse(w, cached, route, logEntry, startTime)
			return
		}
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
		"stream_options": map[string]bool{
			"include_usage": true, // Get token counts in streaming mode
		},
	}
	// Use MaxCompletionTokens if set, otherwise fall back to MaxTokens
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		// Newer OpenAI models use max_completion_tokens
		if strings.HasPrefix(route.Model, "gpt-4o") || strings.HasPrefix(route.Model, "gpt-4.1") ||
			strings.HasPrefix(route.Model, "gpt-5") || strings.HasPrefix(route.Model, "o1") ||
			strings.HasPrefix(route.Model, "o3") || strings.HasPrefix(route.Model, "o4") {
			openaiReq["max_completion_tokens"] = maxTokens
		} else {
			openaiReq["max_tokens"] = maxTokens
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

	// Stream the response while capturing content and tokens
	var responseContent strings.Builder
	var inputTokens, outputTokens int
	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size for large chunks
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		// Forward the line to client
		fmt.Fprintln(w, line)
		flusher.Flush()

		// Parse SSE data lines to extract content and usage
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				continue
			}
			// Parse the chunk to extract content delta and usage
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
				Usage *struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				// Accumulate content
				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					responseContent.WriteString(chunk.Choices[0].Delta.Content)
				}
				// Capture usage from final chunk (OpenAI sends this with stream_options)
				if chunk.Usage != nil {
					inputTokens = chunk.Usage.PromptTokens
					outputTokens = chunk.Usage.CompletionTokens
				}
			}
		}
	}

	// Build a synthetic response for logging and caching
	syntheticResp := map[string]interface{}{
		"choices": []map[string]interface{}{
			{"message": map[string]string{"role": "assistant", "content": responseContent.String()}},
		},
		"usage": map[string]int{"prompt_tokens": inputTokens, "completion_tokens": outputTokens},
	}
	respBytes, _ := json.Marshal(syntheticResp)

	// Cache the response for future streaming requests
	h.Cache.Set(cacheKey, body, respBytes)

	// Log success with captured data
	logEntry.Success = true
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.RequestType = "stream"
	logEntry.ResponseBody = respBytes
	logEntry.InputTokens = inputTokens
	logEntry.OutputTokens = outputTokens
	logEntry.CostUSD = h.CalculateCost(route.Model, inputTokens, outputTokens)
	h.Logger.LogRequest(logEntry)
	h.Metrics.RecordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, inputTokens, outputTokens, logEntry.CostUSD, false)
}

// fakeStreamCachedResponse sends a cached response as a fake SSE stream.
func (h *ChatHandler) fakeStreamCachedResponse(w http.ResponseWriter, cached []byte, route *domain.RouteConfig, logEntry *domain.RequestLog, startTime time.Time) {
	// Parse cached response
	var resp domain.ChatCompletionResponse
	if err := json.Unmarshal(cached, &resp); err != nil {
		http.Error(w, "Failed to parse cached response", http.StatusInternalServerError)
		return
	}

	// Set up SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Cached", "true")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Get response content
	var content string
	if len(resp.Choices) > 0 {
		if c, ok := resp.Choices[0].Message.Content.(string); ok {
			content = c
		}
	}

	// Stream content in chunks (simulate real streaming)
	// Use runes to avoid breaking UTF-8 multi-byte characters
	runes := []rune(content)
	chunkSize := 20 // runes per chunk
	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunk := string(runes[i:end])

		// Build SSE chunk
		sseChunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": chunk}},
			},
		}
		chunkJSON, _ := json.Marshal(sseChunk)
		fmt.Fprintf(w, "data: %s\n\n", chunkJSON)
		flusher.Flush()
	}

	// Send final chunk with usage
	var inputTokens, outputTokens int
	if resp.Usage != nil {
		inputTokens = resp.Usage.PromptTokens
		outputTokens = resp.Usage.CompletionTokens
	}
	finalChunk := map[string]interface{}{
		"choices": []map[string]interface{}{
			{"delta": map[string]interface{}{}, "finish_reason": "stop"},
		},
		"usage": map[string]int{
			"prompt_tokens":     inputTokens,
			"completion_tokens": outputTokens,
		},
	}
	finalJSON, _ := json.Marshal(finalChunk)
	fmt.Fprintf(w, "data: %s\n\n", finalJSON)
	fmt.Fprintln(w, "data: [DONE]")
	flusher.Flush()

	// Log cache hit
	logEntry.Cached = true
	logEntry.Success = true
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.RequestType = "stream"
	logEntry.ResponseBody = cached
	logEntry.InputTokens = inputTokens
	logEntry.OutputTokens = outputTokens
	h.Logger.LogRequest(logEntry)
	h.Metrics.RecordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, 0, 0, 0, true)
}
