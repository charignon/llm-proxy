// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/adapters/providers"
	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// usecaseFromBearer extracts usecase from the Authorization Bearer token.
// This allows clients like Jan that only support API key fields to pass
// the usecase as the API key (e.g., "jan" becomes usecase "jan").
func usecaseFromBearer(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		token := strings.TrimPrefix(auth, "Bearer ")
		// Ignore common placeholder values
		if token != "" && token != "not-needed" && token != "sk-" {
			return token
		}
	}
	return ""
}

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
	AddPending    func(req *domain.ChatCompletionRequest, route *domain.RouteConfig, startTime time.Time, cancel context.CancelFunc) string
	RemovePending func(id string)

	// Model availability check (returns true if model is disabled)
	IsModelDisabled func(model string) bool

	// GetProviderOverride checks if a different provider should be used.
	// Used to route ollama vision models to llamacpp based on backend setting.
	// Returns the override provider and name, or nil/"" to use the default.
	GetProviderOverride func(provider, model string) (ports.ChatProvider, string)

	// CheckBudget verifies if a request should be allowed based on budget limits.
	// Returns an error if budget is exceeded, nil otherwise.
	CheckBudget func(provider string) error

	// Timeouts
	OpenAIStreamingTimeout int // Timeout in seconds for OpenAI streaming requests

	// OllamaHost for streaming requests (e.g., "localhost:11434")
	OllamaHost string
}

// MetricsRecorder interface for recording request metrics.
type MetricsRecorder interface {
	RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool)
}

const statusClientClosedRequest = 499

func isContextCanceled(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
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

// normalizeNVRProxyModel forces scheduled/manual NVR callers off llava/gemma and onto qwen3-vl:30b
func normalizeNVRProxyModel(model, usecase string) string {
	if usecase != "nvr-proxy-scheduled" && usecase != "nvr-proxy-analyze" && usecase != "nvr-proxy-manual" {
		return model
	}
	m := strings.TrimSpace(model)
	l := strings.ToLower(m)
	if l == "" {
		return m
	}
	if strings.Contains(l, "gemma") || strings.Contains(l, "llava") {
		return "ollama/qwen3-vl:30b"
	}
	return m
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

	// Allow usecase from Bearer token (for clients like Jan that only support API key)
	if req.Usecase == "" {
		req.Usecase = usecaseFromBearer(r)
	}

	// Validate usecase is provided
	if req.Usecase == "" {
		http.Error(w, "Missing required field: usecase. Please provide a usecase to identify the caller (body field or X-Usecase header).", http.StatusBadRequest)
		return
	}

	// Rewrite legacy/banned models for nvr-proxy callers to qwen3-vl:30b
	originalModel := req.Model
	req.Model = normalizeNVRProxyModel(req.Model, req.Usecase)

	startTime := time.Now()
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestedModel: originalModel,
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
	if err == nil && route.Provider == "openai" {
		// Configurable via env for now; will be exposed in the UI settings next.
		if v := strings.ToLower(strings.TrimSpace(r.Header.Get("X-LLM-Proxy-Strip-Reasoning-Summary"))); v == "1" || v == "true" || v == "yes" {
			req.StripReasoningSummary = true
		}
	}
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

	// Check budget before processing request
	if h.CheckBudget != nil {
		actualProviderName := route.Provider
		// Check for provider override to get the actual provider name
		if h.GetProviderOverride != nil {
			if override, overrideName := h.GetProviderOverride(route.Provider, route.Model); override != nil {
				actualProviderName = overrideName
			}
		}
		if err := h.CheckBudget(actualProviderName); err != nil {
			logEntry.Success = false
			logEntry.Error = err.Error()
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			h.Logger.LogRequest(logEntry)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusPaymentRequired)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "budget_exceeded",
					"code":    "budget_exceeded",
				},
			})
			return
		}
	}

	// Handle streaming requests
	if req.Stream {
		reqCtx, cancel := context.WithCancel(r.Context())
		defer cancel()

		pendingID := ""
		if h.AddPending != nil {
			pendingID = h.AddPending(&req, route, startTime, cancel)
			defer func() {
				if pendingID != "" && h.RemovePending != nil {
					h.RemovePending(pendingID)
				}
			}()
		}

		h.handleStreaming(w, r, reqCtx, &req, route, body, logEntry, startTime)
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
	reqCtx, cancel := context.WithCancel(r.Context())
	defer cancel()

	pendingID := ""
	if h.AddPending != nil {
		pendingID = h.AddPending(&req, route, startTime, cancel)
		defer func() {
			if pendingID != "" && h.RemovePending != nil {
				h.RemovePending(pendingID)
			}
		}()
	}

	// Make the actual call
	var resp *domain.ChatCompletionResponse
	actualProviderName := route.Provider
	provider, ok := h.Providers[route.Provider]
	if !ok {
		err = fmt.Errorf("unknown provider: %s", route.Provider)
	} else {
		// Check for provider override (e.g., llamacpp instead of ollama for vision)
		if h.GetProviderOverride != nil {
			if override, overrideName := h.GetProviderOverride(route.Provider, route.Model); override != nil {
				provider = override
				actualProviderName = overrideName
				logEntry.Provider = overrideName
			}
		}
		resp, err = provider.Chat(reqCtx, &req, route.Model)
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()

	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()

		h.Logger.LogRequest(logEntry)
		h.Metrics.RecordRequest(actualProviderName, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)

		statusCode := http.StatusInternalServerError
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		http.Error(w, err.Error(), statusCode)
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
	h.Metrics.RecordRequest(actualProviderName, route.Model, "success", logEntry.LatencyMs, logEntry.InputTokens, logEntry.OutputTokens, logEntry.CostUSD, false)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", actualProviderName)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", logEntry.LatencyMs))
	w.Header().Set("X-LLM-Proxy-Cost-USD", fmt.Sprintf("%.6f", logEntry.CostUSD))
	if requestID > 0 {
		w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
	}
	json.NewEncoder(w).Encode(resp)
}

// handleStreaming handles streaming chat completion requests by forwarding to OpenAI/Together/Ollama/Baseten with stream=true.
func (h *ChatHandler) handleStreaming(w http.ResponseWriter, r *http.Request, ctx context.Context, req *domain.ChatCompletionRequest, route *domain.RouteConfig, body []byte, logEntry *domain.RequestLog, startTime time.Time) {
	// Check if provider supports streaming
	if route.Provider != "openai" && route.Provider != "together" && route.Provider != "baseten" && route.Provider != "ollama" && route.Provider != "ollama-cloud" && route.Provider != "mlx" && route.Provider != "llamacpp" {
		http.Error(w, fmt.Sprintf("Streaming not supported for provider: %s", route.Provider), http.StatusBadRequest)
		return
	}

	// Handle Ollama streaming separately (uses NDJSON, not SSE)
	if route.Provider == "mlx" || route.Provider == "llamacpp" {
		h.handleMLXStreaming(w, r, req, route, logEntry, startTime)
		return
	}

	if route.Provider == "ollama" || route.Provider == "ollama-cloud" {
		h.handleOllamaStreaming(w, r, ctx, req, route, logEntry, startTime)
		return
	}

	// Check budget before processing streaming request
	if h.CheckBudget != nil {
		actualProviderName := route.Provider
		// Check for provider override to get the actual provider name
		if h.GetProviderOverride != nil {
			if override, overrideName := h.GetProviderOverride(route.Provider, route.Model); override != nil {
				actualProviderName = overrideName
			}
		}
		if err := h.CheckBudget(actualProviderName); err != nil {
			logEntry.Success = false
			logEntry.Error = err.Error()
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			h.Logger.LogRequest(logEntry)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusPaymentRequired)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "budget_exceeded",
					"code":    "budget_exceeded",
				},
			})
			return
		}
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

	// Get provider to access API key
	provider, ok := h.Providers[route.Provider]
	if !ok {
		http.Error(w, fmt.Sprintf("%s provider not configured", route.Provider), http.StatusInternalServerError)
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

	// Choose endpoint based on provider
	endpoint := "https://api.openai.com/v1/chat/completions"
	if route.Provider == "together" {
		endpoint = "https://api.together.xyz/v1/chat/completions"
	} else if route.Provider == "baseten" {
		endpoint = "https://inference.baseten.co/v1/chat/completions"
	}

	// Make streaming request
	httpReq, _ := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(reqBody))
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: time.Duration(h.OpenAIStreamingTimeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		statusCode := http.StatusBadGateway
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		http.Error(w, fmt.Sprintf("%s request failed: %s", route.Provider, err.Error()), statusCode)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("%s error %d: %s", route.Provider, resp.StatusCode, string(respBody))
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

	if err := scanner.Err(); err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		h.Metrics.RecordRequest(route.Provider, route.Model, "error", logEntry.LatencyMs, inputTokens, outputTokens, 0, false)
		return
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

// ============================================================================
// Ollama Streaming Handler
// ============================================================================

// ollamaStreamRequest matches Ollama's /api/chat request format.
type ollamaStreamRequest struct {
	Model      string                `json:"model"`
	Messages   []ollamaStreamMessage `json:"messages"`
	Stream     bool                  `json:"stream"`
	Think      *bool                 `json:"think,omitempty"` // Disable thinking mode for qwen3.5 etc.
	Tools      []domain.Tool         `json:"tools,omitempty"`
	ToolChoice interface{}           `json:"tool_choice,omitempty"`
	KeepAlive  string                `json:"keep_alive,omitempty"`
	Options    *ollamaStreamOptions  `json:"options,omitempty"`
}

type ollamaStreamMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ollamaStreamOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

// ollamaStreamToolCall matches Ollama's tool_call format in streaming responses.
type ollamaStreamToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Index     int                    `json:"index,omitempty"`
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

// ollamaStreamChunk is a single chunk from Ollama's NDJSON streaming response.
type ollamaStreamChunk struct {
	Model   string `json:"model"`
	Message struct {
		Role      string                 `json:"role"`
		Content   string                 `json:"content"`
		Thinking  string                 `json:"thinking,omitempty"`
		ToolCalls []ollamaStreamToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done            bool `json:"done"`
	PromptEvalCount int  `json:"prompt_eval_count,omitempty"`
	EvalCount       int  `json:"eval_count,omitempty"`
}

// handleOllamaStreaming handles streaming requests to Ollama.
// Ollama uses NDJSON streaming, which we convert to OpenAI-compatible SSE format.
func (h *ChatHandler) handleOllamaStreaming(w http.ResponseWriter, r *http.Request, ctx context.Context, req *domain.ChatCompletionRequest, route *domain.RouteConfig, logEntry *domain.RequestLog, startTime time.Time) {
	if route.Provider == "ollama" && providers.IsOllamaCloudModel(route.Model) {
		err := fmt.Errorf("cloud model %q must use ollama-cloud provider", route.Model)
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Check budget before processing
	if h.CheckBudget != nil {
		if err := h.CheckBudget(route.Provider); err != nil {
			logEntry.Success = false
			logEntry.Error = err.Error()
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			h.Logger.LogRequest(logEntry)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusPaymentRequired)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "budget_exceeded",
					"code":    "budget_exceeded",
				},
			})
			return
		}
	}

	// Convert messages to Ollama format
	var messages []ollamaStreamMessage
	for _, msg := range req.Messages {
		ollamaMsg := ollamaStreamMessage{Role: msg.Role}

		switch c := msg.Content.(type) {
		case string:
			ollamaMsg.Content = c
		case []interface{}:
			// Handle multimodal content
			var textParts []string
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "text" {
						if text, ok := m["text"].(string); ok {
							textParts = append(textParts, text)
						}
					} else if m["type"] == "image_url" {
						if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
							if url, ok := imgURL["url"].(string); ok {
								// Extract base64 from data URL
								if strings.HasPrefix(url, "data:") {
									parts := strings.SplitN(url, ",", 2)
									if len(parts) == 2 {
										ollamaMsg.Images = append(ollamaMsg.Images, parts[1])
									}
								}
							}
						}
					}
				}
			}
			ollamaMsg.Content = strings.Join(textParts, "\n")
		}

		messages = append(messages, ollamaMsg)
	}

	// Build Ollama streaming request
	// Don't set num_ctx - let Ollama use whatever context the model was loaded with
	// Setting a larger context than loaded causes Ollama to hang while reallocating
	// Use request's Think setting, default to false for faster responses
	thinkFalse := false
	think := &thinkFalse
	if req.Think != nil {
		think = req.Think
	}
	ollamaReq := ollamaStreamRequest{
		Model:      route.Model,
		Messages:   messages,
		Stream:     true, // Enable streaming
		Think:      think,
		Tools:      req.Tools,
		ToolChoice: req.ToolChoice,
		KeepAlive:  "30m",
		Options: &ollamaStreamOptions{
			Temperature: 0.3,
		},
	}

	reqBody, _ := json.Marshal(ollamaReq)
	log.Printf("Ollama streaming request to model: %s", route.Model)

	// Make streaming request to Ollama
	ollamaHost := h.OllamaHost
	if ollamaHost == "" {
		ollamaHost = "localhost:11434"
	}

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", "http://"+ollamaHost+"/api/chat", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	// Use a long timeout for streaming (10 minutes)
	client := &http.Client{Timeout: 10 * time.Minute}
	log.Printf("Ollama streaming: making request to http://%s/api/chat", ollamaHost)
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Ollama streaming: request failed: %v", err)
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		statusCode := http.StatusBadGateway
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		http.Error(w, fmt.Sprintf("Ollama request failed: %s", err.Error()), statusCode)
		return
	}
	defer resp.Body.Close()

	log.Printf("Ollama streaming: got response status %d", resp.StatusCode)

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("Ollama streaming: error response: %s", string(respBody))
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("Ollama error %d: %s", resp.StatusCode, string(respBody))
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		http.Error(w, string(respBody), resp.StatusCode)
		return
	}

	// Set up SSE response (convert Ollama NDJSON to OpenAI SSE)
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

	// Read Ollama NDJSON stream and convert to SSE
	var responseContent strings.Builder
	var thinkingContent strings.Builder
	var inputTokens, outputTokens int
	var hasToolCalls bool
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	responseID := fmt.Sprintf("%s-%d", route.Provider, time.Now().UnixNano())

	log.Printf("Ollama streaming: starting to read NDJSON stream")
	chunkCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		chunkCount++
		if chunkCount <= 3 {
			log.Printf("Ollama streaming: chunk %d (first 100 chars): %.100s", chunkCount, line)
		}

		var chunk ollamaStreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			log.Printf("Failed to parse Ollama chunk: %v", err)
			continue
		}

		// Build delta for OpenAI SSE format
		delta := map[string]interface{}{}

		// Handle text content
		if chunk.Message.Content != "" {
			delta["content"] = chunk.Message.Content
			responseContent.WriteString(chunk.Message.Content)
		}

		// Handle thinking content
		if chunk.Message.Thinking != "" {
			thinkingContent.WriteString(chunk.Message.Thinking)
		}

		// Handle tool calls - convert Ollama format to OpenAI format
		if len(chunk.Message.ToolCalls) > 0 {
			log.Printf("Ollama streaming: got %d tool calls", len(chunk.Message.ToolCalls))
			hasToolCalls = true
			var openAIToolCalls []map[string]interface{}
			for i, tc := range chunk.Message.ToolCalls {
				// Convert arguments map to JSON string (OpenAI format)
				argsJSON, _ := json.Marshal(tc.Function.Arguments)
				toolCallID := tc.ID
				if toolCallID == "" {
					toolCallID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i)
				}
				openAIToolCalls = append(openAIToolCalls, map[string]interface{}{
					"index": i,
					"id":    toolCallID,
					"type":  "function",
					"function": map[string]string{
						"name":      tc.Function.Name,
						"arguments": string(argsJSON),
					},
				})
			}
			delta["tool_calls"] = openAIToolCalls
		}

		// Only send chunk if there's content or tool calls
		if len(delta) > 0 {
			sseChunk := map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   chunk.Model,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": delta,
					},
				},
			}
			chunkJSON, _ := json.Marshal(sseChunk)
			fmt.Fprintf(w, "data: %s\n\n", chunkJSON)
			flusher.Flush()
		}

		// Capture token counts from final chunk
		if chunk.Done {
			inputTokens = chunk.PromptEvalCount
			outputTokens = chunk.EvalCount

			// Determine finish reason
			finishReason := "stop"
			if hasToolCalls {
				finishReason = "tool_calls"
			}

			// Send final chunk with finish_reason
			finalChunk := map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   chunk.Model,
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"delta":         map[string]interface{}{},
						"finish_reason": finishReason,
					},
				},
				"usage": map[string]int{
					"prompt_tokens":     inputTokens,
					"completion_tokens": outputTokens,
				},
			}
			finalJSON, _ := json.Marshal(finalChunk)
			fmt.Fprintf(w, "data: %s\n\n", finalJSON)
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.Logger.LogRequest(logEntry)
		h.Metrics.RecordRequest(route.Provider, route.Model, "error", logEntry.LatencyMs, inputTokens, outputTokens, 0, false)
		return
	}

	// Send [DONE] marker
	fmt.Fprintln(w, "data: [DONE]")
	flusher.Flush()

	// Prepend thinking content if present
	fullContent := responseContent.String()
	if thinkingContent.Len() > 0 {
		fullContent = "<think>" + thinkingContent.String() + "</think>" + fullContent
	}

	// Build synthetic response for logging
	syntheticResp := map[string]interface{}{
		"choices": []map[string]interface{}{
			{"message": map[string]string{"role": "assistant", "content": fullContent}},
		},
		"usage": map[string]int{"prompt_tokens": inputTokens, "completion_tokens": outputTokens},
	}
	respBytes, _ := json.Marshal(syntheticResp)

	// Log success
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

// ============================================================================
// Responses API Handler (OpenAI's newer API for reasoning models, web search)
// ============================================================================

// ResponsesHandler handles OpenAI Responses API requests with smart routing.
type ResponsesHandler struct {
	ChatHandler   *ChatHandler            // Reuse chat handler for translated requests
	OpenAIKey     string                  // API key for direct OpenAI forwarding
	Mode          domain.ResponsesAPIMode // auto, openai, or translate
	Logger        ports.RequestLogger
	Metrics       MetricsRecorder
	CalculateCost func(model string, inputTokens, outputTokens int) float64
}

// ServeHTTP handles Responses API requests.
func (h *ResponsesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req domain.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Allow usecase from header
	if req.Usecase == "" {
		req.Usecase = r.Header.Get("X-Usecase")
	}

	// Allow usecase from Bearer token
	if req.Usecase == "" {
		req.Usecase = usecaseFromBearer(r)
	}

	// Validate usecase
	if req.Usecase == "" {
		h.jsonError(w, http.StatusBadRequest, "missing_usecase",
			"Missing required field: usecase. Please provide a usecase to identify the caller (body field or X-Usecase header).")
		return
	}

	startTime := time.Now()

	// Determine routing mode
	mode := h.Mode
	if modeHeader := r.Header.Get("X-Responses-API-Mode"); modeHeader != "" {
		switch modeHeader {
		case "openai":
			mode = domain.ResponsesAPIModeOpenAI
		case "translate":
			mode = domain.ResponsesAPIModeTranslate
		case "auto":
			mode = domain.ResponsesAPIModeAuto
		}
	}

	// Check if request requires OpenAI
	requiresOpenAI, reason := req.RequiresOpenAI()

	// Streaming requires OpenAI (translation of streaming is complex)
	if req.Stream && !requiresOpenAI {
		requiresOpenAI = true
		reason = "streaming requires OpenAI Responses API (translation not supported)"
	}

	log.Printf("[Responses API] mode=%s, requiresOpenAI=%v (%s), model=%s, usecase=%s, stream=%v",
		mode, requiresOpenAI, reason, req.Model, req.Usecase, req.Stream)

	// Route based on mode and requirements
	switch mode {
	case domain.ResponsesAPIModeOpenAI:
		// Always forward to OpenAI
		h.forwardToOpenAI(w, r, body, &req, startTime)

	case domain.ResponsesAPIModeTranslate:
		// Always translate - fail if OpenAI-specific features needed
		if requiresOpenAI {
			h.jsonError(w, http.StatusBadRequest, "translation_impossible",
				fmt.Sprintf("Cannot translate to Chat Completions: %s. Use mode=openai or mode=auto.", reason))
			return
		}
		h.translateAndRoute(w, r, &req, startTime)

	case domain.ResponsesAPIModeAuto:
		fallthrough
	default:
		// Smart routing
		if requiresOpenAI {
			log.Printf("[Responses API] Auto-routing to OpenAI: %s", reason)
			h.forwardToOpenAI(w, r, body, &req, startTime)
		} else {
			log.Printf("[Responses API] Auto-routing via translation to Chat Completions")
			h.translateAndRoute(w, r, &req, startTime)
		}
	}
}

// forwardToOpenAI forwards the request directly to OpenAI's Responses API.
func (h *ResponsesHandler) forwardToOpenAI(w http.ResponseWriter, r *http.Request, body []byte, req *domain.ResponsesRequest, startTime time.Time) {
	if h.OpenAIKey == "" {
		h.jsonError(w, http.StatusInternalServerError, "no_api_key",
			"OpenAI API key not configured for Responses API forwarding")
		return
	}

	// Preserve original body for logging (before any modifications)
	originalBody := body

	// Strip reasoning.summary - requires org verification we don't have
	var reqMap map[string]interface{}
	if err := json.Unmarshal(body, &reqMap); err == nil {
		if reasoning, ok := reqMap["reasoning"].(map[string]interface{}); ok {
			if _, hasSummary := reasoning["summary"]; hasSummary {
				delete(reasoning, "summary")
				log.Printf("[Responses API] WARNING: Stripped reasoning.summary (requires OpenAI org verification)")
				// Add header so client knows we modified the request
				w.Header().Set("X-LLM-Proxy-Warning", "reasoning.summary stripped")
			}
			if len(reasoning) == 0 {
				delete(reqMap, "reasoning")
			}
			body, _ = json.Marshal(reqMap)
		}
	}

	log.Printf("[Responses API] Forwarding to OpenAI: model=%s, stream=%v", req.Model, req.Stream)

	// Forward to OpenAI
	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", "https://api.openai.com/v1/responses", bytes.NewReader(body))
	if err != nil {
		h.jsonError(w, http.StatusInternalServerError, "request_creation_failed", err.Error())
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+h.OpenAIKey)

	client := &http.Client{Timeout: time.Duration(h.ChatHandler.OpenAIStreamingTimeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		statusCode := http.StatusBadGateway
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		h.jsonError(w, statusCode, "openai_request_failed", err.Error())
		return
	}
	defer resp.Body.Close()

	// Log the request (use original body to preserve tool descriptions, etc.)
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestType:    "responses",
		RequestedModel: req.Model,
		Provider:       "openai",
		Model:          req.Model,
		Usecase:        req.Usecase,
		RequestBody:    originalBody,
		ClientIP:       getClientIP(r),
	}
	if req.Sensitive != nil {
		logEntry.Sensitive = *req.Sensitive
	}

	// Handle streaming responses
	if req.Stream {
		h.streamOpenAIResponse(w, resp, logEntry, startTime)
		return
	}

	// Non-streaming: read and forward response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		h.jsonError(w, http.StatusBadGateway, "response_read_failed", err.Error())
		return
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.ResponseBody = respBody

	// Try to extract usage from response
	var respData domain.ResponsesResponse
	if json.Unmarshal(respBody, &respData) == nil && respData.Usage != nil {
		logEntry.InputTokens = respData.Usage.InputTokens
		logEntry.OutputTokens = respData.Usage.OutputTokens
		logEntry.CostUSD = h.CalculateCost(req.Model, logEntry.InputTokens, logEntry.OutputTokens)
	}

	logEntry.Success = resp.StatusCode == 200
	if resp.StatusCode != 200 {
		logEntry.Error = string(respBody)
	}

	h.Logger.LogRequest(logEntry)
	h.Metrics.RecordRequest("openai", req.Model, statusString(logEntry.Success), logEntry.LatencyMs,
		logEntry.InputTokens, logEntry.OutputTokens, logEntry.CostUSD, false)

	// Forward response
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", "openai")
	w.Header().Set("X-LLM-Proxy-Model", req.Model)
	w.Header().Set("X-LLM-Proxy-Responses-Mode", "openai")
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)
}

// streamOpenAIResponse streams the OpenAI Responses API response to the client.
func (h *ResponsesHandler) streamOpenAIResponse(w http.ResponseWriter, resp *http.Response, logEntry *domain.RequestLog, startTime time.Time) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-LLM-Proxy-Provider", "openai")
	w.Header().Set("X-LLM-Proxy-Responses-Mode", "openai")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Stream through while capturing data for logging
	var responseBuilder strings.Builder
	var inputTokens, outputTokens int
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		fmt.Fprintln(w, line)
		flusher.Flush()

		// Try to extract usage from events
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data != "[DONE]" {
				var event map[string]interface{}
				if json.Unmarshal([]byte(data), &event) == nil {
					// Try direct usage field (some formats)
					if usage, ok := event["usage"].(map[string]interface{}); ok {
						if it, ok := usage["input_tokens"].(float64); ok {
							inputTokens = int(it)
						}
						if ot, ok := usage["output_tokens"].(float64); ok {
							outputTokens = int(ot)
						}
					}
					// Try response.usage (Responses API response.done event)
					if response, ok := event["response"].(map[string]interface{}); ok {
						if usage, ok := response["usage"].(map[string]interface{}); ok {
							if it, ok := usage["input_tokens"].(float64); ok {
								inputTokens = int(it)
							}
							if ot, ok := usage["output_tokens"].(float64); ok {
								outputTokens = int(ot)
							}
						}
					}
				}
			}
		}
		responseBuilder.WriteString(line + "\n")
	}

	if err := scanner.Err(); err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.RequestType = "responses-stream"
		logEntry.InputTokens = inputTokens
		logEntry.OutputTokens = outputTokens
		logEntry.ResponseBody = []byte(responseBuilder.String())
		h.Logger.LogRequest(logEntry)
		h.Metrics.RecordRequest("openai", logEntry.Model, "error", logEntry.LatencyMs, inputTokens, outputTokens, 0, false)
		return
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.Success = true
	logEntry.RequestType = "responses-stream"
	logEntry.InputTokens = inputTokens
	logEntry.OutputTokens = outputTokens
	logEntry.ResponseBody = []byte(responseBuilder.String())
	h.Logger.LogRequest(logEntry)
	h.Metrics.RecordRequest("openai", logEntry.Model, "success", logEntry.LatencyMs, inputTokens, outputTokens, 0, false)
}

// translateAndRoute translates the request to Chat Completions and routes through normal routing.
func (h *ResponsesHandler) translateAndRoute(w http.ResponseWriter, r *http.Request, req *domain.ResponsesRequest, startTime time.Time) {
	// Convert to Chat Completions request
	chatReq := req.ToChatCompletionRequest()

	log.Printf("[Responses API] Translated to Chat Completions: model=%s, messages=%d, tools=%d",
		chatReq.Model, len(chatReq.Messages), len(chatReq.Tools))

	// Serialize and create new request
	chatBody, _ := json.Marshal(chatReq)

	// Create a new HTTP request with the translated body
	newReq, _ := http.NewRequestWithContext(r.Context(), "POST", r.URL.String(), bytes.NewReader(chatBody))
	newReq.Header = r.Header.Clone()
	newReq.Header.Set("Content-Type", "application/json")
	newReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(chatBody)))

	// Use a response recorder to capture the chat response
	recorder := &responseRecorder{
		header: make(http.Header),
		body:   &bytes.Buffer{},
	}

	// Call the chat handler
	h.ChatHandler.ServeHTTP(recorder, newReq)

	// If streaming, we need to pass through directly (can't easily convert)
	if req.Stream {
		// For streaming, just pass through the chat completions SSE format
		// Client will need to handle it (or we'd need complex translation)
		w.Header().Set("X-LLM-Proxy-Responses-Mode", "translate-passthrough")
		for k, v := range recorder.header {
			w.Header()[k] = v
		}
		w.WriteHeader(recorder.code)
		w.Write(recorder.body.Bytes())
		return
	}

	// Convert response back to Responses API format
	if recorder.code != http.StatusOK {
		w.Header().Set("X-LLM-Proxy-Responses-Mode", "translate")
		for k, v := range recorder.header {
			w.Header()[k] = v
		}
		w.WriteHeader(recorder.code)
		w.Write(recorder.body.Bytes())
		return
	}

	var chatResp domain.ChatCompletionResponse
	if err := json.Unmarshal(recorder.body.Bytes(), &chatResp); err != nil {
		h.jsonError(w, http.StatusInternalServerError, "response_parse_failed",
			"Failed to parse Chat Completions response: "+err.Error())
		return
	}

	// Convert to Responses API format
	responsesResp := domain.ChatCompletionToResponsesResponse(&chatResp)

	// Copy headers and send
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Responses-Mode", "translate")
	if provider := recorder.header.Get("X-LLM-Proxy-Provider"); provider != "" {
		w.Header().Set("X-LLM-Proxy-Provider", provider)
	}
	if model := recorder.header.Get("X-LLM-Proxy-Model"); model != "" {
		w.Header().Set("X-LLM-Proxy-Model", model)
	}

	json.NewEncoder(w).Encode(responsesResp)
}

// jsonError writes a JSON error response in OpenAI format.
func (h *ResponsesHandler) jsonError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	})
}

// statusString returns "success" or "error" based on boolean.
func statusString(success bool) string {
	if success {
		return "success"
	}
	return "error"
}

// responseRecorder captures HTTP response for internal routing.
type responseRecorder struct {
	code        int
	header      http.Header
	body        *bytes.Buffer
	wroteHeader bool
}

func (r *responseRecorder) Header() http.Header {
	return r.header
}

func (r *responseRecorder) Write(b []byte) (int, error) {
	// If WriteHeader was never called, default to 200 OK
	if !r.wroteHeader {
		r.code = http.StatusOK
		r.wroteHeader = true
	}
	return r.body.Write(b)
}

func (r *responseRecorder) WriteHeader(code int) {
	if !r.wroteHeader {
		r.code = code
		r.wroteHeader = true
	}
}

// Flush implements http.Flusher (no-op for buffered recorder)
func (r *responseRecorder) Flush() {
	// No-op - we buffer everything
}

// handleMLXStreaming handles streaming requests to MLX LM server.
// MLX uses OpenAI-compatible SSE streaming format with data: prefixes.
// handleMLXStreaming handles streaming requests to MLX LM server.
// MLX uses OpenAI-compatible SSE streaming format with data: prefixes.
// This handler cleans up MLX-specific fields (empty tool_calls, reasoning)
// that cause parsing errors in LibreChat.
func (h *ChatHandler) handleMLXStreaming(w http.ResponseWriter, r *http.Request, req *domain.ChatCompletionRequest, route *domain.RouteConfig, logEntry *domain.RequestLog, startTime time.Time) {
	// Resolve host and timeout based on provider (works for both mlx and llamacpp)
	var host string
	var timeout int
	switch route.Provider {
	case "llamacpp":
		p, ok := h.Providers["llamacpp"]
		if !ok {
			http.Error(w, "llamacpp provider not configured", http.StatusInternalServerError)
			return
		}
		lp, ok := p.(*providers.LlamaCppProvider)
		if !ok {
			http.Error(w, "Invalid llamacpp provider type", http.StatusInternalServerError)
			return
		}
		host = lp.Host
		timeout = lp.Timeout
	default: // mlx
		p, ok := h.Providers["mlx"]
		if !ok {
			http.Error(w, "MLX provider not configured", http.StatusInternalServerError)
			return
		}
		mp, ok := p.(*providers.MLXProvider)
		if !ok {
			http.Error(w, "Invalid MLX provider type", http.StatusInternalServerError)
			return
		}
		host = mp.Host
		timeout = mp.Timeout
	}

	// Build streaming request (OpenAI-compatible, works for both MLX and llama.cpp)
	streamReq := map[string]interface{}{
		"model":    route.Model,
		"messages": req.Messages,
		"stream":   true,
	}
	if req.MaxTokens > 0 {
		streamReq["max_tokens"] = req.MaxTokens
	}
	if req.MaxCompletionTokens > 0 {
		streamReq["max_tokens"] = req.MaxCompletionTokens
	}
	if req.Temperature > 0 {
		streamReq["temperature"] = req.Temperature
	}
	if len(req.Tools) > 0 {
		streamReq["tools"] = req.Tools
	}
	// Always request usage in final chunk so clients can track context
	streamReq["stream_options"] = map[string]bool{"include_usage": true}

	body, _ := json.Marshal(streamReq)
	url := "http://" + host + "/v1/chat/completions"

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", url, bytes.NewReader(body))
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		http.Error(w, "Failed to create request", http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: time.Duration(timeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		http.Error(w, "MLX request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("MLX error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, logEntry.Error, resp.StatusCode)
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Stream response from MLX server, cleaning up incompatible fields
	scanner := bufio.NewScanner(resp.Body)
	var totalContent string
	var reasoningContent string
	var inputTokens, outputTokens int

	// Accumulate tool calls from streaming chunks
	type toolCall struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	var toolCalls []toolCall
	toolCallArgs := map[int]*strings.Builder{}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data != "[DONE]" {
				var chunk map[string]interface{}
				if err := json.Unmarshal([]byte(data), &chunk); err == nil {
					if choices, ok := chunk["choices"].([]interface{}); ok {
						for _, choice := range choices {
							if c, ok := choice.(map[string]interface{}); ok {
								if delta, ok := c["delta"].(map[string]interface{}); ok {
									// Remove empty tool_calls array
									if tc, ok := delta["tool_calls"].([]interface{}); ok && len(tc) == 0 {
										delete(delta, "tool_calls")
									}
									// Remove empty reasoning string
									if r, ok := delta["reasoning"].(string); ok && r == "" {
										delete(delta, "reasoning")
									}
									// Extract content
									if content, ok := delta["content"].(string); ok {
										totalContent += content
									}
									// Extract reasoning_content
									if rc, ok := delta["reasoning_content"].(string); ok {
										reasoningContent += rc
									}
									// Accumulate tool calls
									if tcs, ok := delta["tool_calls"].([]interface{}); ok {
										for _, tcRaw := range tcs {
											tc, ok := tcRaw.(map[string]interface{})
											if !ok {
												continue
											}
											idx := 0
											if i, ok := tc["index"].(float64); ok {
												idx = int(i)
											}
											fn, _ := tc["function"].(map[string]interface{})
											if fn == nil {
												continue
											}
											name, _ := fn["name"].(string)
											args, _ := fn["arguments"].(string)
											id, _ := tc["id"].(string)

											if id != "" && name != "" {
												// New tool call
												toolCalls = append(toolCalls, toolCall{
													ID:   id,
													Type: "function",
												})
												toolCalls[len(toolCalls)-1].Function.Name = name
												idx = len(toolCalls) - 1
												toolCallArgs[idx] = &strings.Builder{}
												if args != "" {
													toolCallArgs[idx].WriteString(args)
												}
											} else if args != "" && idx < len(toolCalls) {
												// Append arguments to existing tool call
												if _, ok := toolCallArgs[idx]; !ok {
													toolCallArgs[idx] = &strings.Builder{}
												}
												toolCallArgs[idx].WriteString(args)
											}
										}
									}
								}
							}
						}
					}
					// Extract usage
					if usage, ok := chunk["usage"].(map[string]interface{}); ok {
						if pt, ok := usage["prompt_tokens"].(float64); ok {
							inputTokens = int(pt)
						}
						if ct, ok := usage["completion_tokens"].(float64); ok {
							outputTokens = int(ct)
						}
					}
					cleanedData, _ := json.Marshal(chunk)
					line = "data: " + string(cleanedData)
				}
			}
		}

		fmt.Fprintf(w, "%s\n\n", line)
		flusher.Flush()
	}

	// Finalize tool call arguments
	for i, builder := range toolCallArgs {
		if i < len(toolCalls) {
			toolCalls[i].Function.Arguments = builder.String()
		}
	}

	// Build synthetic response body for logging
	message := map[string]interface{}{
		"role":    "assistant",
		"content": totalContent,
	}
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	syntheticResp := map[string]interface{}{
		"model":    route.Model,
		"provider": route.Provider,
		"choices": []map[string]interface{}{
			{
				"message":       message,
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     inputTokens,
			"completion_tokens": outputTokens,
			"total_tokens":      inputTokens + outputTokens,
		},
	}
	respBytes, _ := json.Marshal(syntheticResp)
	logEntry.ResponseBody = respBytes

	// Log the request
	logEntry.Success = true
	logEntry.InputTokens = inputTokens
	logEntry.OutputTokens = outputTokens
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	h.Logger.LogRequest(logEntry)
}
