package http

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/domain"
)

// AnthropicMessagesHandler exposes an Anthropic-compatible /v1/messages endpoint.
// Requests stay in Anthropic Messages format when the resolved upstream supports it.
type AnthropicMessagesHandler struct {
	ChatHandler *ChatHandler

	AnthropicKey     string
	AnthropicBaseURL string
	OllamaBaseURL    string
	AnthropicTimeout int
	OllamaTimeout    int
}

// ServeHTTP handles Anthropic-compatible /v1/messages requests.
func (h *AnthropicMessagesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		h.jsonError(w, http.StatusMethodNotAllowed, "invalid_request_error", "Method not allowed")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		h.jsonError(w, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}

	var req domain.AnthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		h.jsonError(w, http.StatusBadRequest, "invalid_request_error", "Invalid JSON: "+err.Error())
		return
	}

	if req.Usecase == "" {
		req.Usecase = r.Header.Get("X-Usecase")
	}
	if req.Usecase == "" {
		req.Usecase = usecaseFromBearer(r)
	}
	if req.Usecase == "" {
		h.jsonError(w, http.StatusBadRequest, "invalid_request_error",
			"Missing required field: usecase. Please provide a usecase in the body or X-Usecase header.")
		return
	}

	chatReq := req.ToChatCompletionRequest()
	startTime := time.Now()
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestType:    "messages",
		RequestedModel: req.Model,
		Precision:      req.Precision,
		Usecase:        req.Usecase,
		HasImages:      chatReq.HasImages(),
		RequestBody:    body,
		ClientIP:       getClientIP(r),
	}
	if req.Sensitive != nil {
		logEntry.Sensitive = *req.Sensitive
	}

	route, err := h.ChatHandler.Router.ResolveRoute(chatReq)
	if err != nil {
		logEntry.Provider = "routing_failed"
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.jsonError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if h.ChatHandler.IsModelDisabled != nil && h.ChatHandler.IsModelDisabled(route.Model) {
		logEntry.Provider = route.Provider
		logEntry.Model = route.Model
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("model %s is disabled", route.Model)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.jsonError(w, http.StatusServiceUnavailable, "api_error", fmt.Sprintf("Model %s is currently disabled", route.Model))
		return
	}

	actualProviderName := route.Provider
	if h.ChatHandler.GetProviderOverride != nil {
		if override, overrideName := h.ChatHandler.GetProviderOverride(route.Provider, route.Model); override != nil {
			actualProviderName = overrideName
		}
	}

	logEntry.Provider = actualProviderName
	logEntry.Model = route.Model

	if h.ChatHandler.CheckBudget != nil {
		if err := h.ChatHandler.CheckBudget(actualProviderName); err != nil {
			logEntry.Success = false
			logEntry.Error = err.Error()
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			h.ChatHandler.Logger.LogRequest(logEntry)
			h.jsonError(w, http.StatusPaymentRequired, "budget_exceeded", err.Error())
			return
		}
	}

	if !h.supportsNativeMessages(actualProviderName) {
		if req.Stream {
			logEntry.Success = false
			logEntry.Error = fmt.Sprintf("streaming translation to %s is not supported", actualProviderName)
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			h.ChatHandler.Logger.LogRequest(logEntry)
			h.jsonError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("Streaming Anthropic Messages requests cannot be translated to %s yet", actualProviderName))
			return
		}
		h.translateAndRoute(w, r, chatReq)
		return
	}

	cacheKey := h.generateMessagesCacheKey(body, route)
	logEntry.CacheKey = cacheKey

	if !req.Stream && !req.NoCache {
		if cached, ok := h.ChatHandler.Cache.Get(cacheKey); ok {
			logEntry.Cached = true
			logEntry.Success = true
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			logEntry.ResponseBody = cached
			requestID := h.ChatHandler.Logger.LogRequest(logEntry)
			h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, "success", logEntry.LatencyMs, 0, 0, 0, true)

			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-LLM-Proxy-Provider", actualProviderName)
			w.Header().Set("X-LLM-Proxy-Model", route.Model)
			w.Header().Set("X-LLM-Proxy-Messages-Mode", "passthrough")
			w.Header().Set("X-LLM-Proxy-Cached", "true")
			if requestID > 0 {
				w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
			}
			_, _ = w.Write(cached)
			return
		}
	}

	pendingID := ""
	reqCtx, cancel := context.WithCancel(r.Context())
	defer cancel()
	if h.ChatHandler.AddPending != nil {
		pendingID = h.ChatHandler.AddPending(chatReq, route, startTime, cancel)
		defer func() {
			if pendingID != "" && h.ChatHandler.RemovePending != nil {
				h.ChatHandler.RemovePending(pendingID)
			}
		}()
	}

	if req.Stream {
		h.streamNativeMessages(w, r, reqCtx, body, req, route, actualProviderName, logEntry, startTime)
		return
	}

	h.forwardNativeMessages(w, r, reqCtx, body, req, route, actualProviderName, logEntry, startTime)
}

func (h *AnthropicMessagesHandler) forwardNativeMessages(
	w http.ResponseWriter,
	r *http.Request,
	ctx context.Context,
	body []byte,
	req domain.AnthropicMessagesRequest,
	route *domain.RouteConfig,
	actualProviderName string,
	logEntry *domain.RequestLog,
	startTime time.Time,
) {
	httpReq, err := h.buildPassthroughRequest(ctx, actualProviderName, route.Model, body, r.Header)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.jsonError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}

	client := &http.Client{Timeout: h.providerTimeout(actualProviderName)}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)
		statusCode := http.StatusBadGateway
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		h.jsonError(w, statusCode, "api_error", err.Error())
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)
		h.jsonError(w, http.StatusBadGateway, "api_error", "Failed to read upstream response")
		return
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.ResponseBody = respBody
	logEntry.Success = resp.StatusCode == http.StatusOK

	if resp.StatusCode == http.StatusOK {
		var upstreamResp domain.AnthropicMessagesResponse
		if json.Unmarshal(respBody, &upstreamResp) == nil {
			logEntry.InputTokens = upstreamResp.Usage.InputTokens
			logEntry.OutputTokens = upstreamResp.Usage.OutputTokens
		}
		logEntry.CostUSD = h.ChatHandler.CalculateCost(route.Model, logEntry.InputTokens, logEntry.OutputTokens)
	} else {
		logEntry.Error = extractErrorMessage(respBody)
	}

	if resp.StatusCode == http.StatusOK && !req.NoCache {
		h.ChatHandler.Cache.Set(logEntry.CacheKey, body, respBody)
	}

	requestID := h.ChatHandler.Logger.LogRequest(logEntry)
	h.ChatHandler.Metrics.RecordRequest(
		actualProviderName,
		route.Model,
		statusString(logEntry.Success),
		logEntry.LatencyMs,
		logEntry.InputTokens,
		logEntry.OutputTokens,
		logEntry.CostUSD,
		false,
	)

	w.Header().Set("Content-Type", contentTypeOrDefault(resp.Header.Get("Content-Type"), "application/json"))
	w.Header().Set("X-LLM-Proxy-Provider", actualProviderName)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Messages-Mode", "passthrough")
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", logEntry.LatencyMs))
	w.Header().Set("X-LLM-Proxy-Cost-USD", fmt.Sprintf("%.6f", logEntry.CostUSD))
	if requestID > 0 {
		w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)
}

func (h *AnthropicMessagesHandler) streamNativeMessages(
	w http.ResponseWriter,
	r *http.Request,
	ctx context.Context,
	body []byte,
	req domain.AnthropicMessagesRequest,
	route *domain.RouteConfig,
	actualProviderName string,
	logEntry *domain.RequestLog,
	startTime time.Time,
) {
	httpReq, err := h.buildPassthroughRequest(ctx, actualProviderName, route.Model, body, r.Header)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.jsonError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}

	client := &http.Client{Timeout: h.providerTimeout(actualProviderName)}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)
		statusCode := http.StatusBadGateway
		if isContextCanceled(err) {
			statusCode = statusClientClosedRequest
		}
		h.jsonError(w, statusCode, "api_error", err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.Success = false
		logEntry.Error = extractErrorMessage(respBody)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.ResponseBody = respBody
		h.ChatHandler.Logger.LogRequest(logEntry)
		h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)

		w.Header().Set("Content-Type", contentTypeOrDefault(resp.Header.Get("Content-Type"), "application/json"))
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(respBody)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.jsonError(w, http.StatusInternalServerError, "api_error", "Streaming not supported by response writer")
		return
	}

	w.Header().Set("Content-Type", contentTypeOrDefault(resp.Header.Get("Content-Type"), "text/event-stream"))
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-LLM-Proxy-Provider", actualProviderName)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Messages-Mode", "passthrough")
	w.WriteHeader(http.StatusOK)

	var responseBuilder strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		_, _ = fmt.Fprintln(w, line)
		flusher.Flush()
		responseBuilder.WriteString(line)
		responseBuilder.WriteByte('\n')
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.Success = scanner.Err() == nil
	if scanner.Err() != nil {
		logEntry.Error = scanner.Err().Error()
	}
	logEntry.ResponseBody = []byte(responseBuilder.String())
	h.ChatHandler.Logger.LogRequest(logEntry)
	h.ChatHandler.Metrics.RecordRequest(actualProviderName, route.Model, statusString(logEntry.Success), logEntry.LatencyMs, 0, 0, 0, false)
}

func (h *AnthropicMessagesHandler) translateAndRoute(w http.ResponseWriter, r *http.Request, chatReq *domain.ChatCompletionRequest) {
	chatBody, _ := json.Marshal(chatReq)
	newReq, _ := http.NewRequestWithContext(r.Context(), "POST", r.URL.String(), bytes.NewReader(chatBody))
	newReq.Header = r.Header.Clone()
	newReq.Header.Set("Content-Type", "application/json")
	newReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(chatBody)))

	recorder := &responseRecorder{
		header: make(http.Header),
		body:   &bytes.Buffer{},
	}

	h.ChatHandler.ServeHTTP(recorder, newReq)

	if recorder.code != http.StatusOK {
		h.jsonError(w, recorder.code, anthropicErrorTypeForStatus(recorder.code), extractErrorMessage(recorder.body.Bytes()))
		return
	}

	var chatResp domain.ChatCompletionResponse
	if err := json.Unmarshal(recorder.body.Bytes(), &chatResp); err != nil {
		h.jsonError(w, http.StatusInternalServerError, "api_error", "Failed to parse translated response: "+err.Error())
		return
	}

	messagesResp := domain.ChatCompletionToAnthropicMessagesResponse(&chatResp)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Messages-Mode", "translate")
	if provider := recorder.header.Get("X-LLM-Proxy-Provider"); provider != "" {
		w.Header().Set("X-LLM-Proxy-Provider", provider)
	}
	if model := recorder.header.Get("X-LLM-Proxy-Model"); model != "" {
		w.Header().Set("X-LLM-Proxy-Model", model)
	}
	_ = json.NewEncoder(w).Encode(messagesResp)
}

func (h *AnthropicMessagesHandler) buildPassthroughRequest(ctx context.Context, provider, model string, body []byte, headers http.Header) (*http.Request, error) {
	sanitizedBody, err := sanitizeMessagesBody(body, model)
	if err != nil {
		return nil, fmt.Errorf("failed to sanitize messages request: %w", err)
	}

	url, err := h.upstreamMessagesURL(provider)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(sanitizedBody))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if accept := headers.Get("Accept"); accept != "" {
		httpReq.Header.Set("Accept", accept)
	}

	switch provider {
	case "anthropic":
		if h.AnthropicKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
		httpReq.Header.Set("x-api-key", h.AnthropicKey)
		httpReq.Header.Set("anthropic-version", headerOrDefault(headers, "anthropic-version", "2023-06-01"))
		if beta := headers.Get("anthropic-beta"); beta != "" {
			httpReq.Header.Set("anthropic-beta", beta)
		}
	case "ollama", "ollama-cloud":
		if version := headers.Get("anthropic-version"); version != "" {
			httpReq.Header.Set("anthropic-version", version)
		}
		if beta := headers.Get("anthropic-beta"); beta != "" {
			httpReq.Header.Set("anthropic-beta", beta)
		}
	}

	return httpReq, nil
}

func (h *AnthropicMessagesHandler) upstreamMessagesURL(provider string) (string, error) {
	switch provider {
	case "anthropic":
		baseURL := strings.TrimRight(h.AnthropicBaseURL, "/")
		if baseURL == "" {
			baseURL = "https://api.anthropic.com"
		}
		return baseURL + "/v1/messages", nil
	case "ollama", "ollama-cloud":
		baseURL := strings.TrimRight(h.OllamaBaseURL, "/")
		if baseURL == "" {
			baseURL = "http://" + h.ChatHandler.OllamaHost
		}
		return baseURL + "/v1/messages", nil
	default:
		return "", fmt.Errorf("provider %s does not support native Anthropic Messages passthrough", provider)
	}
}

func (h *AnthropicMessagesHandler) providerTimeout(provider string) time.Duration {
	switch provider {
	case "anthropic":
		if h.AnthropicTimeout > 0 {
			return time.Duration(h.AnthropicTimeout) * time.Second
		}
	case "ollama", "ollama-cloud":
		if h.OllamaTimeout > 0 {
			return time.Duration(h.OllamaTimeout) * time.Second
		}
	}
	return 240 * time.Second
}

func (h *AnthropicMessagesHandler) supportsNativeMessages(provider string) bool {
	return provider == "anthropic" || provider == "ollama" || provider == "ollama-cloud"
}

func (h *AnthropicMessagesHandler) generateMessagesCacheKey(body []byte, route *domain.RouteConfig) string {
	hash := sha256.New()
	hash.Write([]byte("anthropic-messages"))
	hash.Write([]byte(route.Provider))
	hash.Write([]byte(route.Model))
	hash.Write(body)
	return hex.EncodeToString(hash.Sum(nil))
}

func (h *AnthropicMessagesHandler) jsonError(w http.ResponseWriter, status int, errorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"type": "error",
		"error": map[string]string{
			"type":    errorType,
			"message": message,
		},
	})
}

func sanitizeMessagesBody(body []byte, model string) ([]byte, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	delete(payload, "sensitive")
	delete(payload, "precision")
	delete(payload, "usecase")
	delete(payload, "no_cache")
	payload["model"] = model

	if maxTokens, ok := payload["max_tokens"].(float64); !ok || int(maxTokens) <= 0 {
		payload["max_tokens"] = 4096
	}

	return json.Marshal(payload)
}

func anthropicErrorTypeForStatus(status int) string {
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return "authentication_error"
	case status >= 400 && status < 500:
		return "invalid_request_error"
	default:
		return "api_error"
	}
}

func extractErrorMessage(body []byte) string {
	if len(body) == 0 {
		return "upstream request failed"
	}

	var payload map[string]interface{}
	if json.Unmarshal(body, &payload) == nil {
		if errObj, ok := payload["error"].(map[string]interface{}); ok {
			if message, ok := errObj["message"].(string); ok && message != "" {
				return message
			}
		}
		if message, ok := payload["message"].(string); ok && message != "" {
			return message
		}
	}

	return strings.TrimSpace(string(body))
}

func headerOrDefault(headers http.Header, key, fallback string) string {
	if value := headers.Get(key); value != "" {
		return value
	}
	return fallback
}

func contentTypeOrDefault(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}
