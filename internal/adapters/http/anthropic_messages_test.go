package http

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	cacheadapter "llm-proxy/internal/adapters/cache"
	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

func TestAnthropicMessagesHandlerPassthroughToOllama(t *testing.T) {
	t.Parallel()

	var received map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_ollama",
			"type":"message",
			"role":"assistant",
			"content":[{"type":"text","text":"native ollama"}],
			"model":"qwen3-vl:30b",
			"stop_reason":"end_turn",
			"usage":{"input_tokens":11,"output_tokens":7}
		}`))
	}))
	defer upstream.Close()

	handler := &AnthropicMessagesHandler{
		ChatHandler:   minimalChatHandler(t, nil),
		OllamaBaseURL: upstream.URL,
		OllamaTimeout: 5,
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"ollama/qwen3-vl:30b",
		"max_tokens":64,
		"messages":[{"role":"user","content":"hello"}],
		"usecase":"test-anthropic-ollama",
		"sensitive":true,
		"precision":"high"
	}`))
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-LLM-Proxy-Messages-Mode") != "passthrough" {
		t.Fatalf("expected passthrough mode, got %q", rr.Header().Get("X-LLM-Proxy-Messages-Mode"))
	}
	if got := received["model"]; got != "qwen3-vl:30b" {
		t.Fatalf("upstream model = %v, want qwen3-vl:30b", got)
	}
	if _, exists := received["usecase"]; exists {
		t.Fatalf("usecase leaked to upstream request: %+v", received)
	}
	if _, exists := received["sensitive"]; exists {
		t.Fatalf("sensitive leaked to upstream request: %+v", received)
	}
	if _, exists := received["precision"]; exists {
		t.Fatalf("precision leaked to upstream request: %+v", received)
	}

	var resp domain.AnthropicMessagesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "native ollama" {
		t.Fatalf("unexpected response content: %+v", resp.Content)
	}
}

func TestAnthropicMessagesHandlerTranslatesForOpenAI(t *testing.T) {
	t.Parallel()

	provider := stubChatProvider{
		resp: &domain.ChatCompletionResponse{
			ID:      "chatcmpl_openai",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4o",
			Choices: []domain.Choice{{
				Index: 0,
				Message: domain.Message{
					Role:    "assistant",
					Content: "translated response",
				},
				FinishReason: "stop",
			}},
			Usage: &domain.Usage{
				PromptTokens:     9,
				CompletionTokens: 5,
				TotalTokens:      14,
			},
		},
	}

	handler := &AnthropicMessagesHandler{
		ChatHandler: minimalChatHandler(t, map[string]ports.ChatProvider{
			"openai": provider,
		}),
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"gpt-4o",
		"max_tokens":64,
		"messages":[{"role":"user","content":"hello"}],
		"usecase":"test-anthropic-openai"
	}`))
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-LLM-Proxy-Messages-Mode") != "translate" {
		t.Fatalf("expected translate mode, got %q", rr.Header().Get("X-LLM-Proxy-Messages-Mode"))
	}

	var resp domain.AnthropicMessagesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Type != "message" {
		t.Fatalf("response type = %q, want message", resp.Type)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "translated response" {
		t.Fatalf("unexpected response content: %+v", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %q, want end_turn", resp.StopReason)
	}
}

type stubChatProvider struct {
	resp *domain.ChatCompletionResponse
	err  error
}

func (s stubChatProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	return s.resp, s.err
}

type stubRequestLogger struct{}

func (stubRequestLogger) LogRequest(entry *domain.RequestLog) int64 {
	return 1
}

type stubMetricsRecorder struct{}

func (stubMetricsRecorder) RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool) {
}

func minimalChatHandler(t *testing.T, providers map[string]ports.ChatProvider) *ChatHandler {
	t.Helper()

	if providers == nil {
		providers = map[string]ports.ChatProvider{}
	}

	return &ChatHandler{
		Router:    app.NewRouter(nil, nil),
		Providers: providers,
		Cache:     cacheadapter.NewMemoryCache(1),
		Logger:    stubRequestLogger{},
		Metrics:   stubMetricsRecorder{},
		GenerateKey: func(req *domain.ChatCompletionRequest, route *domain.RouteConfig) string {
			return "test-cache-key"
		},
		CalculateCost: func(model string, inputTokens, outputTokens int) float64 {
			return 0
		},
		AddPending: func(req *domain.ChatCompletionRequest, route *domain.RouteConfig, startTime time.Time) string {
			return "pending"
		},
		RemovePending: func(id string) {},
		OllamaHost:    "127.0.0.1:11434",
	}
}
