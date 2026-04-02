package http

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	cacheadapter "llm-proxy/internal/adapters/cache"
	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

type blockingChatProvider struct {
	started chan struct{}
	ctxErr  chan error
}

func (p *blockingChatProvider) Chat(ctx context.Context, req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	_ = req
	_ = model
	close(p.started)
	<-ctx.Done()
	p.ctxErr <- ctx.Err()
	return nil, ctx.Err()
}

func TestChatHandlerPropagatesRequestCancellationToProvider(t *testing.T) {
	t.Parallel()

	provider := &blockingChatProvider{
		started: make(chan struct{}),
		ctxErr:  make(chan error, 1),
	}

	handler := &ChatHandler{
		Router:    app.NewRouter(nil, nil),
		Providers: map[string]ports.ChatProvider{"openai": provider},
		Cache:     cacheadapter.NewMemoryCache(1),
		Logger:    stubRequestLogger{},
		Metrics:   stubMetricsRecorder{},
		GenerateKey: func(req *domain.ChatCompletionRequest, route *domain.RouteConfig) string {
			return "test-cache-key"
		},
		CalculateCost: func(model string, inputTokens, outputTokens int) float64 {
			return 0
		},
		AddPending: func(req *domain.ChatCompletionRequest, route *domain.RouteConfig, startTime time.Time, cancel context.CancelFunc) string {
			_ = req
			_ = route
			_ = startTime
			_ = cancel
			return "pending"
		},
		RemovePending: func(id string) {},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"openai/gpt-4o",
		"messages":[{"role":"user","content":"hello"}],
		"usecase":"cancel-test"
	}`)).WithContext(ctx)
	rr := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(rr, req)
		close(done)
	}()

	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("provider was not called")
	}

	cancel()

	select {
	case err := <-provider.ctxErr:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("provider context error = %v, want context canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("provider did not observe cancellation")
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return after cancellation")
	}

	if rr.Code != statusClientClosedRequest {
		t.Fatalf("status = %d, want %d", rr.Code, statusClientClosedRequest)
	}
}
