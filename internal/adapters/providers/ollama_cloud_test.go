package providers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-proxy/internal/domain"
)

func TestIsOllamaCloudModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model string
		want  bool
	}{
		{model: "qwen3.5:cloud", want: true},
		{model: "qwen3-coder:480b-cloud", want: true},
		{model: "ollama-cloud/glm-5:cloud", want: true},
		{model: "ollama/qwen3-vl:30b", want: false},
		{model: "glm-ocr:latest", want: false},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			if got := IsOllamaCloudModel(tc.model); got != tc.want {
				t.Fatalf("IsOllamaCloudModel(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

func TestOllamaProviderRejectsCloudModels(t *testing.T) {
	t.Parallel()

	p := NewOllamaProvider("127.0.0.1:1", 1)
	_, err := p.Chat(&domain.ChatCompletionRequest{}, "qwen3.5:cloud")
	if err == nil {
		t.Fatal("expected cloud model rejection error")
	}
	if !strings.Contains(err.Error(), "ollama-cloud") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOllamaModelListsSeparateLocalAndCloudModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen3-vl:30b"},{"name":"qwen3.5:cloud"},{"name":"gemma3:27b"}]}`))
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	localProvider := NewOllamaProvider(host, 1)
	cloudProvider := NewOllamaCloudProvider(host, 1)

	localModels := strings.Join(localProvider.GetModels(), ",")
	if strings.Contains(localModels, "qwen3.5:cloud") {
		t.Fatalf("local ollama models unexpectedly include cloud model: %s", localModels)
	}
	if strings.Contains(localModels, "gemma3:27b") {
		t.Fatalf("local ollama models unexpectedly include filtered gemma model: %s", localModels)
	}
	if !strings.Contains(localModels, "qwen3-vl:30b") {
		t.Fatalf("local ollama models missing local model: %s", localModels)
	}

	cloudModels := strings.Join(cloudProvider.GetModels(), ",")
	if !strings.Contains(cloudModels, "qwen3.5:cloud") {
		t.Fatalf("cloud provider models missing dynamic cloud model: %s", cloudModels)
	}
	if !strings.Contains(cloudModels, "glm-5:cloud") {
		t.Fatalf("cloud provider models missing advertised cloud model: %s", cloudModels)
	}
}
