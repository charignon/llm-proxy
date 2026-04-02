package app

import (
	"testing"

	"llm-proxy/internal/domain"
)

func TestResolveExplicitModelRoutesOllamaCloudModels(t *testing.T) {
	t.Parallel()

	router := NewRouter(nil, nil)
	tests := []struct {
		model    string
		provider string
		want     string
	}{
		{model: "qwen3.5:cloud", provider: "ollama-cloud", want: "qwen3.5:cloud"},
		{model: "ollama-cloud/glm-5:cloud", provider: "ollama-cloud", want: "glm-5:cloud"},
		{model: "ollama/qwen3.5:cloud", provider: "ollama", want: "qwen3.5:cloud"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()

			route := router.resolveExplicitModel(tc.model)
			if route.Provider != tc.provider {
				t.Fatalf("resolveExplicitModel(%q) provider = %q, want %q", tc.model, route.Provider, tc.provider)
			}
			if route.Model != tc.want {
				t.Fatalf("resolveExplicitModel(%q) model = %q, want %q", tc.model, route.Model, tc.want)
			}
		})
	}
}

func TestResolveRouteRejectsSensitiveExplicitCloudModels(t *testing.T) {
	t.Parallel()

	router := NewRouter(nil, nil)
	sensitive := true
	_, err := router.ResolveRoute(&domain.ChatCompletionRequest{
		Model:     "qwen3.5:cloud",
		Sensitive: &sensitive,
	})
	if err == nil {
		t.Fatal("expected sensitive explicit cloud model to be rejected")
	}
}

func TestResolveRouteUsesAssistantAliasResolverForMLXAlias(t *testing.T) {
	t.Parallel()

	router := NewRouter(nil, nil)
	router.SetAssistantResolver(func(alias string) (string, string, bool) {
		if alias == "assistant-mlx" {
			return "mlx", "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", true
		}
		return "", "", false
	})

	route, err := router.ResolveRoute(&domain.ChatCompletionRequest{
		Model: "mlx/assistant-mlx:latest",
	})
	if err != nil {
		t.Fatalf("ResolveRoute returned error: %v", err)
	}
	if route.Provider != "mlx" {
		t.Fatalf("ResolveRoute provider = %q, want %q", route.Provider, "mlx")
	}
	if route.Model != "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit" {
		t.Fatalf("ResolveRoute model = %q, want %q", route.Model, "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit")
	}
}
