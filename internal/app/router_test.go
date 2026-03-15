package app

import "testing"

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
