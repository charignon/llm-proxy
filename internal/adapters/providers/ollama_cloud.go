package providers

import (
	"context"
	"sort"
	"strings"

	"llm-proxy/internal/domain"
)

var advertisedOllamaCloudModels = []string{
	"glm-4.7:cloud",
	"glm-5:cloud",
	"minimax-m2.1:cloud",
	"minimax-m2.5:cloud",
	"qwen3-coder:480b-cloud",
	"qwen3.5:cloud",
}

// OllamaCloudProvider forwards cloud-model requests through the local Ollama daemon.
type OllamaCloudProvider struct {
	base *OllamaProvider
}

// NewOllamaCloudProvider creates a virtual provider backed by the local Ollama daemon.
func NewOllamaCloudProvider(host string, timeout int) *OllamaCloudProvider {
	return &OllamaCloudProvider{base: NewOllamaProvider(host, timeout)}
}

// Chat implements ChatProvider.Chat for Ollama cloud models.
func (p *OllamaCloudProvider) Chat(ctx context.Context, req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	return p.base.chat(ctx, req, model, "ollama-cloud", true)
}

// GetModels returns the cloud models exposed through the local Ollama daemon.
func (p *OllamaCloudProvider) GetModels() []string {
	seen := make(map[string]bool)
	models := make([]string, 0, len(advertisedOllamaCloudModels))

	for _, model := range advertisedOllamaCloudModels {
		seen[strings.ToLower(model)] = true
		models = append(models, model)
	}

	for _, model := range p.base.fetchModelNames(nil) {
		if !IsOllamaCloudModel(model) {
			continue
		}
		key := strings.ToLower(model)
		if seen[key] {
			continue
		}
		seen[key] = true
		models = append(models, model)
	}

	sort.Strings(models)
	return models
}

// IsOllamaCloudModel reports whether a model should route through ollama-cloud.
func IsOllamaCloudModel(model string) bool {
	model = normalizeOllamaModelName(model)
	if model == "" {
		return false
	}

	lower := strings.ToLower(model)
	if strings.HasSuffix(lower, ":cloud") || strings.HasSuffix(lower, "-cloud") {
		return true
	}

	for _, candidate := range advertisedOllamaCloudModels {
		if strings.EqualFold(model, candidate) {
			return true
		}
	}

	return false
}

func normalizeOllamaModelName(model string) string {
	model = strings.TrimSpace(model)
	model = strings.TrimPrefix(model, "ollama-cloud/")
	model = strings.TrimPrefix(model, "ollama/")
	return model
}
