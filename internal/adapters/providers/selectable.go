// Package providers implements chat provider adapters.
package providers

import (
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// SelectableProvider implements ChatProvider by delegating to one of two
// providers based on a dynamic preference function.
type SelectableProvider struct {
	ollamaProvider   ports.ChatProvider
	llamacppProvider ports.ChatProvider
	getPreference    func() string // Returns "ollama" or "llamacpp"
}

// NewSelectableProvider creates a new selectable provider that delegates
// to either the Ollama or llama.cpp provider based on the preference function.
func NewSelectableProvider(ollama, llamacpp ports.ChatProvider, getPref func() string) *SelectableProvider {
	return &SelectableProvider{
		ollamaProvider:   ollama,
		llamacppProvider: llamacpp,
		getPreference:    getPref,
	}
}

// Chat implements the ChatProvider interface.
func (p *SelectableProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	return p.getSelectedProvider().Chat(req, model)
}

// IsHealthy returns true if the currently selected provider is healthy.
func (p *SelectableProvider) IsHealthy() bool {
	if checker, ok := p.getSelectedProvider().(interface{ IsHealthy() bool }); ok {
		return checker.IsHealthy()
	}
	return true
}

// getSelectedProvider returns the provider based on current preference.
func (p *SelectableProvider) getSelectedProvider() ports.ChatProvider {
	if p.getPreference() == "llamacpp" && p.llamacppProvider != nil {
		return p.llamacppProvider
	}
	return p.ollamaProvider
}

// GetCurrentBackend returns which backend is currently selected.
func (p *SelectableProvider) GetCurrentBackend() string {
	return p.getPreference()
}
