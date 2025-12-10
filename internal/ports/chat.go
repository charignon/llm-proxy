// Package ports defines the primary and secondary ports for the hexagonal architecture.
// Ports are interfaces that define how the application core interacts with the outside world.
package ports

import "llm-proxy/internal/domain"

// ChatProvider is a secondary port (driven) for sending chat completion requests to LLM backends.
// Implementations include OpenAI, Anthropic, and Ollama adapters.
type ChatProvider interface {
	// Chat sends a completion request to the provider and returns the response.
	Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error)
}
