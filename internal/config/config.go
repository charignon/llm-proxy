// Package config provides configuration for the LLM Proxy service.
package config

import (
	"os"

	"llm-proxy/internal/domain"
)

// Config holds all configuration values for the LLM Proxy service.
type Config struct {
	Port             string
	OpenAIKey        string
	AnthropicKey     string
	GeminiKey        string
	AidaToken        string
	OllamaHost       string
	DataDir          string
	CacheTTLHours    int
	WhisperServerURL string
	TTSServerURL     string
}

// Load returns configuration from environment variables with defaults.
func Load() *Config {
	return &Config{
		Port:             getEnv("PORT", "8080"),
		OpenAIKey:        getEnv("OPENAI_API_KEY", ""),
		AnthropicKey:     getEnv("ANTHROPIC_API_KEY", ""),
		GeminiKey:        getEnv("GEMINI_API_KEY", ""),
		AidaToken:        getEnv("AIDA_TOKEN", ""),
		OllamaHost:       getEnv("OLLAMA_HOST", "localhost:11434"),
		DataDir:          getEnv("DATA_DIR", "./data"),
		CacheTTLHours:    24 * 7, // 1 week cache
		WhisperServerURL: getEnv("WHISPER_SERVER_URL", "http://localhost:8890"),
		TTSServerURL:     getEnv("TTS_SERVER_URL", "http://localhost:7788"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ModelPricing maps model names to pricing per 1M tokens [input, output].
var ModelPricing = map[string][2]float64{
	// OpenAI GPT-4o series
	"gpt-4o":      {2.50, 10.00},
	"gpt-4o-mini": {0.15, 0.60},
	// OpenAI GPT-4 legacy
	"gpt-4-turbo": {10.00, 30.00},
	"gpt-4":       {30.00, 60.00},
	// OpenAI GPT-3.5
	"gpt-3.5-turbo": {0.50, 1.50},
	// OpenAI reasoning models (o-series)
	"o1":         {15.00, 60.00},
	"o1-mini":    {1.10, 4.40},
	"o3-mini":    {1.10, 4.40},
	"o1-pro":     {150.00, 600.00},
	"o3":         {10.00, 40.00},
	"o4-mini":    {1.10, 4.40},
	"codex-mini": {0.25, 2.00},
	// Anthropic - Claude 4.5 models only
	"claude-opus-4-5-20251101":   {5.00, 25.00},
	"claude-sonnet-4-5-20250929": {3.00, 15.00},
	"claude-haiku-4-5-20251001":  {1.00, 5.00},
	// Ollama (free)
	"qwen3-vl:30b":       {0, 0},
	"qwen3-vl:235b":      {0, 0},
	"llama3.3:70b":       {0, 0},
	"llama3.1-large":     {0, 0},
	"llama4:scout":       {0, 0},
	"mistral:7b":         {0, 0},
	"devstral:24b":       {0, 0},
	"deepseek-r1:70b":    {0, 0},
	"qwen3-coder:30b":    {0, 0},
	"deepseek-coder:33b": {0, 0},
	"phi4:14b":           {0, 0},
	"codestral:latest":   {0, 0},
	"granite3.1-moe:3b":  {0, 0},
	"qwen2.5:1.5b":       {0, 0},
	// Google Gemini models (real Google API model names)
	"gemini-2.5-pro-preview-06-05":   {1.25, 10.00},
	"gemini-2.5-flash-preview-05-20": {0.15, 0.60},
	"gemini-2.0-flash":               {0.10, 0.40},
	"gemini-2.0-flash-lite":          {0.02, 0.08},
	"gemini-1.5-pro":                 {1.25, 5.00},
	"gemini-1.5-flash":               {0.075, 0.30},
	"gemini-1.5-flash-8b":            {0.0375, 0.15},
	"gemini-exp-1206":                {0, 0}, // Free experimental
	"gemini-2.0-flash-thinking-exp":  {0, 0}, // Free experimental
}

// CalculateCost computes the cost for a request based on token counts.
func CalculateCost(model string, inputTokens, outputTokens int) float64 {
	pricing, ok := ModelPricing[model]
	if !ok {
		return 0
	}
	inputCost := float64(inputTokens) * pricing[0] / 1_000_000
	outputCost := float64(outputTokens) * pricing[1] / 1_000_000
	return inputCost + outputCost
}

// DefaultTextRoutes returns the default text routing table.
func DefaultTextRoutes() map[string]map[string]*domain.RouteConfig {
	return map[string]map[string]*domain.RouteConfig{
		// sensitive: false (can use cloud)
		"false": {
			"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-5-20250929"},
			"high":      {Provider: "openai", Model: "gpt-4o"},
			"medium":    {Provider: "openai", Model: "gpt-4o-mini"},
			"low":       {Provider: "ollama", Model: "qwen2.5:1.5b"},
		},
		// sensitive: true (local only)
		"true": {
			"very_high": nil, // Not available - Claude requires cloud
			"high":      {Provider: "ollama", Model: "llama3.3:70b"},
			"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
			"low":       {Provider: "ollama", Model: "qwen2.5:1.5b"},
		},
	}
}

// DefaultVisionRoutes returns the default vision routing table.
func DefaultVisionRoutes() map[string]map[string]*domain.RouteConfig {
	return map[string]map[string]*domain.RouteConfig{
		// sensitive: false (can use cloud)
		"false": {
			"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-5-20250929"},
			"high":      {Provider: "openai", Model: "gpt-4o-mini"},
			"medium":    {Provider: "openai", Model: "gpt-4o"},
			"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
		},
		// sensitive: true (local only)
		"true": {
			"very_high": {Provider: "ollama", Model: "qwen3-vl:235b"},
			"high":      {Provider: "ollama", Model: "qwen3-vl:30b"},
			"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
			"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
		},
	}
}
