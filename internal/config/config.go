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
	// OpenAI GPT-5 series
	"gpt-5":     {1.25, 10.00},
	"gpt-5-mini": {0.50, 2.00},
	"gpt-5-nano": {0.10, 0.40},
	"gpt-5-pro":  {5.00, 20.00},
	"gpt-5.1":    {1.25, 10.00},
	"gpt-5.2":    {1.75, 14.00},
	"gpt-5.4":    {2.00, 16.00},
	// OpenAI GPT-4.1 series
	"gpt-4.1":      {2.00, 8.00},
	"gpt-4.1-mini": {0.40, 1.60},
	"gpt-4.1-nano": {0.10, 0.40},
	// OpenAI GPT-4o series
	"gpt-4o":      {2.50, 10.00},
	"gpt-4o-mini": {0.15, 0.60},
	// OpenAI reasoning models (o-series)
	"o1":      {15.00, 60.00},
	"o1-pro":  {150.00, 600.00},
	"o3":      {10.00, 40.00},
	"o3-mini": {1.10, 4.40},
	"o4-mini": {1.10, 4.40},
	// Anthropic - Claude models
	"claude-opus-4-6":            {5.00, 25.00},
	"claude-opus-4-5-20251101":   {5.00, 25.00},
	"claude-opus-4-1-20250805":   {5.00, 25.00},
	"claude-opus-4-20250514":     {15.00, 75.00},
	"claude-sonnet-4-6":          {3.00, 15.00},
	"claude-sonnet-4-5-20250929": {3.00, 15.00},
	"claude-sonnet-4-20250514":   {3.00, 15.00},
	"claude-haiku-4-5-20251001":  {1.00, 5.00},
	// Ollama (free - models currently installed on studio.lan)
	"myqwen3.5:120b":                                     {0, 0},
	"myqwen3.5:35b":                                      {0, 0},
	"myqwen2.5:14b-128k":                                 {0, 0},
	"qwen3.5:122b-a10b-q4_K_M":                          {0, 0},
	"huihui_ai/qwen3.5-abliterated:35b":                  {0, 0},
	"glm-ocr:latest":                                     {0, 0},
	"fixt/home-3b-v3:latest":                             {0, 0},
	"qwen3:4b-instruct":                                  {0, 0},
	"qwen3:30b-instruct":                                 {0, 0},
	"qwen2.5:14b-instruct":                               {0, 0},
	"qwen2.5:32b-instruct":                               {0, 0},
	"iquest-coder:40b-instruct-q4_K_M":                  {0, 0},
	"hf.co/mradermacher/IQuest-Coder-V1-40B-Instruct-GGUF:Q4_K_M": {0, 0},
	"gpt-oss:120b":                                       {0, 0},
	"devstral-small-2:24b":                               {0, 0},
	"devstral:24b":                                       {0, 0},
	"qwen3-vl:30b":                                       {0, 0},
	"qwen3-vl:32b":                                       {0, 0},
	"qwen3-vl:235b":                                      {0, 0},
	"qwen3:32b":                                          {0, 0},
	"qwen2.5:32b":                                        {0, 0},
	"qwen2.5-coder:32b":                                  {0, 0},
	"exaone-deep:32b":                                    {0, 0},
	"deepseek-coder:33b":                                 {0, 0},
	"qwen3-coder:30b":                                    {0, 0},
	"codestral:latest":                                   {0, 0},
	// Google Gemini models (real Google API model names)
	"gemini-3-pro-preview":           {2.50, 15.00},
	"gemini-3-flash-preview":         {0.30, 1.20},
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
			"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-6"},
			"high":      {Provider: "openai", Model: "gpt-4.1"},
			"medium":    {Provider: "openai", Model: "gpt-4o-mini"},
			"low":       {Provider: "ollama", Model: "qwen3:4b-instruct"},
		},
		// sensitive: true (local only)
		"true": {
			"very_high": nil, // Not available - Claude requires cloud
			"high":      {Provider: "ollama", Model: "qwen2.5:32b-instruct"},
			"medium":    {Provider: "ollama", Model: "qwen3:30b-instruct"},
			"low":       {Provider: "ollama", Model: "qwen3:4b-instruct"},
		},
	}
}

// DefaultVisionRoutes returns the default vision routing table.
func DefaultVisionRoutes() map[string]map[string]*domain.RouteConfig {
	return map[string]map[string]*domain.RouteConfig{
		// sensitive: false (can use cloud)
		"false": {
			"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-6"},
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
