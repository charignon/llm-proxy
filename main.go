// Package main runs the llm-proxy service: an HTTP router/proxy for LLM, STT,
// and TTS requests with routing, caching, history, analytics, and UI.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"llm-proxy/internal/adapters/audiocache"
	"llm-proxy/internal/adapters/budget"
	"llm-proxy/internal/adapters/cache"
	httphandlers "llm-proxy/internal/adapters/http"
	"llm-proxy/internal/adapters/loadmanager"
	"llm-proxy/internal/adapters/providers"
	"llm-proxy/internal/adapters/repository"
	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"

	_ "github.com/mattn/go-sqlite3"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/mem"
)

// Configuration
var (
	port                   = getEnv("PORT", "8080")
	openaiKey              = getEnv("OPENAI_API_KEY", "")
	anthropicKey           = getEnv("ANTHROPIC_API_KEY", "")
	geminiKey              = getEnv("GEMINI_API_KEY", "")
	aidaToken              = getEnv("AIDA_TOKEN", "") // Google AIDA API token for Jules
	ollamaHost             = getEnv("OLLAMA_HOST", "localhost:11434")
	llamacppHost           = getEnv("LLAMACPP_HOST", "")        // Optional: llama.cpp text server (e.g., "localhost:8091")
	llamacppVisionHost     = getEnv("LLAMACPP_VISION_HOST", "") // Optional: llama.cpp vision server with mmproj (e.g., "localhost:8081")
	mlxHost                = getEnv("MLX_HOST", "")      // Optional: MLX LM server (e.g., "localhost:8086")
	dataDir                = getEnv("DATA_DIR", "./data")
	postgresConnStr        = getEnv("POSTGRES_CONN_STR", "")                       // PostgreSQL connection string for analytics (optional)
	cacheTTLHours          = 24 * 7                                                // 1 week cache
	whisperServerURL       = getEnv("WHISPER_SERVER_URL", "http://localhost:8890") // Local whisper server
	ttsServerURL           = getEnv("TTS_SERVER_URL", "http://localhost:7788")     // Local TTS server (Kokoro)
	ttsCacheDir            = getEnv("TTS_CACHE_DIR", "")                           // TTS audio cache directory (default: DATA_DIR/tts-cache)
	ttsCacheMaxSize        = getEnvInt("TTS_CACHE_MAX_SIZE_MB", 1024)              // TTS cache max size in MB (default: 1GB)
	ttsCacheTTLDays        = getEnvInt("TTS_CACHE_TTL_DAYS", 7)                    // TTS cache TTL in days (default: 7 days)
	chatTimeout            = getEnvInt("CHAT_TIMEOUT", 240)                        // Chat timeout in seconds
	speechTimeout          = getEnvInt("SPEECH_TIMEOUT", 240)                      // Speech transcription timeout in seconds
	speechStreamingTimeout = getEnvInt("SPEECH_STREAMING_TIMEOUT", 240)            // Speech streaming transcription timeout in seconds
	aidaTimeout            = getEnvInt("AIDA_TIMEOUT", 300)                        // AIDA proxy timeout in seconds
	geminiTimeout          = getEnvInt("GEMINI_TIMEOUT", 300)                      // Gemini proxy and provider timeout in seconds
	openaiTimeout          = getEnvInt("OPENAI_TIMEOUT", 240)                      // OpenAI provider timeout in seconds
	openaiStreamingTimeout = getEnvInt("OPENAI_STREAMING_TIMEOUT", 300)            // OpenAI streaming timeout in seconds
	anthropicTimeout       = getEnvInt("ANTHROPIC_TIMEOUT", 240)                   // Anthropic provider timeout in seconds
	imageGenTimeout        = getEnvInt("IMAGE_GEN_TIMEOUT", 120)                   // Image generation timeout in seconds
	ttsTimeout             = getEnvInt("TTS_TIMEOUT", 120)                         // TTS timeout in seconds (OpenAI)
	ttsKokoroTimeout       = getEnvInt("TTS_KOKORO_TIMEOUT", 60)                   // TTS Kokoro timeout in seconds
	webSearchTimeout       = getEnvInt("WEB_SEARCH_TIMEOUT", 120)                  // Web search timeout in seconds
	llamacppTimeout        = getEnvInt("LLAMACPP_TIMEOUT", 300)                    // llama.cpp vision timeout in seconds
	mlxTimeout             = getEnvInt("MLX_TIMEOUT", 300)                         // MLX server timeout in seconds
	togetherKey            = getEnv("TOGETHER_API_KEY", "")                        // Together.ai API key
	togetherTimeout        = getEnvInt("TOGETHER_TIMEOUT", 240)                    // Together.ai provider timeout in seconds
	basetenKey             = getEnv("BASETEN_API_KEY", "")                         // Baseten API key
	basetenTimeout         = getEnvInt("BASETEN_TIMEOUT", 240)                     // Baseten provider timeout in seconds
)

// Model pricing per 1M tokens (input, output)
var modelPricing = map[string][2]float64{
	// OpenAI GPT-5 series
	"gpt-5":      {1.25, 10.00},
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
	"myqwen3.5:120b":                    {0, 0},
	"myqwen3.5:35b":                     {0, 0},
	"myqwen2.5:14b-128k":                {0, 0},
	"qwen3.5:122b-a10b-q4_K_M":          {0, 0},
	"huihui_ai/qwen3.5-abliterated:35b": {0, 0},
	"glm-ocr:latest":                    {0, 0},
	"fixt/home-3b-v3:latest":            {0, 0},
	"qwen3:4b-instruct":                 {0, 0},
	"qwen3:30b-instruct":                {0, 0},
	"qwen2.5:14b-instruct":              {0, 0},
	"qwen2.5:32b-instruct":              {0, 0},
	"iquest-coder:40b-instruct-q4_K_M":  {0, 0},
	"hf.co/mradermacher/IQuest-Coder-V1-40B-Instruct-GGUF:Q4_K_M": {0, 0},
	"gpt-oss:120b":         {0, 0},
	"devstral-small-2:24b": {0, 0},
	"devstral:24b":         {0, 0},
	"qwen3-vl:30b":         {0, 0},
	"qwen3-vl:32b":         {0, 0},
	"qwen3-vl:235b":        {0, 0},
	"qwen3:32b":            {0, 0},
	"qwen2.5:32b":          {0, 0},
	"qwen2.5-coder:32b":    {0, 0},
	"exaone-deep:32b":      {0, 0},
	"deepseek-coder:33b":   {0, 0},
	"qwen3-coder:30b":      {0, 0},
	"codestral:latest":     {0, 0},
	"gemma4:26b":           {0, 0},
	"gemma4:31b":           {0, 0},
	"gemma4:e4b":           {0, 0},
	"gemma4:e2b":           {0, 0},
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
	// Together.ai models (top 20 largest/most capable)
	"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {0.27, 0.85}, // 1024K ctx, 128-expert MoE
	"meta-llama/Llama-4-Scout-17B-16E-Instruct":         {0.18, 0.59}, // 1024K ctx, 16-expert MoE
	"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":           {2.00, 2.00}, // 256K ctx, 480B params
	"moonshotai/Kimi-K2-Thinking":                       {1.20, 4.00}, // 256K ctx, 1T params
	"moonshotai/Kimi-K2-Instruct":                       {1.00, 3.00}, // 256K ctx, 1T params
	"moonshotai/Kimi-K2-Instruct-0905":                  {1.00, 3.00}, // 256K ctx, 1T params
	"Qwen/Qwen3-235B-A22B-Thinking-2507":                {0.65, 3.00}, // 256K ctx, thinking model
	"Qwen/Qwen3-VL-32B-Instruct":                        {0.50, 1.50}, // 256K ctx, vision
	"Qwen/Qwen3-235B-A22B-Instruct-2507-tput":           {0.20, 0.60}, // 256K ctx, throughput
	"Qwen/Qwen3-Next-80B-A3B-Thinking":                  {0.15, 1.50}, // 256K ctx, thinking
	"Qwen/Qwen3-Next-80B-A3B-Instruct":                  {0.15, 1.50}, // 256K ctx
	"zai-org/GLM-4.6":                                   {0.60, 2.20}, // 198K ctx, 357B params
	"deepseek-ai/DeepSeek-R1":                           {3.00, 7.00}, // 160K ctx, reasoning
	"deepcogito/cogito-v2-1-671b":                       {1.25, 1.25}, // 160K ctx, 671B MoE
	"deepseek-ai/DeepSeek-R1-0528-tput":                 {0.55, 2.19}, // 160K ctx, throughput
	"deepseek-ai/DeepSeek-V3":                           {1.25, 1.25}, // 128K ctx
	"deepseek-ai/DeepSeek-V3.1":                         {0.60, 1.70}, // 128K ctx
	"Qwen/Qwen2.5-72B-Instruct-Turbo":                   {1.20, 1.20}, // 128K ctx
	"meta-llama/Llama-3.3-70B-Instruct-Turbo":           {0.88, 0.88}, // 128K ctx
	"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo":     {3.50, 3.50}, // 128K ctx, 405B params
	"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":      {0.88, 0.88}, // 128K ctx
}

// RouteConfig is an alias to the domain type for routing decisions.
type RouteConfig = domain.RouteConfig

// Route based on sensitive + precision + hasImages flags
// Format: routingTable[sensitive][precision] for text
//
//	visionRoutingTable[sensitive][precision] for vision
//
// Precision levels: very_high > high > medium > low
var routingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (text only)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-6"},
		"high":      {Provider: "openai", Model: "gpt-4.1"},
		"medium":    {Provider: "openai", Model: "gpt-4o-mini"},
		"low":       {Provider: "ollama", Model: "qwen3:4b-instruct"},
	},
	// sensitive: true (text only, local)
	"true": {
		"very_high": nil, // Not available - Claude requires cloud
		"high":      {Provider: "ollama", Model: "qwen2.5:32b-instruct"},
		"medium":    {Provider: "ollama", Model: "qwen3:30b-instruct"},
		"low":       {Provider: "ollama", Model: "qwen3:4b-instruct"},
	},
}

// Vision routing (requests with images)
// Precision levels: very_high, high, medium, low
var visionRoutingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (can use cloud)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-6"}, // Claude has great vision
		"high":      {Provider: "openai", Model: "gpt-4o-mini"},          // Fast and cheap
		"medium":    {Provider: "openai", Model: "gpt-4o"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
	// sensitive: true (local only)
	"true": {
		"very_high": {Provider: "ollama", Model: "qwen3-vl:235b"}, // Largest local vision model (143GB)
		"high":      {Provider: "ollama", Model: "qwen3-vl:30b"},
		"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
}

// Router service for request routing decisions
var router *app.Router

// HTTP handler for chat completions (primary adapter)
var chatHandler *httphandlers.ChatHandler
var responsesHandler *httphandlers.ResponsesHandler
var anthropicMessagesHandler *httphandlers.AnthropicMessagesHandler

// Audio and search handlers (extracted adapters)
var sttHandler *httphandlers.STTHandler
var ttsHandler *httphandlers.TTSHandler
var ttsAudioCache *audiocache.FileAudioCache
var webSearchHandler *httphandlers.WebSearchHandler
var imageGenHandler *httphandlers.ImageGenHandler

// Concurrency manager for dynamic load-based limiting
var sttConcurrencyMgr *loadmanager.ConcurrencyManager

// History and stats handler
var historyHandler *httphandlers.HistoryHandler

// UI handler
var uiHandler *httphandlers.UIHandler

// Database
var db *sql.DB
var dbMutex sync.Mutex

// Request logger - using the port interface
var requestLogger ports.RequestLogger

// Budget repository and checker
var budgetRepository ports.BudgetRepository
var budgetChecker *budget.BudgetChecker

// Cache - using the port interface
var requestCache ports.Cache

// CacheEntry is an alias to the domain type for cached entries.
type CacheEntry = domain.CacheEntry

func initCache() {
	requestCache = cache.NewMemoryCache(cacheTTLHours)
}

// Disabled models set (models that are turned off in the UI)
var disabledModels = make(map[string]bool)
var disabledModelsMutex sync.RWMutex

// Local vision backend preference: "ollama" or "llamacpp"
var localVisionBackend = "ollama" // default to ollama
var localVisionBackendMutex sync.RWMutex

// Assistant model mappings: alias -> {provider, model}
type AssistantModelConfig struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

var assistantAliasOrder = []string{
	"assistant",
	"assistant-mini",
	"assistant-nano",
	"assistant-mlx",
}

var assistantAliasDefaults = map[string]AssistantModelConfig{
	"assistant": {
		Provider: "ollama",
		Model:    "qwen3.5:122b-a10b-q4_K_M",
	},
	"assistant-mini": {
		Provider: "ollama",
		Model:    "myqwen3.5:35b",
	},
	"assistant-nano": {
		Provider: "ollama",
		Model:    "qwen3:4b-instruct",
	},
	"assistant-mlx": {
		Provider: "mlx",
		Model:    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
	},
}

var assistantModels = make(map[string]AssistantModelConfig)
var assistantModelsMutex sync.RWMutex

func isSupportedAssistantAlias(alias string) bool {
	_, ok := assistantAliasDefaults[alias]
	return ok
}

func formatSupportedAssistantAliases() string {
	quotedAliases := make([]string, 0, len(assistantAliasOrder))
	for _, alias := range assistantAliasOrder {
		quotedAliases = append(quotedAliases, fmt.Sprintf("%q", alias))
	}
	return strings.Join(quotedAliases, ", ")
}

func isModelDisabled(model string) bool {
	disabledModelsMutex.RLock()
	defer disabledModelsMutex.RUnlock()
	return disabledModels[model]
}

func loadDisabledModels() {
	rows, err := db.Query(`SELECT model FROM disabled_models`)
	if err != nil {
		log.Printf("Failed to load disabled models: %v", err)
		return
	}
	defer rows.Close()

	disabledModelsMutex.Lock()
	defer disabledModelsMutex.Unlock()
	disabledModels = make(map[string]bool)
	for rows.Next() {
		var model string
		if err := rows.Scan(&model); err != nil {
			continue
		}
		disabledModels[model] = true
	}
	log.Printf("Loaded %d disabled models", len(disabledModels))
}

// loadLocalVisionBackend loads the local vision backend preference from the database.
func loadLocalVisionBackend() {
	var value string
	err := db.QueryRow(`SELECT value FROM settings WHERE key = 'local_vision_backend'`).Scan(&value)
	if err != nil {
		// Default to ollama if not set
		localVisionBackend = "ollama"
		log.Printf("Local vision backend: ollama (default)")
		return
	}
	localVisionBackendMutex.Lock()
	defer localVisionBackendMutex.Unlock()
	if value == "llamacpp" || value == "ollama" {
		localVisionBackend = value
	} else {
		localVisionBackend = "ollama"
	}
	log.Printf("Local vision backend: %s", localVisionBackend)
}

// getLocalVisionBackend returns the current local vision backend preference.
func getLocalVisionBackend() string {
	localVisionBackendMutex.RLock()
	defer localVisionBackendMutex.RUnlock()
	return localVisionBackend
}

// setLocalVisionBackend sets the local vision backend preference and persists to database.
func setLocalVisionBackend(backend string) error {
	if backend != "ollama" && backend != "llamacpp" {
		return fmt.Errorf("invalid backend: %s (must be 'ollama' or 'llamacpp')", backend)
	}

	dbMutex.Lock()
	defer dbMutex.Unlock()

	_, err := db.Exec(`INSERT OR REPLACE INTO settings (key, value) VALUES ('local_vision_backend', ?)`, backend)
	if err != nil {
		return err
	}

	localVisionBackendMutex.Lock()
	localVisionBackend = backend
	localVisionBackendMutex.Unlock()

	log.Printf("Local vision backend changed to: %s", backend)
	return nil
}

// getLocalVisionProvider returns the chat provider to use for local vision requests.
func getLocalVisionProvider() ports.ChatProvider {
	backend := getLocalVisionBackend()
	if backend == "llamacpp" && llamacppProvider != nil {
		return llamacppProvider
	}
	return ollamaProvider
}

// Lockdown mode: when enabled, all cloud providers (OpenAI, Anthropic, Gemini,
// Ollama Cloud, Together, Baseten) are blocked and only local models
// (ollama, llamacpp, mlx) are allowed. Persisted in the settings table.
var lockdownMode = false
var lockdownModeMutex sync.RWMutex

// loadLockdownMode loads the lockdown preference from the database.
func loadLockdownMode() {
	var value string
	err := db.QueryRow(`SELECT value FROM settings WHERE key = 'lockdown_mode'`).Scan(&value)
	if err != nil {
		lockdownMode = false
		log.Printf("Lockdown mode: off (default)")
		return
	}
	lockdownModeMutex.Lock()
	defer lockdownModeMutex.Unlock()
	lockdownMode = value == "1" || value == "true"
	if lockdownMode {
		log.Printf("Lockdown mode: ON (cloud providers blocked)")
	} else {
		log.Printf("Lockdown mode: off")
	}
}

// getLockdownMode returns whether lockdown mode is currently enabled.
func getLockdownMode() bool {
	lockdownModeMutex.RLock()
	defer lockdownModeMutex.RUnlock()
	return lockdownMode
}

// setLockdownMode sets and persists the lockdown mode preference.
func setLockdownMode(enabled bool) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	value := "0"
	if enabled {
		value = "1"
	}
	_, err := db.Exec(`INSERT OR REPLACE INTO settings (key, value) VALUES ('lockdown_mode', ?)`, value)
	if err != nil {
		return err
	}

	lockdownModeMutex.Lock()
	lockdownMode = enabled
	lockdownModeMutex.Unlock()

	if enabled {
		log.Printf("Lockdown mode ENABLED: cloud providers blocked")
	} else {
		log.Printf("Lockdown mode disabled")
	}
	return nil
}

// getProviderOverride returns an override provider for vision models if llamacpp is preferred.
// This intercepts ollama vision model requests and routes them to llamacpp when the backend is set.
// Returns the provider and name, or nil/"" if no override.
func getProviderOverride(provider, model string) (ports.ChatProvider, string) {
	// Only intercept ollama vision model requests when llamacpp is preferred
	if provider != "ollama" || llamacppProvider == nil {
		return nil, ""
	}

	// Check if this is a vision model (qwen3-vl)
	if !strings.Contains(strings.ToLower(model), "qwen3-vl") {
		return nil, ""
	}

	// Return llamacpp provider if that's the preferred backend
	if getLocalVisionBackend() == "llamacpp" {
		return llamacppProvider, "llamacpp"
	}

	return nil, ""
}

// loadAssistantModels loads assistant model mappings from the database.
func loadAssistantModels() {
	rows, err := db.Query(`SELECT alias, provider, model FROM assistant_models`)
	if err != nil {
		log.Printf("Failed to load assistant models: %v", err)
		return
	}
	defer rows.Close()

	assistantModelsMutex.Lock()
	defer assistantModelsMutex.Unlock()
	assistantModels = make(map[string]AssistantModelConfig)
	for rows.Next() {
		var alias, provider, model string
		if err := rows.Scan(&alias, &provider, &model); err != nil {
			continue
		}
		assistantModels[alias] = AssistantModelConfig{Provider: provider, Model: model}
	}
	log.Printf("Loaded %d assistant model mappings", len(assistantModels))
}

// getAssistantModel returns the provider and model for an assistant alias, or empty if not found.
func getAssistantModel(alias string) (string, string, bool) {
	assistantModelsMutex.RLock()
	defer assistantModelsMutex.RUnlock()
	if config, ok := assistantModels[alias]; ok {
		return config.Provider, config.Model, true
	}
	return "", "", false
}

// setAssistantModel sets the provider and model for an assistant alias and persists to database.
func setAssistantModel(alias, provider, model string) error {
	if !isSupportedAssistantAlias(alias) {
		return fmt.Errorf("invalid alias: %s (must be one of %s)", alias, formatSupportedAssistantAliases())
	}

	dbMutex.Lock()
	defer dbMutex.Unlock()

	_, err := db.Exec(`INSERT OR REPLACE INTO assistant_models (alias, provider, model) VALUES (?, ?, ?)`, alias, provider, model)
	if err != nil {
		return err
	}

	assistantModelsMutex.Lock()
	assistantModels[alias] = AssistantModelConfig{Provider: provider, Model: model}
	assistantModelsMutex.Unlock()

	log.Printf("Assistant model %s changed to: %s/%s", alias, provider, model)
	return nil
}

// getAllAssistantModels returns all assistant model mappings.
func getAllAssistantModels() map[string]AssistantModelConfig {
	assistantModelsMutex.RLock()
	defer assistantModelsMutex.RUnlock()
	result := make(map[string]AssistantModelConfig)
	for k, v := range assistantModels {
		result[k] = v
	}
	return result
}

type pendingRequestState struct {
	request *PendingRequest
	cancel  context.CancelFunc
}

// Pending requests tracker
var pendingRequests = make(map[string]*pendingRequestState)
var pendingMutex sync.RWMutex
var pendingCounter int64

// PendingRequest is an alias to the domain type for in-flight request tracking.
type PendingRequest = domain.PendingRequest

// Prometheus metrics
type Metrics struct {
	RequestsTotal map[string]int64 // provider:model:status -> count
	TokensTotal   map[string]int64 // provider:model:direction -> count
	DurationSumMs map[string]int64 // provider:model -> sum of ms
	DurationCount map[string]int64 // provider:model -> count
	CostTotal     float64
	CacheHits     int64
	CacheMisses   int64
	mutex         sync.RWMutex
}

var metrics = &Metrics{
	RequestsTotal: make(map[string]int64),
	TokensTotal:   make(map[string]int64),
	DurationSumMs: make(map[string]int64),
	DurationCount: make(map[string]int64),
}

func (m *Metrics) recordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Use | as separator since model names can contain :
	key := fmt.Sprintf("%s|%s|%s", provider, model, status)
	m.RequestsTotal[key]++

	if status == "success" {
		durKey := fmt.Sprintf("%s|%s", provider, model)
		m.DurationSumMs[durKey] += durationMs
		m.DurationCount[durKey]++

		inputKey := fmt.Sprintf("%s|%s|input", provider, model)
		outputKey := fmt.Sprintf("%s|%s|output", provider, model)
		m.TokensTotal[inputKey] += int64(inputTokens)
		m.TokensTotal[outputKey] += int64(outputTokens)

		m.CostTotal += cost
	}

	if cached {
		m.CacheHits++
	} else {
		m.CacheMisses++
	}
}

// RecordRequest implements the http.MetricsRecorder interface.
func (m *Metrics) RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool) {
	m.recordRequest(provider, model, status, durationMs, inputTokens, outputTokens, cost, cached)
}

// OpenAI API types - aliases to domain types
type ChatCompletionRequest = domain.ChatCompletionRequest
type Tool = domain.Tool
type ToolFunction = domain.ToolFunction
type ToolCall = domain.ToolCall
type ToolCallFunction = domain.ToolCallFunction
type Message = domain.Message
type ContentPart = domain.ContentPart
type ImageURL = domain.ImageURL
type ChatCompletionResponse = domain.ChatCompletionResponse
type Choice = domain.Choice
type Usage = domain.Usage

// withCORS wraps a handler to add CORS headers for cross-origin requests
func withCORS(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		h(w, r)
	}
}

// handleVisionModels returns only vision-capable models (local only)
func handleVisionModels(w http.ResponseWriter, r *http.Request) {
	visionModels := ollamaProvider.GetVisionModels()

	// Prefix with ollama/ for clarity
	models := make([]string, len(visionModels))
	for i, m := range visionModels {
		models[i] = "ollama/" + m
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

// Whisper API types (OpenAI-compatible)
// Whisper/TTS types - aliases to domain types
type WhisperTranscriptionResponse = domain.WhisperTranscriptionResponse
type TTSRequest = domain.TTSRequest
type RequestLog = domain.RequestLog

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func initDB() error {
	dbPath := filepath.Join(dataDir, "llm_proxy.db")

	var err error
	db, err = sql.Open("sqlite3", dbPath)
	if err != nil {
		return err
	}

	// Create table without indexes first (for existing databases)
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			request_type TEXT DEFAULT 'llm',
			provider TEXT NOT NULL,
			model TEXT NOT NULL,
			requested_model TEXT,
			sensitive INTEGER DEFAULT 0,
			precision TEXT,
			cached INTEGER DEFAULT 0,
			input_tokens INTEGER DEFAULT 0,
			output_tokens INTEGER DEFAULT 0,
			latency_ms INTEGER DEFAULT 0,
			cost_usd REAL DEFAULT 0,
			success INTEGER DEFAULT 1,
			error TEXT,
			cache_key TEXT,
			has_images INTEGER DEFAULT 0,
			request_body TEXT,
			response_body TEXT,
			voice TEXT,
			audio_duration_ms INTEGER DEFAULT 0,
			input_chars INTEGER DEFAULT 0
		)
	`)
	if err != nil {
		return err
	}

	// Add columns for existing databases (migration) - must run BEFORE index creation
	db.Exec(`ALTER TABLE requests ADD COLUMN request_body TEXT`)
	db.Exec(`ALTER TABLE requests ADD COLUMN response_body TEXT`)
	db.Exec(`ALTER TABLE requests ADD COLUMN request_type TEXT DEFAULT 'llm'`)
	db.Exec(`ALTER TABLE requests ADD COLUMN voice TEXT`)
	db.Exec(`ALTER TABLE requests ADD COLUMN audio_duration_ms INTEGER DEFAULT 0`)
	db.Exec(`ALTER TABLE requests ADD COLUMN input_chars INTEGER DEFAULT 0`)
	db.Exec(`ALTER TABLE requests ADD COLUMN usecase TEXT`)
	db.Exec(`ALTER TABLE requests ADD COLUMN is_replay INTEGER DEFAULT 0`)
	db.Exec(`ALTER TABLE requests ADD COLUMN client_ip TEXT`)

	// Create route_overrides table for usecase-specific routing
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS route_overrides (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			usecase TEXT NOT NULL,
			route_type TEXT NOT NULL,
			sensitive INTEGER NOT NULL,
			precision TEXT NOT NULL,
			provider TEXT NOT NULL,
			model TEXT NOT NULL,
			UNIQUE(usecase, route_type, sensitive, precision)
		)
	`)
	if err != nil {
		return err
	}
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_route_overrides_usecase ON route_overrides(usecase)`)

	// Create disabled_models table to track which models are turned off
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS disabled_models (
			model TEXT PRIMARY KEY NOT NULL
		)
	`)
	if err != nil {
		return err
	}

	// Create settings table for key-value configuration
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS settings (
			key TEXT PRIMARY KEY NOT NULL,
			value TEXT NOT NULL
		)
	`)
	if err != nil {
		return err
	}

	// Create assistant_models table for configurable assistant model mappings.
	// Maps assistant aliases to actual provider/model pairs.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS assistant_models (
			alias TEXT PRIMARY KEY NOT NULL,
			provider TEXT NOT NULL,
			model TEXT NOT NULL
		)
	`)
	if err != nil {
		return err
	}

	// Insert default assistant model mappings if they don't exist.
	for _, alias := range assistantAliasOrder {
		config := assistantAliasDefaults[alias]
		db.Exec(
			`INSERT OR IGNORE INTO assistant_models (alias, provider, model) VALUES (?, ?, ?)`,
			alias,
			config.Provider,
			config.Model,
		)
	}

	// Create provider_budgets table for per-provider budget limits
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS provider_budgets (
			provider TEXT PRIMARY KEY NOT NULL,
			budget_usd REAL NOT NULL,
			month_start_day INTEGER DEFAULT 1,
			enabled INTEGER DEFAULT 1,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			updated_at TEXT NOT NULL DEFAULT (datetime('now'))
		)
	`)
	if err != nil {
		return err
	}

	// Create trigger to automatically update updated_at on provider_budgets
	_, err = db.Exec(`
		CREATE TRIGGER IF NOT EXISTS update_provider_budgets_timestamp 
		AFTER UPDATE ON provider_budgets
		BEGIN
			UPDATE provider_budgets SET updated_at = datetime('now') WHERE provider = NEW.provider;
		END
	`)
	if err != nil {
		return err
	}

	// Create global_budgets table for global budget limits
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS global_budgets (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			budget_usd REAL NOT NULL,
			month_start_day INTEGER DEFAULT 1,
			enabled INTEGER DEFAULT 1,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			updated_at TEXT NOT NULL DEFAULT (datetime('now'))
		)
	`)
	if err != nil {
		return err
	}

	// Create trigger to automatically update updated_at on global_budgets
	_, err = db.Exec(`
		CREATE TRIGGER IF NOT EXISTS update_global_budgets_timestamp 
		AFTER UPDATE ON global_budgets
		BEGIN
			UPDATE global_budgets SET updated_at = datetime('now') WHERE id = NEW.id;
		END
	`)
	if err != nil {
		return err
	}

	// Fix empty provider values from routing failures
	db.Exec(`UPDATE requests SET provider = 'routing_failed' WHERE provider = '' OR provider IS NULL`)
	// Default existing rows to 'llm' type
	db.Exec(`UPDATE requests SET request_type = 'llm' WHERE request_type IS NULL OR request_type = ''`)

	// Create indexes AFTER migrations ensure columns exist
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_provider ON requests(provider)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_model ON requests(model)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_request_type ON requests(request_type)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_client_ip ON requests(client_ip)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_cached ON requests(cached)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_usecase ON requests(usecase)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_usecase_sensitive_precision ON requests(usecase, sensitive, precision)`)

	return nil
}

func loadRouteOverrides() {
	rows, err := db.Query(`SELECT usecase, route_type, sensitive, precision, provider, model FROM route_overrides`)
	if err != nil {
		log.Printf("Failed to load route overrides: %v", err)
		return
	}
	defer rows.Close()

	// Build routes map from database
	usecaseRoutes := make(map[string]map[string]map[string]map[string]*RouteConfig)

	for rows.Next() {
		var usecase, routeType, precision, provider, model string
		var sensitive int
		if err := rows.Scan(&usecase, &routeType, &sensitive, &precision, &provider, &model); err != nil {
			log.Printf("Failed to scan route override: %v", err)
			continue
		}

		sensitiveStr := "false"
		if sensitive == 1 {
			sensitiveStr = "true"
		}

		// Initialize nested maps as needed
		if usecaseRoutes[usecase] == nil {
			usecaseRoutes[usecase] = make(map[string]map[string]map[string]*RouteConfig)
		}
		if usecaseRoutes[usecase][routeType] == nil {
			usecaseRoutes[usecase][routeType] = make(map[string]map[string]*RouteConfig)
		}
		if usecaseRoutes[usecase][routeType][sensitiveStr] == nil {
			usecaseRoutes[usecase][routeType][sensitiveStr] = make(map[string]*RouteConfig)
		}

		usecaseRoutes[usecase][routeType][sensitiveStr][precision] = &RouteConfig{
			Provider: provider,
			Model:    model,
		}
	}

	// Load into router
	router.LoadUsecaseRoutes(usecaseRoutes)
	log.Printf("Loaded %d usecase route configurations", len(usecaseRoutes))
}

func saveRouteOverride(usecase, routeType string, sensitive bool, precision, provider, model string) error {
	if sensitive {
		if err := validateSensitiveRouteConfig(&RouteConfig{Provider: provider, Model: model}); err != nil {
			return fmt.Errorf("invalid sensitive route override: %w", err)
		}
	}

	dbMutex.Lock()
	defer dbMutex.Unlock()

	sensitiveInt := 0
	if sensitive {
		sensitiveInt = 1
	}

	_, err := db.Exec(`
		INSERT OR REPLACE INTO route_overrides (usecase, route_type, sensitive, precision, provider, model)
		VALUES (?, ?, ?, ?, ?, ?)
	`, usecase, routeType, sensitiveInt, precision, provider, model)
	if err != nil {
		return err
	}

	// Update router's in-memory cache
	router.SetUsecaseRoute(usecase, routeType, sensitive, precision, provider, model)
	return nil
}

func deleteRouteOverride(usecase, routeType string, sensitive bool, precision string) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	sensitiveInt := 0
	if sensitive {
		sensitiveInt = 1
	}

	_, err := db.Exec(`
		DELETE FROM route_overrides WHERE usecase = ? AND route_type = ? AND sensitive = ? AND precision = ?
	`, usecase, routeType, sensitiveInt, precision)
	if err != nil {
		return err
	}

	// Update router's in-memory cache
	router.DeleteUsecaseRoute(usecase, routeType, sensitive, precision)
	return nil
}

func getUsecaseRoute(usecase, routeType, sensitive, precision string) *RouteConfig {
	return router.GetUsecaseRoute(usecase, routeType, sensitive, precision)
}

// logRequest delegates to the requestLogger port.
func logRequest(entry *RequestLog) int64 {
	return requestLogger.LogRequest(entry)
}

func calculateCost(model string, inputTokens, outputTokens int) float64 {
	pricing, ok := modelPricing[model]
	if !ok {
		// Try case-insensitive match for models that might have different casing
		for m, p := range modelPricing {
			if strings.EqualFold(model, m) {
				pricing = p
				ok = true
				break
			}
		}
	}
	if !ok {
		return 0
	}
	return (float64(inputTokens) * pricing[0] / 1_000_000) + (float64(outputTokens) * pricing[1] / 1_000_000)
}

func generateCacheKey(req *ChatCompletionRequest, route *RouteConfig) string {
	// Hash the request content including routed provider/model
	h := sha256.New()
	h.Write([]byte(route.Provider))
	h.Write([]byte(route.Model))
	for _, msg := range req.Messages {
		h.Write([]byte(msg.Role))
		switch c := msg.Content.(type) {
		case string:
			h.Write([]byte(c))
		default:
			b, _ := json.Marshal(c)
			h.Write(b)
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Cache wrapper functions - delegate to the requestCache port
func getCached(key string) ([]byte, bool) {
	return requestCache.Get(key)
}

func setCache(key string, request, response []byte) {
	requestCache.Set(key, request, response)
}

func getCachedRequest(key string) ([]byte, bool) {
	return requestCache.GetRequest(key)
}

func getCachedResponse(key string) ([]byte, bool) {
	return requestCache.GetResponse(key)
}

// Pending request management
func addPendingRequest(req *ChatCompletionRequest, route *RouteConfig, startTime time.Time, cancel context.CancelFunc) string {
	pendingMutex.Lock()
	defer pendingMutex.Unlock()

	pendingCounter++
	id := fmt.Sprintf("req-%d", pendingCounter)

	// Extract preview from first user message
	preview := ""
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			switch c := msg.Content.(type) {
			case string:
				preview = c
			case []interface{}:
				for _, part := range c {
					if m, ok := part.(map[string]interface{}); ok {
						if m["type"] == "text" {
							if text, ok := m["text"].(string); ok {
								preview = text
								break
							}
						}
					}
				}
			}
			break
		}
	}
	if len(preview) > 100 {
		preview = preview[:100] + "..."
	}

	sensitive := true // Default to sensitive (local Ollama) for privacy
	if req.Sensitive != nil {
		sensitive = *req.Sensitive
	}

	pendingRequests[id] = &pendingRequestState{
		request: &PendingRequest{
			ID:        id,
			StartTime: startTime,
			Provider:  route.Provider,
			Model:     route.Model,
			HasImages: req.HasImages(),
			Sensitive: sensitive,
			Precision: req.Precision,
			Usecase:   req.Usecase,
			Preview:   preview,
		},
		cancel: cancel,
	}

	return id
}

// addPendingSTTRequest adds a pending STT request to the tracking map.
func addPendingSTTRequest(provider, model string, sensitive bool, startTime time.Time, cancel context.CancelFunc) string {
	pendingMutex.Lock()
	defer pendingMutex.Unlock()

	pendingCounter++
	id := fmt.Sprintf("stt-%d", pendingCounter)

	pendingRequests[id] = &pendingRequestState{
		request: &PendingRequest{
			ID:        id,
			StartTime: startTime,
			Provider:  provider,
			Model:     model,
			HasImages: false,
			Sensitive: sensitive,
			Precision: "",
			Usecase:   "",
			Preview:   "STT transcription",
		},
		cancel: cancel,
	}

	return id
}

func removePendingRequest(id string) {
	pendingMutex.Lock()
	defer pendingMutex.Unlock()
	delete(pendingRequests, id)
}

func getPendingRequests() []*PendingRequest {
	pendingMutex.RLock()
	defer pendingMutex.RUnlock()

	result := make([]*PendingRequest, 0, len(pendingRequests))
	for _, req := range pendingRequests {
		result = append(result, req.request)
	}
	// Sort by StartTime ascending (oldest first) for stable display order
	sort.Slice(result, func(i, j int) bool {
		return result[i].StartTime.Before(result[j].StartTime)
	})
	return result
}

func cancelPendingRequest(id string) bool {
	pendingMutex.RLock()
	req, ok := pendingRequests[id]
	pendingMutex.RUnlock()
	if !ok || req.cancel == nil {
		return false
	}
	req.cancel()
	return true
}

// resolveRoute delegates to the router service.
func resolveRoute(req *ChatCompletionRequest) (*RouteConfig, error) {
	return router.ResolveRoute(req)
}

// ChatProvider is an alias to the port interface.
type ChatProvider = ports.ChatProvider

// chatProviders maps provider names to their implementations.
var chatProviders map[string]ChatProvider

// ollamaProvider is stored separately for model discovery functions.
var ollamaProvider *providers.OllamaProvider

// ollamaCloudProvider is a virtual provider backed by the local Ollama daemon.
var ollamaCloudProvider *providers.OllamaCloudProvider

// llamacppProvider is stored separately for health checks and backend switching.
var llamacppProvider *providers.LlamaCppProvider
var llamacppVisionProvider *providers.LlamaCppProvider

func initChatProviders() {
	ollamaProvider = providers.NewOllamaProvider(ollamaHost, chatTimeout)
	ollamaCloudProvider = providers.NewOllamaCloudProvider(ollamaHost, chatTimeout)
	chatProviders = map[string]ChatProvider{
		"openai":       providers.NewOpenAIProvider(openaiKey, openaiTimeout, openaiStreamingTimeout),
		"anthropic":    providers.NewAnthropicProvider(anthropicKey, anthropicTimeout),
		"ollama":       ollamaProvider,
		"ollama-cloud": ollamaCloudProvider,
		"gemini":       providers.NewGeminiProvider(geminiKey, geminiTimeout),
	}

	// If llama.cpp host is configured, add it as a provider
	if llamacppHost != "" {
		llamacppProvider = providers.NewLlamaCppProvider(llamacppHost, llamacppTimeout)
		chatProviders["llamacpp"] = llamacppProvider
		log.Printf("llama.cpp text provider configured at %s", llamacppHost)
	}

	// If llama.cpp vision host is configured, add it as a separate provider
	if llamacppVisionHost != "" {
		llamacppVisionProvider = providers.NewLlamaCppProvider(llamacppVisionHost, llamacppTimeout)
		chatProviders["llamacpp-vision"] = llamacppVisionProvider
		log.Printf("llama.cpp vision provider configured at %s", llamacppVisionHost)
	}

	// If MLX host is configured, add it as a provider
	if mlxHost != "" {
		chatProviders["mlx"] = providers.NewMLXProvider(mlxHost, mlxTimeout)
		log.Printf("MLX provider configured at %s", mlxHost)
	}

	// If Together.ai key is configured, add it as a provider
	if togetherKey != "" {
		chatProviders["together"] = providers.NewTogetherProvider(togetherKey, togetherTimeout)
		log.Printf("Together.ai provider configured")
	}

	// If Baseten key is configured, add it as a provider
	if basetenKey != "" {
		chatProviders["baseten"] = providers.NewBasetenProvider(basetenKey, basetenTimeout)
		log.Printf("Baseten provider configured")
	}
}

// enableVisionBackendSwitch registers a selectable provider for local vision
// that delegates to either Ollama or llama.cpp based on the backend preference.
func enableVisionBackendSwitch() {
	if llamacppHost == "" || llamacppProvider == nil {
		return
	}

	// Create a selectable provider that delegates based on preference
	selectableProvider := providers.NewSelectableProvider(
		ollamaProvider,
		llamacppProvider,
		getLocalVisionBackend,
	)
	chatProviders["local-vision"] = selectableProvider

	// Update sensitive vision routes to use selectable provider
	for precision, config := range visionRoutingTable["true"] {
		if config != nil && config.Provider == "ollama" && config.Model == "qwen3-vl:30b" {
			visionRoutingTable["true"][precision] = &RouteConfig{
				Provider: "local-vision",
				Model:    "qwen3-vl:30b",
			}
			log.Printf("Vision route sensitive=true precision=%s now uses selectable backend", precision)
		}
	}

	// Also update non-sensitive low precision route if it uses ollama
	if config := visionRoutingTable["false"]["low"]; config != nil && config.Provider == "ollama" {
		visionRoutingTable["false"]["low"] = &RouteConfig{
			Provider: "local-vision",
			Model:    config.Model,
		}
		log.Printf("Vision route sensitive=false precision=low now uses selectable backend")
	}
}

// Whisper transcription handlers

// getClientIP extracts the client IP address from the request.
// It checks X-Forwarded-For and X-Real-IP headers first (for proxied requests),
// then falls back to RemoteAddr.
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (may contain multiple IPs, take the first)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr (strip port if present)
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr // Return as-is if no port
	}
	return ip
}

func isModelLocal(model string) bool {
	if strings.HasPrefix(model, "ollama-cloud/") || providers.IsOllamaCloudModel(model) {
		return false
	}
	if strings.HasPrefix(model, "ollama/") {
		return true
	}
	if strings.HasPrefix(model, "claude") || strings.HasPrefix(model, "gemini") {
		return false
	}
	if strings.Contains(model, ":") || model == "llama3" || model == "llava" || strings.HasPrefix(model, "qwen") {
		return true
	}
	// Default to false (cloud provider)
	return false
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	models := []map[string]interface{}{}
	seen := make(map[string]bool)

	// Add routing models
	models = append(models, map[string]interface{}{
		"id":       "auto",
		"object":   "model",
		"owned_by": "llm-proxy",
		"local":    false,
	})
	seen["auto"] = true

	// Add known models from pricing
	for model := range modelPricing {
		if !seen[model] {
			models = append(models, map[string]interface{}{
				"id":       model,
				"object":   "model",
				"owned_by": "llm-proxy",
				"local":    isModelLocal(model),
			})
			seen[model] = true
		}
	}

	// Add all Ollama models dynamically
	for _, model := range ollamaProvider.GetModels() {
		if !seen[model] {
			models = append(models, map[string]interface{}{
				"id":       model,
				"object":   "model",
				"owned_by": "llm-proxy",
				"local":    isModelLocal(model),
			})
			seen[model] = true
		}
	}

	for _, model := range ollamaCloudProvider.GetModels() {
		if !seen[model] {
			models = append(models, map[string]interface{}{
				"id":       model,
				"object":   "model",
				"owned_by": "llm-proxy",
				"local":    false,
			})
			seen[model] = true
		}
	}

	// Add llama.cpp text models if the provider is configured
	if llamacppProvider != nil {
		if llamaModels, err := llamacppProvider.GetModels(); err == nil {
			for _, model := range llamaModels {
				prefixedModel := "llamacpp/" + model
				if !seen[prefixedModel] {
					models = append(models, map[string]interface{}{
						"id":       prefixedModel,
						"object":   "model",
						"owned_by": "llm-proxy",
						"local":    true,
					})
					seen[prefixedModel] = true
				}
			}
		}
	}

	// Add llama.cpp vision models if the provider is configured
	if llamacppVisionProvider != nil {
		if llamaModels, err := llamacppVisionProvider.GetModels(); err == nil {
			for _, model := range llamaModels {
				prefixedModel := "llamacpp-vision/" + model
				if !seen[prefixedModel] {
					models = append(models, map[string]interface{}{
						"id":           prefixedModel,
						"object":       "model",
						"owned_by":     "llm-proxy",
						"local":        true,
						"vision_model": true,
						"capabilities": []string{"vision"},
					})
					seen[prefixedModel] = true
				}
			}
		}
	}

	// Add MLX models if the provider is configured
	if mlxProvider, ok := chatProviders["mlx"].(*providers.MLXProvider); ok {
		if mlxModels, err := mlxProvider.GetModels(); err == nil {
			for _, model := range mlxModels {
				// Prefix with "mlx/" so router knows to use MLX provider
				prefixedModel := "mlx/" + model
				if !seen[prefixedModel] {
					models = append(models, map[string]interface{}{
						"id":       prefixedModel,
						"object":   "model",
						"owned_by": "llm-proxy",
						"local":    true, // MLX runs locally on Apple Silicon
					})
					seen[prefixedModel] = true
				}
			}
		}
	}

	// Sort models alphabetically by ID
	sort.Slice(models, func(i, j int) bool {
		return models[i]["id"].(string) < models[j]["id"].(string)
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// EstimateRequest is a simplified request for route estimation
type EstimateRequest struct {
	Model     string `json:"model"`
	Sensitive *bool  `json:"sensitive,omitempty"`
	Precision string `json:"precision,omitempty"`
	Usecase   string `json:"usecase,omitempty"`
	HasImages bool   `json:"has_images,omitempty"`
}

// EstimateResponse returns the estimated model and time
type EstimateResponse struct {
	Provider         string  `json:"provider"`
	Model            string  `json:"model"`
	EstimatedSeconds float64 `json:"estimated_seconds"`
	SampleCount      int64   `json:"sample_count"`
	IsLocal          bool    `json:"is_local"`
}

func handleEstimate(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EstimateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Build a minimal ChatCompletionRequest for route resolution
	chatReq := &ChatCompletionRequest{
		Model:     req.Model,
		Sensitive: req.Sensitive,
		Precision: req.Precision,
		Usecase:   req.Usecase,
	}

	// If has_images is true, add a dummy image message to trigger vision routing
	if req.HasImages {
		chatReq.Messages = []Message{{
			Role: "user",
			Content: []ContentPart{{
				Type:     "image_url",
				ImageURL: &ImageURL{URL: "data:image/png;base64,dummy"},
			}},
		}}
	}

	route, err := resolveRoute(chatReq)
	if err != nil {
		http.Error(w, "Route resolution failed: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Get average duration from metrics
	durKey := fmt.Sprintf("%s|%s", route.Provider, route.Model)
	metrics.mutex.RLock()
	sumMs := metrics.DurationSumMs[durKey]
	count := metrics.DurationCount[durKey]
	metrics.mutex.RUnlock()

	var estimatedSeconds float64
	if count > 0 {
		estimatedSeconds = float64(sumMs) / float64(count) / 1000.0
	} else {
		// Default estimates if no history
		switch route.Provider {
		case "ollama":
			estimatedSeconds = 30.0 // Local models are slower
		case "anthropic":
			estimatedSeconds = 10.0
		case "openai":
			estimatedSeconds = 8.0
		default:
			estimatedSeconds = 15.0
		}
	}

	isLocal := route.Provider == "ollama"

	resp := EstimateResponse{
		Provider:         route.Provider,
		Model:            route.Model,
		EstimatedSeconds: estimatedSeconds,
		SampleCount:      count,
		IsLocal:          isLocal,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleReplayRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get original request ID and optional model override
	var replayReq struct {
		RequestID int64  `json:"request_id"`
		Model     string `json:"model"` // Optional: override model for replay
	}
	if err := json.NewDecoder(r.Body).Decode(&replayReq); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Get original request's cache_key and request_body from database
	dbMutex.Lock()
	var cacheKey sql.NullString
	var requestBody sql.NullString
	err := db.QueryRow(`SELECT cache_key, request_body FROM requests WHERE id = ?`, replayReq.RequestID).Scan(&cacheKey, &requestBody)
	dbMutex.Unlock()

	if err != nil {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Get the original request body - try cache first, then DB
	var reqBody []byte
	if cacheKey.Valid {
		reqBody, _ = getCachedRequest(cacheKey.String)
	}
	if reqBody == nil && requestBody.Valid {
		reqBody = []byte(requestBody.String)
	}
	if reqBody == nil {
		http.Error(w, "Original request content not available - cannot replay", http.StatusBadRequest)
		return
	}

	// Parse original request
	var origReq ChatCompletionRequest
	if err := json.Unmarshal(reqBody, &origReq); err != nil {
		http.Error(w, "Failed to parse request body", http.StatusInternalServerError)
		return
	}

	// Apply model override if specified
	if replayReq.Model != "" {
		origReq.Model = replayReq.Model
	}

	// Force no-cache to get fresh response
	origReq.NoCache = true

	// Forward to chat completions handler by creating internal request
	reqBytes, _ := json.Marshal(origReq)
	internalReq, _ := http.NewRequestWithContext(r.Context(), "POST", "/v1/chat/completions", bytes.NewReader(reqBytes))
	internalReq.Header.Set("Content-Type", "application/json")
	internalReq.Header.Set("X-LLM-Proxy-Replay", "true")

	// Use response recorder to capture the response
	recorder := &responseRecorder{
		ResponseWriter: w,
		statusCode:     http.StatusOK,
	}

	chatHandler.ServeHTTP(recorder, internalReq)
}

// responseRecorder wraps ResponseWriter to capture status code
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
}

func handleRoutes(w http.ResponseWriter, r *http.Request) {
	usecase := r.URL.Query().Get("usecase")
	routes := []map[string]interface{}{}

	// Text routes
	for sensitive, precisions := range routingTable {
		for precision, config := range precisions {
			entry := map[string]interface{}{
				"type":      "text",
				"sensitive": sensitive == "true",
				"precision": precision,
				"available": config != nil,
			}
			// Store base config
			if config != nil {
				entry["base_provider"] = config.Provider
				entry["base_model"] = config.Model
				entry["provider"] = config.Provider
				entry["model"] = config.Model
			}
			// Check for usecase override
			if usecase != "" {
				if override := getUsecaseRoute(usecase, "text", sensitive, precision); override != nil {
					entry["provider"] = override.Provider
					entry["model"] = override.Model
					entry["overridden"] = true
				}
			}
			routes = append(routes, entry)
		}
	}

	// Vision routes
	for sensitive, precisions := range visionRoutingTable {
		for precision, config := range precisions {
			entry := map[string]interface{}{
				"type":      "vision",
				"sensitive": sensitive == "true",
				"precision": precision,
				"available": config != nil,
			}
			// Store base config
			if config != nil {
				entry["base_provider"] = config.Provider
				entry["base_model"] = config.Model
				entry["provider"] = config.Provider
				entry["model"] = config.Model
			}
			// Check for usecase override
			if usecase != "" {
				if override := getUsecaseRoute(usecase, "vision", sensitive, precision); override != nil {
					entry["provider"] = override.Provider
					entry["model"] = override.Model
					entry["overridden"] = true
				}
			}
			routes = append(routes, entry)
		}
	}

	// TTS routes (always local Kokoro for now)
	routes = append(routes, map[string]interface{}{
		"type":          "tts",
		"sensitive":     true,
		"provider":      "kokoro",
		"model":         "kokoro-tts",
		"base_provider": "kokoro",
		"base_model":    "kokoro-tts",
		"available":     true,
	})
	routes = append(routes, map[string]interface{}{
		"type":          "tts",
		"sensitive":     false,
		"provider":      "kokoro",
		"model":         "kokoro-tts",
		"base_provider": "kokoro",
		"base_model":    "kokoro-tts",
		"available":     true,
	})

	// STT routes
	routes = append(routes, map[string]interface{}{
		"type":          "stt",
		"sensitive":     true,
		"provider":      "local",
		"model":         "whisper-large-v3",
		"base_provider": "local",
		"base_model":    "whisper-large-v3",
		"available":     true,
	})
	routes = append(routes, map[string]interface{}{
		"type":          "stt",
		"sensitive":     false,
		"provider":      "openai",
		"model":         "whisper-1",
		"base_provider": "openai",
		"base_model":    "whisper-1",
		"available":     true,
	})

	// Sort for consistent output
	sort.Slice(routes, func(i, j int) bool {
		si := fmt.Sprintf("%s-%v-%s", routes[i]["type"], routes[i]["sensitive"], routes[i]["precision"])
		sj := fmt.Sprintf("%s-%v-%s", routes[j]["type"], routes[j]["sensitive"], routes[j]["precision"])
		return si < sj
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(routes)
}

// handleRouteOverride handles setting/deleting route overrides for usecases
func handleRouteOverride(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var req struct {
			Usecase   string `json:"usecase"`
			Type      string `json:"type"` // text, vision
			Sensitive bool   `json:"sensitive"`
			Precision string `json:"precision"`
			Provider  string `json:"provider"`
			Model     string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}

		if req.Usecase == "" {
			http.Error(w, "usecase is required", http.StatusBadRequest)
			return
		}
		if req.Type != "text" && req.Type != "vision" {
			http.Error(w, "type must be 'text' or 'vision'", http.StatusBadRequest)
			return
		}
		if req.Precision == "" {
			http.Error(w, "precision is required", http.StatusBadRequest)
			return
		}
		if req.Provider == "" || req.Model == "" {
			http.Error(w, "provider and model are required", http.StatusBadRequest)
			return
		}

		if err := saveRouteOverride(req.Usecase, req.Type, req.Sensitive, req.Precision, req.Provider, req.Model); err != nil {
			http.Error(w, "Failed to save override: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		return
	}

	if r.Method == "DELETE" {
		usecase := r.URL.Query().Get("usecase")
		routeType := r.URL.Query().Get("type")
		sensitiveStr := r.URL.Query().Get("sensitive")
		precision := r.URL.Query().Get("precision")

		if usecase == "" || routeType == "" || precision == "" {
			http.Error(w, "usecase, type, and precision are required", http.StatusBadRequest)
			return
		}

		sensitive := sensitiveStr == "true"

		if err := deleteRouteOverride(usecase, routeType, sensitive, precision); err != nil {
			http.Error(w, "Failed to delete override: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// handleUsecases returns list of usecases that have route overrides
func handleUsecases(w http.ResponseWriter, r *http.Request) {
	usecaseRoutes := router.GetAllUsecaseRoutes()

	usecases := []string{}
	for usecase := range usecaseRoutes {
		usecases = append(usecases, usecase)
	}
	sort.Strings(usecases)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(usecases)
}

// handleAvailableModels returns list of available models for each provider
func handleAvailableModels(w http.ResponseWriter, r *http.Request) {
	models := map[string][]string{
		"anthropic": {
			"claude-opus-4-6",
			"claude-opus-4-5-20251101",
			"claude-opus-4-1-20250805",
			"claude-opus-4-20250514",
			"claude-sonnet-4-6",
			"claude-sonnet-4-5-20250929",
			"claude-sonnet-4-20250514",
			"claude-haiku-4-5-20251001",
		},
		"openai": {
			"gpt-5.4",
			"gpt-5.2",
			"gpt-5.1",
			"gpt-5",
			"gpt-5-pro",
			"gpt-5-mini",
			"gpt-5-nano",
			"gpt-4.1",
			"gpt-4.1-mini",
			"gpt-4.1-nano",
			"gpt-4o",
			"gpt-4o-mini",
			"o1",
			"o1-pro",
			"o3",
			"o3-mini",
			"o4-mini",
		},
		"ollama":       ollamaProvider.GetModels(),
		"ollama-cloud": ollamaCloudProvider.GetModels(),
	}

	if mlxProvider, ok := chatProviders["mlx"].(*providers.MLXProvider); ok {
		if mlxModels, err := mlxProvider.GetModels(); err == nil {
			models["mlx"] = mlxModels
		} else {
			log.Printf("Failed to fetch MLX models: %v", err)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

// handleModelsConfig returns all models with their enabled/disabled status
// GET: returns {models: [{provider, model, enabled},...]}
// POST: sets enabled status for a model {model: string, enabled: bool}
func handleModelsConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		// Build list of all models with their enabled status
		type ModelConfig struct {
			Provider string `json:"provider"`
			Model    string `json:"model"`
			Enabled  bool   `json:"enabled"`
		}
		var configs []ModelConfig

		// Anthropic models
		anthropicModels := []string{
			"claude-opus-4-6",
			"claude-opus-4-5-20251101",
			"claude-opus-4-1-20250805",
			"claude-opus-4-20250514",
			"claude-sonnet-4-6",
			"claude-sonnet-4-5-20250929",
			"claude-sonnet-4-20250514",
			"claude-haiku-4-5-20251001",
		}
		for _, m := range anthropicModels {
			configs = append(configs, ModelConfig{
				Provider: "anthropic",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
		}

		// OpenAI models
		openaiModels := []string{
			"gpt-5.4",
			"gpt-5.2",
			"gpt-5.1",
			"gpt-5",
			"gpt-5-pro",
			"gpt-5-mini",
			"gpt-5-nano",
			"gpt-4.1",
			"gpt-4.1-mini",
			"gpt-4.1-nano",
			"gpt-4o",
			"gpt-4o-mini",
			"o1",
			"o1-pro",
			"o3",
			"o3-mini",
			"o4-mini",
		}
		for _, m := range openaiModels {
			configs = append(configs, ModelConfig{
				Provider: "openai",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
		}

		// Ollama models
		ollamaModels := ollamaProvider.GetModels()
		for _, m := range ollamaModels {
			configs = append(configs, ModelConfig{
				Provider: "ollama",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
		}

		ollamaCloudModels := ollamaCloudProvider.GetModels()
		for _, m := range ollamaCloudModels {
			configs = append(configs, ModelConfig{
				Provider: "ollama-cloud",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
		}

		if mlxProvider, ok := chatProviders["mlx"].(*providers.MLXProvider); ok {
			if mlxModels, err := mlxProvider.GetModels(); err == nil {
				for _, m := range mlxModels {
					configs = append(configs, ModelConfig{
						Provider: "mlx",
						Model:    m,
						Enabled:  !isModelDisabled(m),
					})
				}
			} else {
				log.Printf("Failed to fetch MLX models for config: %v", err)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"models": configs})
		return
	}

	if r.Method == "POST" {
		var req struct {
			Model   string `json:"model"`
			Enabled bool   `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		dbMutex.Lock()
		defer dbMutex.Unlock()

		if req.Enabled {
			// Remove from disabled_models table
			_, err := db.Exec(`DELETE FROM disabled_models WHERE model = ?`, req.Model)
			if err != nil {
				http.Error(w, "Database error", http.StatusInternalServerError)
				return
			}
			disabledModelsMutex.Lock()
			delete(disabledModels, req.Model)
			disabledModelsMutex.Unlock()
			log.Printf("Enabled model: %s", req.Model)
		} else {
			// Add to disabled_models table
			_, err := db.Exec(`INSERT OR IGNORE INTO disabled_models (model) VALUES (?)`, req.Model)
			if err != nil {
				http.Error(w, "Database error", http.StatusInternalServerError)
				return
			}
			disabledModelsMutex.Lock()
			disabledModels[req.Model] = true
			disabledModelsMutex.Unlock()
			log.Printf("Disabled model: %s", req.Model)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]bool{"success": true})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// handleBackend returns or sets the local vision backend preference (ollama vs llamacpp).
// GET: returns {"backend": "ollama" or "llamacpp", "available": bool}
// POST: sets backend with {"backend": "ollama" or "llamacpp"}
func handleBackend(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"backend":   getLocalVisionBackend(),
			"available": llamacppHost != "" && llamacppProvider != nil,
		})
		return
	}

	if r.Method == "POST" {
		var req struct {
			Backend string `json:"backend"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if err := setLocalVisionBackend(req.Backend); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"backend": getLocalVisionBackend(),
		})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// handleLockdownMode returns or sets the lockdown preference.
// GET: returns {"enabled": bool}
// POST: sets lockdown state with {"enabled": bool}
func handleLockdownMode(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"enabled": getLockdownMode(),
		})
		return
	}

	if r.Method == "POST" {
		var req struct {
			Enabled bool `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if err := setLockdownMode(req.Enabled); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"enabled": getLockdownMode(),
		})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// handleAssistantModels returns or sets assistant model mappings.
// GET: returns all assistant model mappings
// POST: sets a single mapping with {"alias": "assistant", "provider": "ollama", "model": "qwen3.5:122b"}
func handleAssistantModels(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(getAllAssistantModels())
		return
	}

	if r.Method == "POST" {
		var req struct {
			Alias    string `json:"alias"`
			Provider string `json:"provider"`
			Model    string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.Alias == "" || req.Provider == "" || req.Model == "" {
			http.Error(w, "alias, provider, and model are required", http.StatusBadRequest)
			return
		}

		if err := setAssistantModel(req.Alias, req.Provider, req.Model); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"alias":   req.Alias,
			"config":  AssistantModelConfig{Provider: req.Provider, Model: req.Model},
		})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// handleServerConfig returns server configuration values.
// GET: returns all timeout and configuration values
func handleServerConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ollama_host":              ollamaHost,
		"whisper_server_url":       whisperServerURL,
		"chat_timeout":             chatTimeout,
		"speech_timeout":           speechTimeout,
		"speech_streaming_timeout": speechStreamingTimeout,
		"aida_timeout":             aidaTimeout,
		"gemini_timeout":           geminiTimeout,
		"openai_timeout":           openaiTimeout,
		"openai_streaming_timeout": openaiStreamingTimeout,
		"anthropic_timeout":        anthropicTimeout,
		"image_gen_timeout":        imageGenTimeout,
		"tts_timeout":              ttsTimeout,
		"tts_kokoro_timeout":       ttsKokoroTimeout,
		"web_search_timeout":       webSearchTimeout,
		"llamacpp_timeout":         llamacppTimeout,
	})
}

// handleLlamaCppInfo returns llama.cpp server info (model, slots, status).
func handleLlamaCppInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if llamacppHost == "" || llamacppProvider == nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"available": false,
			"error":     "llama.cpp not configured",
		})
		return
	}

	// Fetch /props from llama-server
	propsResp, err := http.Get("http://" + llamacppHost + "/props")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"available": false,
			"healthy":   false,
			"error":     err.Error(),
		})
		return
	}
	defer propsResp.Body.Close()

	var props map[string]interface{}
	if err := json.NewDecoder(propsResp.Body).Decode(&props); err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"available": true,
			"healthy":   false,
			"error":     "failed to parse props: " + err.Error(),
		})
		return
	}

	// Fetch /slots from llama-server
	slotsResp, err := http.Get("http://" + llamacppHost + "/slots")
	var slots []map[string]interface{}
	if err == nil {
		defer slotsResp.Body.Close()
		json.NewDecoder(slotsResp.Body).Decode(&slots)
	}

	// Count active/idle slots
	totalSlots := len(slots)
	activeSlots := 0
	for _, slot := range slots {
		if processing, ok := slot["is_processing"].(bool); ok && processing {
			activeSlots++
		}
	}

	result := map[string]interface{}{
		"available":    true,
		"healthy":      llamacppProvider.IsHealthy(),
		"host":         llamacppHost,
		"model_alias":  props["model_alias"],
		"model_path":   props["model_path"],
		"total_slots":  props["total_slots"],
		"active_slots": activeSlots,
		"idle_slots":   totalSlots - activeSlots,
		"modalities":   props["modalities"],
		"build_info":   props["build_info"],
	}

	json.NewEncoder(w).Encode(result)
}

// handleLlamaCppMetrics returns llama.cpp Prometheus metrics.
func handleLlamaCppMetrics(w http.ResponseWriter, r *http.Request) {
	if llamacppHost == "" || llamacppProvider == nil {
		http.Error(w, "llama.cpp not configured", http.StatusServiceUnavailable)
		return
	}

	// Fetch /metrics from llama-server
	metricsResp, err := http.Get("http://" + llamacppHost + "/metrics")
	if err != nil {
		http.Error(w, "Failed to fetch metrics: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer metricsResp.Body.Close()

	// Forward the Prometheus metrics
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	io.Copy(w, metricsResp.Body)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "service": "llm-proxy"})
}

// handleTTSCacheStats returns TTS audio cache statistics
func handleTTSCacheStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if ttsAudioCache == nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"enabled":    false,
			"hits":       0,
			"misses":     0,
			"hit_rate":   0.0,
			"size_bytes": 0,
			"size_mb":    0.0,
		})
		return
	}

	hits, misses, sizeBytes := ttsAudioCache.Stats()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"enabled":     true,
		"hits":        hits,
		"misses":      misses,
		"hit_rate":    hitRate,
		"size_bytes":  sizeBytes,
		"size_mb":     float64(sizeBytes) / (1024 * 1024),
		"max_size_mb": ttsCacheMaxSize,
		"ttl_days":    ttsCacheTTLDays,
	})
}

// handleTTSCacheClear clears the TTS audio cache
func handleTTSCacheClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" && r.Method != "DELETE" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	if ttsAudioCache == nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "TTS cache not enabled",
		})
		return
	}

	if err := ttsAudioCache.Clear(); err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   err.Error(),
		})
		return
	}

	log.Printf("TTS audio cache cleared")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "TTS cache cleared",
	})
}

// handleSystemMetrics returns CPU, memory, and system info
func handleSystemMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Get CPU percent (1 second sample)
	cpuPercent, err := cpu.Percent(time.Second, false)
	cpuUsage := 0.0
	if err == nil && len(cpuPercent) > 0 {
		cpuUsage = cpuPercent[0]
	}

	// Get per-core CPU
	perCoreCPU, _ := cpu.Percent(0, true)

	// Get memory stats
	memStats, err := mem.VirtualMemory()
	memUsage := 0.0
	memTotal := uint64(0)
	memUsed := uint64(0)
	if err == nil {
		memUsage = memStats.UsedPercent
		memTotal = memStats.Total
		memUsed = memStats.Used
	}

	// Get host info
	hostInfo, _ := host.Info()
	hostname := ""
	uptime := uint64(0)
	platform := ""
	if hostInfo != nil {
		hostname = hostInfo.Hostname
		uptime = hostInfo.Uptime
		platform = hostInfo.Platform + " " + hostInfo.PlatformVersion
	}

	// Try to get GPU info via nvidia-smi (if available)
	gpuUsage := -1.0 // -1 means not available
	gpuMemUsage := -1.0
	gpuTemp := -1.0
	gpuName := ""

	// Check for NVIDIA GPU via nvidia-smi
	if out, err := runCommand("nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,name", "--format=csv,noheader,nounits"); err == nil {
		lines := strings.Split(strings.TrimSpace(out), "\n")
		if len(lines) > 0 {
			parts := strings.Split(lines[0], ", ")
			if len(parts) >= 4 {
				fmt.Sscanf(parts[0], "%f", &gpuUsage)
				fmt.Sscanf(parts[1], "%f", &gpuMemUsage)
				fmt.Sscanf(parts[2], "%f", &gpuTemp)
				gpuName = strings.TrimSpace(parts[3])
			}
		}
	}

	result := map[string]interface{}{
		"cpu_percent":        cpuUsage,
		"cpu_per_core":       perCoreCPU,
		"memory_percent":     memUsage,
		"memory_total_bytes": memTotal,
		"memory_used_bytes":  memUsed,
		"hostname":           hostname,
		"uptime_seconds":     uptime,
		"platform":           platform,
		"timestamp":          time.Now().UTC().Format(time.RFC3339),
	}

	// Only include GPU stats if available
	if gpuUsage >= 0 {
		result["gpu_percent"] = gpuUsage
		result["gpu_memory_percent"] = gpuMemUsage
		result["gpu_temp_celsius"] = gpuTemp
		result["gpu_name"] = gpuName
	}

	json.NewEncoder(w).Encode(result)
}

// handleTruncateLogs truncates the requests table to the last N entries
func handleTruncateLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Default to 5000 if not specified
	keepCount := 5000
	if countStr := r.URL.Query().Get("keep"); countStr != "" {
		if parsed, err := strconv.Atoi(countStr); err == nil && parsed > 0 {
			keepCount = parsed
		}
	}

	// Count before truncation
	var countBefore int
	db.QueryRow("SELECT COUNT(*) FROM requests").Scan(&countBefore)

	// Delete all but the last N requests
	result, err := db.Exec(`
		DELETE FROM requests WHERE id NOT IN (
			SELECT id FROM requests ORDER BY id DESC LIMIT ?
		)
	`, keepCount)

	if err != nil {
		log.Printf("Failed to truncate logs: %v", err)
		http.Error(w, "Failed to truncate: "+err.Error(), http.StatusInternalServerError)
		return
	}

	deleted, _ := result.RowsAffected()

	// Vacuum to reclaim space
	db.Exec("VACUUM")

	log.Printf("Truncated logs: deleted %d rows, kept %d", deleted, keepCount)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":      true,
		"deleted":      deleted,
		"kept":         keepCount,
		"count_before": countBefore,
		"count_after":  countBefore - int(deleted),
		"db_size":      getDbSize(),
	})
}

// handleDbStats returns database statistics
func handleDbStats(w http.ResponseWriter, r *http.Request) {
	var requestCount int
	db.QueryRow("SELECT COUNT(*) FROM requests").Scan(&requestCount)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"request_count": requestCount,
		"db_size":       getDbSize(),
		"db_size_human": formatBytes(getDbSize()),
	})
}

// getDbSize returns the database file size in bytes
func getDbSize() int64 {
	dbPath := filepath.Join(dataDir, "llm_proxy.db")
	if info, err := os.Stat(dbPath); err == nil {
		return info.Size()
	}
	return 0
}

// formatBytes formats bytes into human readable string
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// runCommand runs a command and returns stdout
func runCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.Output()
	return string(out), err
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	var sb strings.Builder

	// ===== IN-MEMORY METRICS (since service start) =====
	metrics.mutex.RLock()

	// Requests total (in-memory since restart)
	sb.WriteString("# HELP llm_proxy_requests_total Total number of requests since service start\n")
	sb.WriteString("# TYPE llm_proxy_requests_total counter\n")
	for key, count := range metrics.RequestsTotal {
		parts := strings.SplitN(key, "|", 3)
		if len(parts) == 3 {
			sb.WriteString(fmt.Sprintf("llm_proxy_requests_total{provider=\"%s\",model=\"%s\",status=\"%s\"} %d\n",
				parts[0], parts[1], parts[2], count))
		}
	}

	// Tokens total
	sb.WriteString("\n# HELP llm_proxy_tokens_total Total tokens processed since service start\n")
	sb.WriteString("# TYPE llm_proxy_tokens_total counter\n")
	for key, count := range metrics.TokensTotal {
		parts := strings.SplitN(key, "|", 3)
		if len(parts) == 3 {
			sb.WriteString(fmt.Sprintf("llm_proxy_tokens_total{provider=\"%s\",model=\"%s\",direction=\"%s\"} %d\n",
				parts[0], parts[1], parts[2], count))
		}
	}

	// Duration (as sum and count for calculating average)
	sb.WriteString("\n# HELP llm_proxy_request_duration_ms_sum Sum of request durations in milliseconds\n")
	sb.WriteString("# TYPE llm_proxy_request_duration_ms_sum counter\n")
	for key, sum := range metrics.DurationSumMs {
		parts := strings.SplitN(key, "|", 2)
		if len(parts) == 2 {
			sb.WriteString(fmt.Sprintf("llm_proxy_request_duration_ms_sum{provider=\"%s\",model=\"%s\"} %d\n",
				parts[0], parts[1], sum))
		}
	}

	sb.WriteString("\n# HELP llm_proxy_request_duration_ms_count Number of requests for duration calculation\n")
	sb.WriteString("# TYPE llm_proxy_request_duration_ms_count counter\n")
	for key, count := range metrics.DurationCount {
		parts := strings.SplitN(key, "|", 2)
		if len(parts) == 2 {
			sb.WriteString(fmt.Sprintf("llm_proxy_request_duration_ms_count{provider=\"%s\",model=\"%s\"} %d\n",
				parts[0], parts[1], count))
		}
	}

	// Cost (in-memory)
	sb.WriteString("\n# HELP llm_proxy_cost_usd_total Total cost in USD since service start\n")
	sb.WriteString("# TYPE llm_proxy_cost_usd_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cost_usd_total %.6f\n", metrics.CostTotal))

	// Cache (in-memory)
	sb.WriteString("\n# HELP llm_proxy_cache_hits_total Total cache hits since service start\n")
	sb.WriteString("# TYPE llm_proxy_cache_hits_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cache_hits_total %d\n", metrics.CacheHits))

	sb.WriteString("\n# HELP llm_proxy_cache_misses_total Total cache misses since service start\n")
	sb.WriteString("# TYPE llm_proxy_cache_misses_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cache_misses_total %d\n", metrics.CacheMisses))

	metrics.mutex.RUnlock()

	// ===== DATABASE METRICS (historical, survives restarts) =====
	dbMutex.Lock()
	defer dbMutex.Unlock()

	// --- Totals ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_total Total requests in database (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_total gauge\n")
	var totalRequests int64
	db.QueryRow("SELECT COUNT(*) FROM requests").Scan(&totalRequests)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_total %d\n", totalRequests))

	sb.WriteString("\n# HELP llm_proxy_db_cost_usd_total Total cost in database (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_usd_total gauge\n")
	var totalCost float64
	db.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests").Scan(&totalCost)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_usd_total %.6f\n", totalCost))

	sb.WriteString("\n# HELP llm_proxy_db_tokens_total Total tokens in database (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tokens_total gauge\n")
	var totalInputTokens, totalOutputTokens int64
	db.QueryRow("SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests").Scan(&totalInputTokens, &totalOutputTokens)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_total{direction=\"input\"} %d\n", totalInputTokens))
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_total{direction=\"output\"} %d\n", totalOutputTokens))

	// --- By Provider ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_provider Requests by provider (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_provider gauge\n")
	rows, _ := db.Query("SELECT provider, COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests GROUP BY provider")
	for rows.Next() {
		var provider string
		var count int64
		var cost float64
		var inputTokens, outputTokens int64
		rows.Scan(&provider, &count, &cost, &inputTokens, &outputTokens)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_provider{provider=\"%s\"} %d\n", provider, count))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_cost_by_provider Cost by provider (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_by_provider gauge\n")
	rows, _ = db.Query("SELECT provider, COALESCE(SUM(cost_usd), 0) FROM requests GROUP BY provider")
	for rows.Next() {
		var provider string
		var cost float64
		rows.Scan(&provider, &cost)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_by_provider{provider=\"%s\"} %.6f\n", provider, cost))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_tokens_by_provider Tokens by provider (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tokens_by_provider gauge\n")
	rows, _ = db.Query("SELECT provider, COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests GROUP BY provider")
	for rows.Next() {
		var provider string
		var inputTokens, outputTokens int64
		rows.Scan(&provider, &inputTokens, &outputTokens)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_provider{provider=\"%s\",direction=\"input\"} %d\n", provider, inputTokens))
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_provider{provider=\"%s\",direction=\"output\"} %d\n", provider, outputTokens))
	}
	rows.Close()

	// --- By Model ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_model Requests by model (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_model gauge\n")
	rows, _ = db.Query("SELECT model, COUNT(*) FROM requests GROUP BY model")
	for rows.Next() {
		var model string
		var count int64
		rows.Scan(&model, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_model{model=\"%s\"} %d\n", model, count))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_cost_by_model Cost by model (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_by_model gauge\n")
	rows, _ = db.Query("SELECT model, COALESCE(SUM(cost_usd), 0) FROM requests GROUP BY model")
	for rows.Next() {
		var model string
		var cost float64
		rows.Scan(&model, &cost)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_by_model{model=\"%s\"} %.6f\n", model, cost))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_tokens_by_model Tokens by model (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tokens_by_model gauge\n")
	rows, _ = db.Query("SELECT model, COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests GROUP BY model")
	for rows.Next() {
		var model string
		var inputTokens, outputTokens int64
		rows.Scan(&model, &inputTokens, &outputTokens)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_model{model=\"%s\",direction=\"input\"} %d\n", model, inputTokens))
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_model{model=\"%s\",direction=\"output\"} %d\n", model, outputTokens))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_avg_latency_by_model Average latency in ms by model\n")
	sb.WriteString("# TYPE llm_proxy_db_avg_latency_by_model gauge\n")
	rows, _ = db.Query("SELECT model, AVG(latency_ms) FROM requests WHERE latency_ms > 0 GROUP BY model")
	for rows.Next() {
		var model string
		var avgLatency float64
		rows.Scan(&model, &avgLatency)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_avg_latency_by_model{model=\"%s\"} %.2f\n", model, avgLatency))
	}
	rows.Close()

	// --- By Usecase ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_usecase Requests by usecase (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_usecase gauge\n")
	rows, _ = db.Query("SELECT COALESCE(usecase, 'unknown'), COUNT(*) FROM requests GROUP BY usecase")
	for rows.Next() {
		var usecase string
		var count int64
		rows.Scan(&usecase, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_usecase{usecase=\"%s\"} %d\n", usecase, count))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_cost_by_usecase Cost by usecase (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_by_usecase gauge\n")
	rows, _ = db.Query("SELECT COALESCE(usecase, 'unknown'), COALESCE(SUM(cost_usd), 0) FROM requests GROUP BY usecase")
	for rows.Next() {
		var usecase string
		var cost float64
		rows.Scan(&usecase, &cost)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_by_usecase{usecase=\"%s\"} %.6f\n", usecase, cost))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_tokens_by_usecase Tokens by usecase (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tokens_by_usecase gauge\n")
	rows, _ = db.Query("SELECT COALESCE(usecase, 'unknown'), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests GROUP BY usecase")
	for rows.Next() {
		var usecase string
		var inputTokens, outputTokens int64
		rows.Scan(&usecase, &inputTokens, &outputTokens)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_usecase{usecase=\"%s\",direction=\"input\"} %d\n", usecase, inputTokens))
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_by_usecase{usecase=\"%s\",direction=\"output\"} %d\n", usecase, outputTokens))
	}
	rows.Close()

	// --- By Request Type ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_type Requests by type (llm, tts, stt)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_type gauge\n")
	rows, _ = db.Query("SELECT COALESCE(request_type, 'llm'), COUNT(*) FROM requests GROUP BY request_type")
	for rows.Next() {
		var reqType string
		var count int64
		rows.Scan(&reqType, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_type{type=\"%s\"} %d\n", reqType, count))
	}
	rows.Close()

	// --- By Sensitivity ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_sensitivity Requests by sensitivity (local vs cloud)\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_sensitivity gauge\n")
	rows, _ = db.Query("SELECT sensitive, COUNT(*) FROM requests GROUP BY sensitive")
	for rows.Next() {
		var sensitive int
		var count int64
		rows.Scan(&sensitive, &count)
		sensitiveStr := "false"
		if sensitive == 1 {
			sensitiveStr = "true"
		}
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_sensitivity{sensitive=\"%s\"} %d\n", sensitiveStr, count))
	}
	rows.Close()

	// --- By Precision ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_precision Requests by precision level\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_precision gauge\n")
	rows, _ = db.Query("SELECT COALESCE(precision, 'unknown'), COUNT(*) FROM requests GROUP BY precision")
	for rows.Next() {
		var precision string
		var count int64
		rows.Scan(&precision, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_precision{precision=\"%s\"} %d\n", precision, count))
	}
	rows.Close()

	// --- Success/Error Rates ---
	sb.WriteString("\n# HELP llm_proxy_db_success_total Successful requests (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_success_total gauge\n")
	var successCount int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE success = 1").Scan(&successCount)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_success_total %d\n", successCount))

	sb.WriteString("\n# HELP llm_proxy_db_error_total Failed requests (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_error_total gauge\n")
	var errorCount int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE success = 0").Scan(&errorCount)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_error_total %d\n", errorCount))

	sb.WriteString("\n# HELP llm_proxy_db_errors_by_provider Failed requests by provider\n")
	sb.WriteString("# TYPE llm_proxy_db_errors_by_provider gauge\n")
	rows, _ = db.Query("SELECT provider, COUNT(*) FROM requests WHERE success = 0 GROUP BY provider")
	for rows.Next() {
		var provider string
		var count int64
		rows.Scan(&provider, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_errors_by_provider{provider=\"%s\"} %d\n", provider, count))
	}
	rows.Close()

	// --- Cache Stats ---
	sb.WriteString("\n# HELP llm_proxy_db_cached_total Cached requests (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_cached_total gauge\n")
	var cachedCount int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE cached = 1").Scan(&cachedCount)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_cached_total %d\n", cachedCount))

	// --- Vision Requests ---
	sb.WriteString("\n# HELP llm_proxy_db_vision_requests_total Vision requests with images (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_vision_requests_total gauge\n")
	var visionCount int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE has_images = 1").Scan(&visionCount)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_vision_requests_total %d\n", visionCount))

	// --- TTS Metrics ---
	sb.WriteString("\n# HELP llm_proxy_db_tts_audio_duration_ms_total Total TTS audio duration generated (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tts_audio_duration_ms_total gauge\n")
	var ttsDuration int64
	db.QueryRow("SELECT COALESCE(SUM(audio_duration_ms), 0) FROM requests WHERE request_type = 'tts'").Scan(&ttsDuration)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tts_audio_duration_ms_total %d\n", ttsDuration))

	sb.WriteString("\n# HELP llm_proxy_db_tts_input_chars_total Total TTS input characters (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_tts_input_chars_total gauge\n")
	var ttsChars int64
	db.QueryRow("SELECT COALESCE(SUM(input_chars), 0) FROM requests WHERE request_type = 'tts'").Scan(&ttsChars)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tts_input_chars_total %d\n", ttsChars))

	sb.WriteString("\n# HELP llm_proxy_db_tts_requests_by_voice TTS requests by voice\n")
	sb.WriteString("# TYPE llm_proxy_db_tts_requests_by_voice gauge\n")
	rows, _ = db.Query("SELECT COALESCE(voice, 'unknown'), COUNT(*) FROM requests WHERE request_type = 'tts' GROUP BY voice")
	for rows.Next() {
		var voice string
		var count int64
		rows.Scan(&voice, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_tts_requests_by_voice{voice=\"%s\"} %d\n", voice, count))
	}
	rows.Close()

	// --- Time-based metrics (last 24h, 7d, 30d) ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_24h Requests in last 24 hours\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_24h gauge\n")
	var requests24h int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE timestamp > datetime('now', '-1 day')").Scan(&requests24h)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_24h %d\n", requests24h))

	sb.WriteString("\n# HELP llm_proxy_db_cost_24h Cost in last 24 hours\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_24h gauge\n")
	var cost24h float64
	db.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE timestamp > datetime('now', '-1 day')").Scan(&cost24h)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_24h %.6f\n", cost24h))

	sb.WriteString("\n# HELP llm_proxy_db_tokens_24h Tokens in last 24 hours\n")
	sb.WriteString("# TYPE llm_proxy_db_tokens_24h gauge\n")
	var inputTokens24h, outputTokens24h int64
	db.QueryRow("SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM requests WHERE timestamp > datetime('now', '-1 day')").Scan(&inputTokens24h, &outputTokens24h)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_24h{direction=\"input\"} %d\n", inputTokens24h))
	sb.WriteString(fmt.Sprintf("llm_proxy_db_tokens_24h{direction=\"output\"} %d\n", outputTokens24h))

	sb.WriteString("\n# HELP llm_proxy_db_requests_7d Requests in last 7 days\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_7d gauge\n")
	var requests7d int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE timestamp > datetime('now', '-7 day')").Scan(&requests7d)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_7d %d\n", requests7d))

	sb.WriteString("\n# HELP llm_proxy_db_cost_7d Cost in last 7 days\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_7d gauge\n")
	var cost7d float64
	db.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE timestamp > datetime('now', '-7 day')").Scan(&cost7d)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_7d %.6f\n", cost7d))

	sb.WriteString("\n# HELP llm_proxy_db_requests_30d Requests in last 30 days\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_30d gauge\n")
	var requests30d int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE timestamp > datetime('now', '-30 day')").Scan(&requests30d)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_30d %d\n", requests30d))

	sb.WriteString("\n# HELP llm_proxy_db_cost_30d Cost in last 30 days\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_30d gauge\n")
	var cost30d float64
	db.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE timestamp > datetime('now', '-30 day')").Scan(&cost30d)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_30d %.6f\n", cost30d))

	// --- Requests by client IP ---
	sb.WriteString("\n# HELP llm_proxy_db_requests_by_client Requests by client IP/hostname\n")
	sb.WriteString("# TYPE llm_proxy_db_requests_by_client gauge\n")
	rows, _ = db.Query("SELECT COALESCE(client_ip, 'unknown'), COUNT(*) FROM requests GROUP BY client_ip")
	for rows.Next() {
		var clientIP string
		var count int64
		rows.Scan(&clientIP, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_requests_by_client{client=\"%s\"} %d\n", clientIP, count))
	}
	rows.Close()

	sb.WriteString("\n# HELP llm_proxy_db_cost_by_client Cost by client IP/hostname\n")
	sb.WriteString("# TYPE llm_proxy_db_cost_by_client gauge\n")
	rows, _ = db.Query("SELECT COALESCE(client_ip, 'unknown'), COALESCE(SUM(cost_usd), 0) FROM requests GROUP BY client_ip")
	for rows.Next() {
		var clientIP string
		var cost float64
		rows.Scan(&clientIP, &cost)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_cost_by_client{client=\"%s\"} %.6f\n", clientIP, cost))
	}
	rows.Close()

	// --- Routing analytics ---
	sb.WriteString("\n# HELP llm_proxy_db_routing_requests Requests by requested vs actual model\n")
	sb.WriteString("# TYPE llm_proxy_db_routing_requests gauge\n")
	rows, _ = db.Query("SELECT COALESCE(requested_model, 'unknown'), model, COUNT(*) FROM requests GROUP BY requested_model, model")
	for rows.Next() {
		var requestedModel, actualModel string
		var count int64
		rows.Scan(&requestedModel, &actualModel, &count)
		sb.WriteString(fmt.Sprintf("llm_proxy_db_routing_requests{requested=\"%s\",actual=\"%s\"} %d\n", requestedModel, actualModel, count))
	}
	rows.Close()

	// --- Replay requests ---
	sb.WriteString("\n# HELP llm_proxy_db_replay_requests_total Replayed requests (all time)\n")
	sb.WriteString("# TYPE llm_proxy_db_replay_requests_total gauge\n")
	var replayCount int64
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE is_replay = 1").Scan(&replayCount)
	sb.WriteString(fmt.Sprintf("llm_proxy_db_replay_requests_total %d\n", replayCount))

	// --- Pending requests ---
	pendingMutex.RLock()
	pendingCount := len(pendingRequests)
	pendingMutex.RUnlock()
	sb.WriteString("\n# HELP llm_proxy_pending_requests Current pending/in-flight requests\n")
	sb.WriteString("# TYPE llm_proxy_pending_requests gauge\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_pending_requests %d\n", pendingCount))

	// --- STT Concurrency Metrics ---
	if sttConcurrencyMgr != nil {
		metrics := sttConcurrencyMgr.GetMetrics()
		stats := sttConcurrencyMgr.GetStats()

		sb.WriteString("\n# HELP llm_proxy_stt_cpu_percent Current CPU usage percentage\n")
		sb.WriteString("# TYPE llm_proxy_stt_cpu_percent gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_cpu_percent %.2f\n", metrics.CPUPercent))

		sb.WriteString("\n# HELP llm_proxy_stt_mem_percent Current memory usage percentage\n")
		sb.WriteString("# TYPE llm_proxy_stt_mem_percent gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_mem_percent %.2f\n", metrics.MemPercent))

		sb.WriteString("\n# HELP llm_proxy_stt_mem_free_gb Free memory in gigabytes\n")
		sb.WriteString("# TYPE llm_proxy_stt_mem_free_gb gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_mem_free_gb %.2f\n", metrics.MemFreeGB))

		sb.WriteString("\n# HELP llm_proxy_stt_allowed_concurrent Dynamically allowed concurrent STT requests\n")
		sb.WriteString("# TYPE llm_proxy_stt_allowed_concurrent gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_allowed_concurrent %d\n", metrics.AllowedConcurrent))

		sb.WriteString("\n# HELP llm_proxy_stt_current_concurrent Current concurrent STT requests\n")
		sb.WriteString("# TYPE llm_proxy_stt_current_concurrent gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_current_concurrent %d\n", stats["current_concurrent"]))

		sb.WriteString("\n# HELP llm_proxy_stt_queue_pending Pending STT requests in queue\n")
		sb.WriteString("# TYPE llm_proxy_stt_queue_pending gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_queue_pending %d\n", stats["queue_pending"]))

		sb.WriteString("\n# HELP llm_proxy_stt_requests_total Total STT requests since service start\n")
		sb.WriteString("# TYPE llm_proxy_stt_requests_total counter\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_requests_total %d\n", stats["total_requests"]))

		sb.WriteString("\n# HELP llm_proxy_stt_rejections_total Rejected STT requests (queue full)\n")
		sb.WriteString("# TYPE llm_proxy_stt_rejections_total counter\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_rejections_total %d\n", stats["total_rejections"]))

		sb.WriteString("\n# HELP llm_proxy_stt_rejection_rate Current rejection rate (0.0-1.0)\n")
		sb.WriteString("# TYPE llm_proxy_stt_rejection_rate gauge\n")
		sb.WriteString(fmt.Sprintf("llm_proxy_stt_rejection_rate %.4f\n", stats["rejection_rate"]))
	}

	// Service info
	sb.WriteString("\n# HELP llm_proxy_info Service information\n")
	sb.WriteString("# TYPE llm_proxy_info gauge\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_info{version=\"1.0\",ollama_host=\"%s\"} 1\n", ollamaHost))

	w.Write([]byte(sb.String()))
}

// isLocalProvider returns true if the provider runs locally (not cloud)
func isLocalProvider(provider string) bool {
	switch provider {
	case "ollama", "llamacpp", "llamacpp-vision", "local-vision":
		return true
	default:
		return false
	}
}

func validateSensitiveRouteConfig(config *RouteConfig) error {
	if config == nil {
		return nil
	}
	if !isLocalProvider(config.Provider) {
		return fmt.Errorf("%s/%s must use a local provider", config.Provider, config.Model)
	}
	if config.Provider == "ollama" && providers.IsOllamaCloudModel(config.Model) {
		return fmt.Errorf("%s/%s is a cloud model and must use ollama-cloud", config.Provider, config.Model)
	}
	return nil
}

// validateRoutingTableSecurity ensures sensitive data never goes to cloud providers
// This is a critical security check that runs on startup
func validateRoutingTableSecurity() {
	violations := []string{}

	// Check text routing table
	for sensitive, precisions := range routingTable {
		if sensitive == "true" {
			for precision, config := range precisions {
				if err := validateSensitiveRouteConfig(config); err != nil {
					violations = append(violations,
						fmt.Sprintf("TEXT sensitive=true precision=%s routes to %v", precision, err))
				}
			}
		}
	}

	// Check vision routing table
	for sensitive, precisions := range visionRoutingTable {
		if sensitive == "true" {
			for precision, config := range precisions {
				if err := validateSensitiveRouteConfig(config); err != nil {
					violations = append(violations,
						fmt.Sprintf("VISION sensitive=true precision=%s routes to %v", precision, err))
				}
			}
		}
	}

	if router != nil {
		for usecase, routeTypes := range router.GetAllUsecaseRoutes() {
			for routeType, sensitivities := range routeTypes {
				for sensitive, precisions := range sensitivities {
					if sensitive != "true" {
						continue
					}
					for precision, config := range precisions {
						if err := validateSensitiveRouteConfig(config); err != nil {
							violations = append(violations,
								fmt.Sprintf("OVERRIDE usecase=%s type=%s precision=%s routes to %v",
									usecase, routeType, precision, err))
						}
					}
				}
			}
		}
	}

	if len(violations) > 0 {
		log.Println("SECURITY VIOLATION: Sensitive data would be sent to cloud providers!")
		for _, v := range violations {
			log.Printf("  - %s", v)
		}
		log.Fatal("Refusing to start due to routing configuration security violations")
	}

	log.Println("Security check passed: all sensitive routes use local providers")
}

// handleAIDAProxy proxies requests to Google's AIDA API (used by Jules CLI)
// This allows centralizing API tokens on the server and logging all requests
func handleAIDAProxy(w http.ResponseWriter, r *http.Request) {
	if getLockdownMode() {
		http.Error(w, "lockdown mode: cloud provider (AIDA) blocked", http.StatusForbidden)
		return
	}
	if aidaToken == "" {
		http.Error(w, "AIDA_TOKEN not configured", http.StatusServiceUnavailable)
		return
	}

	startTime := time.Now()
	clientIP := getClientIP(r)

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Forward to Google AIDA API
	targetURL := "https://aida.googleapis.com" + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	log.Printf("[AIDA] Proxying %s %s from %s", r.Method, r.URL.Path, clientIP)

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}

	// Copy headers but replace authorization
	for key, values := range r.Header {
		if key == "Authorization" {
			continue // We'll set our own
		}
		if key == "Host" {
			continue // Let Go set the correct host
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Set our stored AIDA token
	proxyReq.Header.Set("Authorization", "Bearer "+aidaToken)

	client := &http.Client{Timeout: time.Duration(aidaTimeout) * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("[AIDA] Proxy error: %v", err)
		http.Error(w, "AIDA API request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Copy status code and body
	w.WriteHeader(resp.StatusCode)
	respBody, _ := io.ReadAll(resp.Body)
	w.Write(respBody)

	latencyMs := time.Since(startTime).Milliseconds()
	log.Printf("[AIDA] %s %s -> %d (%dms)", r.Method, r.URL.Path, resp.StatusCode, latencyMs)

	// Log request (simplified logging for AIDA)
	status := "success"
	if resp.StatusCode >= 400 {
		status = "error"
	}
	dbMutex.Lock()
	db.Exec(`INSERT INTO requests (timestamp, request_type, provider, model, latency_ms, success, client_ip, request_body, response_body)
		VALUES (?, 'aida', 'google', 'jules-aida', ?, ?, ?, ?, ?)`,
		startTime.Format(time.RFC3339), latencyMs, status == "success", clientIP, string(body), string(respBody))
	dbMutex.Unlock()
}

// handleGeminiProxy provides a direct proxy to Google's Gemini API
// This allows using Gemini with the stored API key
func handleGeminiProxy(w http.ResponseWriter, r *http.Request) {
	if getLockdownMode() {
		http.Error(w, "lockdown mode: cloud provider (Gemini) blocked", http.StatusForbidden)
		return
	}
	if geminiKey == "" {
		http.Error(w, "GEMINI_API_KEY not configured", http.StatusServiceUnavailable)
		return
	}

	startTime := time.Now()
	clientIP := getClientIP(r)

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Strip the /gemini prefix and forward to Google
	path := strings.TrimPrefix(r.URL.Path, "/gemini")
	targetURL := "https://generativelanguage.googleapis.com" + path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	log.Printf("[Gemini] Proxying %s %s from %s", r.Method, path, clientIP)

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		if key == "Authorization" {
			continue
		}
		if key == "Host" {
			continue
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Set our stored Gemini API key
	proxyReq.Header.Set("Authorization", "Bearer "+geminiKey)

	client := &http.Client{Timeout: time.Duration(geminiTimeout) * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("[Gemini] Proxy error: %v", err)
		http.Error(w, "Gemini API request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Copy status code and body
	w.WriteHeader(resp.StatusCode)
	respBody, _ := io.ReadAll(resp.Body)
	w.Write(respBody)

	latencyMs := time.Since(startTime).Milliseconds()
	log.Printf("[Gemini] %s %s -> %d (%dms)", r.Method, path, resp.StatusCode, latencyMs)
}

// handleAIDALog receives AIDA request logs from mitmproxy addon
// This allows logging Jules requests that are intercepted via HTTPS proxy
func handleAIDALog(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	var logEntry struct {
		Timestamp      float64           `json:"timestamp"`
		Method         string            `json:"method"`
		URL            string            `json:"url"`
		RequestHeaders map[string]string `json:"request_headers"`
		RequestBody    string            `json:"request_body"`
		ResponseStatus int               `json:"response_status"`
		ResponseBody   string            `json:"response_body"`
	}

	if err := json.Unmarshal(body, &logEntry); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	clientIP := getClientIP(r)
	log.Printf("[AIDA-Log] %s %s -> %d from %s", logEntry.Method, logEntry.URL, logEntry.ResponseStatus, clientIP)

	// Store in database
	success := logEntry.ResponseStatus >= 200 && logEntry.ResponseStatus < 400
	dbMutex.Lock()
	db.Exec(`INSERT INTO requests (timestamp, request_type, provider, model, success, client_ip, request_body, response_body)
		VALUES (datetime('now'), 'aida', 'google', 'jules-aida', ?, ?, ?, ?)`,
		success, clientIP, logEntry.RequestBody, logEntry.ResponseBody)
	dbMutex.Unlock()

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"logged"}`))
}

func main() {
	// Create data directory
	os.MkdirAll(dataDir, 0755)

	// Load API keys from files if not in env
	if openaiKey == "" {
		if data, err := os.ReadFile("openai_key.txt"); err == nil {
			openaiKey = strings.TrimSpace(string(data))
		}
	}
	if anthropicKey == "" {
		if data, err := os.ReadFile("anthropic_key.txt"); err == nil {
			anthropicKey = strings.TrimSpace(string(data))
		}
	}
	if geminiKey == "" {
		if data, err := os.ReadFile("gemini_key.txt"); err == nil {
			geminiKey = strings.TrimSpace(string(data))
		}
	}
	if aidaToken == "" {
		if data, err := os.ReadFile("aida_token.txt"); err == nil {
			aidaToken = strings.TrimSpace(string(data))
		}
	}

	// Initialize database
	if err := initDB(); err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}

	// Initialize request logger (uses the database)
	sqliteLogger := repository.NewSQLiteLogger(db)

	// If PostgreSQL is configured, use multi-logger to write to both
	if postgresConnStr != "" {
		pgLogger, err := repository.NewPostgresLogger(postgresConnStr)
		if err != nil {
			log.Printf("Warning: Failed to connect to PostgreSQL, using SQLite only: %v", err)
			requestLogger = sqliteLogger
		} else {
			log.Printf("PostgreSQL logging enabled")
			requestLogger = repository.NewMultiLogger(sqliteLogger, pgLogger)
		}
	} else {
		requestLogger = sqliteLogger
	}

	// Initialize budget repository and checker
	budgetRepository = repository.NewSQLiteBudgetRepository(db)
	budgetChecker = budget.NewBudgetChecker(budgetRepository)

	// Initialize router with routing tables
	router = app.NewRouter(routingTable, visionRoutingTable)

	// Load usecase route overrides from database
	loadRouteOverrides()

	// Load disabled models from database
	loadDisabledModels()

	// Load local vision backend preference
	loadLocalVisionBackend()

	// Load lockdown mode preference
	loadLockdownMode()

	// Load assistant model mappings
	loadAssistantModels()

	// Set assistant model resolver on router
	router.SetAssistantResolver(getAssistantModel)

	// Wire lockdown checker so the router can reject cloud routes when enabled
	router.SetLockdownChecker(getLockdownMode)

	// Initialize cache
	initCache()

	// SECURITY ASSERTION: Verify sensitive routes only use Ollama (local)
	// This runs on startup to catch configuration errors before serving requests
	validateRoutingTableSecurity()

	// Initialize provider adapters (ports -> implementations)
	initChatProviders()

	// Enable backend switching for vision routes if llama.cpp is configured
	enableVisionBackendSwitch()

	// Initialize chat handler (primary adapter)
	chatHandler = &httphandlers.ChatHandler{
		Router:                 router,
		Providers:              chatProviders,
		Cache:                  requestCache,
		Logger:                 requestLogger,
		Metrics:                metrics,
		GenerateKey:            generateCacheKey,
		CalculateCost:          calculateCost,
		AddPending:             addPendingRequest,
		RemovePending:          removePendingRequest,
		IsModelDisabled:        isModelDisabled,
		GetProviderOverride:    getProviderOverride,
		CheckBudget:            budgetChecker.CheckBudget,
		OpenAIStreamingTimeout: openaiStreamingTimeout,
		OllamaHost:             ollamaHost,
	}

	// Initialize Responses API handler (OpenAI's newer API with smart routing)
	// Mode can be: auto (smart routing), openai (always forward), translate (always convert to chat)
	responsesMode := domain.ResponsesAPIModeAuto
	if modeEnv := os.Getenv("RESPONSES_API_MODE"); modeEnv != "" {
		switch modeEnv {
		case "openai":
			responsesMode = domain.ResponsesAPIModeOpenAI
		case "translate":
			responsesMode = domain.ResponsesAPIModeTranslate
		}
	}
	responsesHandler = &httphandlers.ResponsesHandler{
		ChatHandler:   chatHandler,
		OpenAIKey:     openaiKey,
		Mode:          responsesMode,
		Logger:        requestLogger,
		Metrics:       metrics,
		CalculateCost: calculateCost,
	}
	log.Printf("Responses API mode: %s", responsesMode)

	anthropicMessagesHandler = &httphandlers.AnthropicMessagesHandler{
		ChatHandler:      chatHandler,
		AnthropicKey:     anthropicKey,
		AnthropicTimeout: anthropicTimeout,
		OllamaTimeout:    chatTimeout,
		OllamaBaseURL:    "http://" + ollamaHost,
	}

	// Initialize STT concurrency manager (max 5 concurrent, queue up to 200)
	sttConcurrencyMgr = loadmanager.NewConcurrencyManager(5, 200)
	log.Printf("STT concurrency manager initialized: max 5 concurrent, queue 200")

	// Initialize STT handler (Whisper transcription)
	sttHandler = httphandlers.NewSTTHandler(whisperServerURL, openaiKey, requestLogger, speechTimeout, speechStreamingTimeout)
	sttHandler.ConcurrencyMgr = sttConcurrencyMgr
	sttHandler.AddPending = addPendingSTTRequest
	sttHandler.RemovePending = removePendingRequest

	// Initialize TTS audio cache
	cacheDir := ttsCacheDir
	if cacheDir == "" {
		cacheDir = filepath.Join(dataDir, "tts-cache")
	}
	var err error
	ttsAudioCache, err = audiocache.NewFileAudioCache(
		cacheDir,
		int64(ttsCacheMaxSize)*1024*1024, // Convert MB to bytes
		time.Duration(ttsCacheTTLDays)*24*time.Hour,
	)
	if err != nil {
		log.Printf("WARNING: Failed to initialize TTS audio cache: %v", err)
	}

	// Initialize TTS handler (Kokoro TTS + OpenAI TTS)
	ttsHandler = httphandlers.NewTTSHandler(ttsServerURL, openaiKey, requestLogger, ttsTimeout, ttsKokoroTimeout)
	ttsHandler.CheckBudget = budgetChecker.CheckBudget
	if ttsAudioCache != nil {
		ttsHandler.AudioCache = ttsAudioCache
		ttsHandler.CacheTTL = time.Duration(ttsCacheTTLDays) * 24 * time.Hour
	}

	// Initialize Web Search handler
	webSearchHandler = httphandlers.NewWebSearchHandler(anthropicKey, openaiKey, requestLogger, webSearchTimeout)

	// Initialize Image Generation handler
	imageGenHandler = httphandlers.NewImageGenHandler(openaiKey, requestLogger, router, imageGenTimeout)
	imageGenHandler.CheckBudget = budgetChecker.CheckBudget

	// Initialize History handler
	historyHandler = httphandlers.NewHistoryHandler(db, &dbMutex, requestCache)
	historyHandler.GetPending = getPendingRequests
	historyHandler.CancelPending = cancelPendingRequest
	historyHandler.BudgetRepo = budgetRepository

	// Initialize UI handler
	uiHandler = httphandlers.NewUIHandler()

	// Initialize budget handler
	budgetHandler := httphandlers.NewBudgetHandler(budgetRepository)

	// Budget API wrapper functions
	handleBudgetProviders := func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			budgetHandler.HandleGetProviderBudgets(w, r)
		} else if r.Method == "POST" {
			budgetHandler.HandleSetProviderBudget(w, r)
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}

	handleBudgetProviderDelete := func(w http.ResponseWriter, r *http.Request) {
		budgetHandler.HandleDeleteProviderBudget(w, r)
	}

	handleBudgetGlobal := func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			budgetHandler.HandleGetGlobalBudget(w, r)
		} else if r.Method == "POST" {
			budgetHandler.HandleSetGlobalBudget(w, r)
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}

	// Routes
	http.HandleFunc("/", uiHandler.HandleDashboard)
	http.HandleFunc("/request/", uiHandler.HandleRequestPage)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/metrics", handleMetrics)
	http.Handle("/v1/chat/completions", chatHandler)
	http.Handle("/v1/messages", anthropicMessagesHandler)
	http.HandleFunc("/v1/estimate", handleEstimate)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/api/stats", withCORS(historyHandler.HandleStats))
	http.HandleFunc("/api/stats/models", withCORS(historyHandler.HandleModelStats))
	http.HandleFunc("/api/timing", withCORS(historyHandler.HandleTiming))
	http.HandleFunc("/api/routes", withCORS(handleRoutes))
	http.HandleFunc("/api/routes/override", withCORS(handleRouteOverride))
	http.HandleFunc("/api/usecases", withCORS(handleUsecases))
	http.HandleFunc("/api/usecases/history", withCORS(historyHandler.HandleUsecasesHistory))
	http.HandleFunc("/api/usecases/distribution", withCORS(historyHandler.HandleUsecaseDistribution))
	http.HandleFunc("/api/models", withCORS(handleAvailableModels))
	http.HandleFunc("/api/models/config", withCORS(handleModelsConfig))
	http.HandleFunc("/api/models/vision", withCORS(handleVisionModels))
	http.HandleFunc("/api/history", withCORS(historyHandler.HandleRequestHistory))
	http.HandleFunc("/api/request", withCORS(historyHandler.HandleRequestDetail))
	http.HandleFunc("/api/replay", withCORS(handleReplayRequest))
	http.HandleFunc("/api/cache/clear", withCORS(historyHandler.HandleClearCache))
	http.HandleFunc("/api/pending", withCORS(historyHandler.HandlePendingRequests))
	http.HandleFunc("/api/pending/", withCORS(historyHandler.HandleCancelPendingRequest))
	http.HandleFunc("/api/tts-history", withCORS(historyHandler.HandleTTSHistory))
	http.HandleFunc("/api/tts-cache-stats", withCORS(handleTTSCacheStats))
	http.HandleFunc("/api/tts-cache/clear", withCORS(handleTTSCacheClear))
	http.HandleFunc("/api/stt-history", withCORS(historyHandler.HandleSTTHistory))
	http.HandleFunc("/api/analytics", withCORS(historyHandler.HandleAnalytics))
	http.HandleFunc("/api/system-metrics", withCORS(handleSystemMetrics))
	http.HandleFunc("/api/system/truncate-logs", withCORS(handleTruncateLogs))
	http.HandleFunc("/api/system/db-stats", withCORS(handleDbStats))
	http.HandleFunc("/api/backend", withCORS(handleBackend))
	http.HandleFunc("/api/assistant-models", withCORS(handleAssistantModels))
	http.HandleFunc("/api/settings/server-config", withCORS(handleServerConfig))
	http.HandleFunc("/api/settings/lockdown-mode", withCORS(handleLockdownMode))
	http.HandleFunc("/api/llamacpp/info", withCORS(handleLlamaCppInfo))
	http.HandleFunc("/api/llamacpp/metrics", withCORS(handleLlamaCppMetrics))
	http.HandleFunc("/analytics", uiHandler.HandleAnalyticsPage)
	http.HandleFunc("/stats", uiHandler.HandleStatsPage)
	http.HandleFunc("/budgets", uiHandler.HandleBudgetsPage)
	http.HandleFunc("/test", uiHandler.HandleTestPlayground)

	// Budget API routes
	http.HandleFunc("/api/budgets/providers", withCORS(handleBudgetProviders))
	http.HandleFunc("/api/budgets/providers/", withCORS(handleBudgetProviderDelete))
	http.HandleFunc("/api/budgets/global", withCORS(handleBudgetGlobal))
	http.HandleFunc("/api/budgets/spending", withCORS(budgetHandler.HandleGetBudgetSpending))
	http.HandleFunc("/v1/audio/transcriptions", sttHandler.HandleTranscription)
	http.HandleFunc("/v1/audio/transcriptions/stream", sttHandler.HandleStream)
	http.HandleFunc("/v1/audio/speech", ttsHandler.HandleTTS)
	http.HandleFunc("/tts", ttsHandler.HandleTTSCompat) // Compat endpoint for voice_cloning format (tts.lan)
	http.HandleFunc("/v1/websearch", webSearchHandler.HandleWebSearch)
	http.HandleFunc("/v1/images/generations", imageGenHandler.HandleImageGeneration)
	http.Handle("/v1/responses", responsesHandler) // Smart Responses API with auto/openai/translate modes

	// AIDA proxy for Jules CLI (Google's AI Development Assistant API)
	http.HandleFunc("/v1/aida/", handleAIDAProxy)
	// Gemini proxy for direct Gemini API access
	http.HandleFunc("/gemini/", handleGeminiProxy)

	log.Printf("LLM Proxy starting on port %s", port)
	log.Printf("Whisper server: %s", whisperServerURL)
	log.Printf("TTS server: %s", ttsServerURL)
	log.Printf("OpenAI key: %v", openaiKey != "")
	log.Printf("Anthropic key: %v", anthropicKey != "")
	log.Printf("Gemini key: %v", geminiKey != "")
	log.Printf("AIDA token: %v", aidaToken != "")
	log.Printf("Ollama host: %s", ollamaHost)
	timeouts := map[string]int{
		"chat":             chatTimeout,
		"speech":           speechTimeout,
		"speech_streaming": speechStreamingTimeout,
		"aida":             aidaTimeout,
		"gemini":           geminiTimeout,
		"openai":           openaiTimeout,
		"openai_streaming": openaiStreamingTimeout,
		"anthropic":        anthropicTimeout,
		"image_gen":        imageGenTimeout,
		"tts":              ttsTimeout,
		"tts_kokoro":       ttsKokoroTimeout,
		"web_search":       webSearchTimeout,
		"llamacpp":         llamacppTimeout,
	}
	log.Printf("Timeouts: %v", timeouts)
	if llamacppHost != "" {
		log.Printf("llama.cpp text host: %s", llamacppHost)
	}
	if llamacppVisionHost != "" {
		log.Printf("llama.cpp vision host: %s", llamacppVisionHost)
	}

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
