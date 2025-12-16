// Package main runs the llm-proxy service: an HTTP router/proxy for LLM, STT,
// and TTS requests with routing, caching, history, analytics, and UI.
package main

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"llm-proxy/internal/adapters/cache"
	httphandlers "llm-proxy/internal/adapters/http"
	"llm-proxy/internal/adapters/providers"
	"llm-proxy/internal/adapters/repository"
	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
	"llm-proxy/web"

	_ "github.com/mattn/go-sqlite3"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/mem"
)

// Configuration
var (
	port             = getEnv("PORT", "8080")
	openaiKey        = getEnv("OPENAI_API_KEY", "")
	anthropicKey     = getEnv("ANTHROPIC_API_KEY", "")
	geminiKey        = getEnv("GEMINI_API_KEY", "")
	aidaToken        = getEnv("AIDA_TOKEN", "") // Google AIDA API token for Jules
	ollamaHost       = getEnv("OLLAMA_HOST", "localhost:11434")
	dataDir          = getEnv("DATA_DIR", "./data")
	cacheTTLHours    = 24 * 7                                                // 1 week cache
	whisperServerURL = getEnv("WHISPER_SERVER_URL", "http://localhost:8890") // Local whisper server
	ttsServerURL     = getEnv("TTS_SERVER_URL", "http://localhost:7788")     // Local TTS server (Kokoro)
)

// Model pricing per 1M tokens (input, output)
var modelPricing = map[string][2]float64{
	// OpenAI GPT-5 series
	"gpt-5":   {1.25, 10.00},
	"gpt-5.1": {1.25, 10.00},
	"gpt-5.2": {1.75, 14.00},
	// OpenAI GPT-4.1 series
	"gpt-4.1":      {2.00, 8.00},
	"gpt-4.1-mini": {0.40, 1.60},
	"gpt-4.1-nano": {0.10, 0.40},
	// OpenAI GPT-4o series
	"gpt-4o":      {2.50, 10.00},
	"gpt-4o-mini": {0.15, 0.60},
	// OpenAI GPT-4 legacy
	"gpt-4-turbo": {10.00, 30.00},
	"gpt-4":       {30.00, 60.00},
	// OpenAI GPT-3.5
	"gpt-3.5-turbo": {0.50, 1.50},
	// OpenAI reasoning models (o-series)
	"o1":      {15.00, 60.00},
	"o1-mini": {1.10, 4.40},
	"o3-mini": {1.10, 4.40},
	"o4-mini": {1.10, 4.40},
	// OpenAI Codex models (agentic coding)
	"gpt-5-codex":        {1.25, 10.00},
	"gpt-5-codex-mini":   {0.25, 2.00},
	"gpt-5.1-codex":      {1.25, 10.00},
	"gpt-5.1-codex-max":  {1.25, 10.00},
	"gpt-5.1-codex-mini": {0.25, 2.00},
	// Anthropic - Claude 4.5 models only
	"claude-opus-4-5-20251101":   {5.00, 25.00},
	"claude-sonnet-4-5-20250929": {3.00, 15.00},
	"claude-haiku-4-5-20251001":  {1.00, 5.00},
	// Ollama (free)
	"qwen3-vl:30b":         {0, 0},
	"qwen3-vl:235b":        {0, 0},
	"llama3.3:70b":         {0, 0},
	"llama3.1-large":       {0, 0},
	"llama4:scout":         {0, 0},
	"gemma3:latest":        {0, 0},
	"mistral:7b":           {0, 0},
	"devstral:24b":         {0, 0},
	"deepseek-r1:70b":      {0, 0},
	"qwen3-coder:30b":      {0, 0},
	"deepseek-coder:33b":   {0, 0},
	"phi4:14b":             {0, 0},
	"codestral:latest":     {0, 0},
	"granite3.1-moe:3b":    {0, 0},
	// Google Gemini models
	"gemini-3.0-pro":                 {2.50, 15.00},
	"gemini-3.0-flash":               {0.25, 1.00},
	"gemini-2.5-pro":                 {1.25, 10.00},
	"gemini-2.5-flash":               {0.15, 0.60},
	"gemini-2.5-flash-lite":          {0.02, 0.10},
	"gemini-2.0-flash":               {0.10, 0.40},
	"gemini-2.0-flash-lite":          {0.02, 0.08},
	"gemini-1.5-pro":                 {1.25, 5.00},
	"gemini-1.5-flash":               {0.075, 0.30},
	"gemini-1.5-flash-8b":            {0.0375, 0.15},
	"gemini-exp-1206":                {0, 0}, // Free experimental
	"gemini-2.0-flash-thinking-exp":  {0, 0}, // Free experimental
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
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-5-20250929"},
		"high":      {Provider: "openai", Model: "gpt-4o"},
		"medium":    {Provider: "openai", Model: "gpt-4o-mini"},
		"low":       {Provider: "ollama", Model: "mistral:7b"},
	},
	// sensitive: true (text only, local)
	"true": {
		"very_high": nil, // Not available - Claude requires cloud
		"high":      {Provider: "ollama", Model: "llama3.3:70b"},
		"medium":    {Provider: "ollama", Model: "gemma3:latest"},
		"low":       {Provider: "ollama", Model: "mistral:7b"},
	},
}

// Vision routing (requests with images)
// Precision levels: very_high, high, medium, low
var visionRoutingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (can use cloud)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-5-20250929"}, // Claude has great vision
		"high":      {Provider: "openai", Model: "gpt-4o-mini"},                   // Fast and cheap
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

// Database
var db *sql.DB
var dbMutex sync.Mutex

// Request logger - using the port interface
var requestLogger ports.RequestLogger

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

// Pending requests tracker
var pendingRequests = make(map[string]*PendingRequest)
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

// OllamaTagsResponse is the response from Ollama /api/tags
type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// getOllamaModels fetches the list of available models from Ollama
func getOllamaModels() []string {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://" + ollamaHost + "/api/tags")
	if err != nil {
		log.Printf("Failed to fetch Ollama models: %v", err)
		return []string{"mistral:7b", "gemma3:latest"} // fallback
	}
	defer resp.Body.Close()

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		log.Printf("Failed to decode Ollama models: %v", err)
		return []string{"mistral:7b", "gemma3:latest"} // fallback
	}

	models := make([]string, 0, len(tagsResp.Models))
	for _, m := range tagsResp.Models {
		models = append(models, m.Name)
	}
	return models
}

// OllamaModelDetails is the detailed response from Ollama /api/show
type OllamaModelDetails struct {
	Details struct {
		Family   string   `json:"family"`
		Families []string `json:"families"`
	} `json:"details"`
}

// getOllamaVisionModels returns only vision-capable Ollama models
func getOllamaVisionModels() []string {
	client := &http.Client{Timeout: 10 * time.Second}

	// First get all models
	resp, err := client.Get("http://" + ollamaHost + "/api/tags")
	if err != nil {
		log.Printf("Failed to fetch Ollama models: %v", err)
		return []string{}
	}
	defer resp.Body.Close()

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		log.Printf("Failed to decode Ollama models: %v", err)
		return []string{}
	}

	visionModels := []string{}
	for _, m := range tagsResp.Models {
		// Check if model supports vision by querying details
		detailResp, err := client.Post("http://"+ollamaHost+"/api/show",
			"application/json",
			strings.NewReader(`{"name":"`+m.Name+`"}`))
		if err != nil {
			continue
		}

		var details OllamaModelDetails
		json.NewDecoder(detailResp.Body).Decode(&details)
		detailResp.Body.Close()

		// Check for vision capability indicators
		isVision := false
		// Check family names for vision indicators
		family := strings.ToLower(details.Details.Family)
		if strings.Contains(family, "vl") || strings.Contains(family, "llava") ||
			strings.Contains(family, "vision") || strings.Contains(family, "clip") {
			isVision = true
		}
		// Check families array for clip (vision encoder)
		for _, f := range details.Details.Families {
			if strings.ToLower(f) == "clip" || strings.Contains(strings.ToLower(f), "vl") {
				isVision = true
				break
			}
		}
		// Also check model name for common vision model patterns
		nameLower := strings.ToLower(m.Name)
		if strings.Contains(nameLower, "llava") || strings.Contains(nameLower, "-vl") ||
			strings.Contains(nameLower, "vision") {
			isVision = true
		}

		if isVision {
			visionModels = append(visionModels, m.Name)
		}
	}
	return visionModels
}

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
	visionModels := getOllamaVisionModels()

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
		// Try prefix match
		for m, p := range modelPricing {
			if strings.HasPrefix(model, m) {
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
func addPendingRequest(req *ChatCompletionRequest, route *RouteConfig, startTime time.Time) string {
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

	pendingRequests[id] = &PendingRequest{
		ID:        id,
		StartTime: startTime,
		Provider:  route.Provider,
		Model:     route.Model,
		HasImages: req.HasImages(),
		Sensitive: sensitive,
		Precision: req.Precision,
		Usecase:   req.Usecase,
		Preview:   preview,
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
		result = append(result, req)
	}
	return result
}

// resolveRoute delegates to the router service.
func resolveRoute(req *ChatCompletionRequest) (*RouteConfig, error) {
	return router.ResolveRoute(req)
}

// ChatProvider is an alias to the port interface.
type ChatProvider = ports.ChatProvider

// chatProviders maps provider names to their implementations.
var chatProviders map[string]ChatProvider

func initChatProviders() {
	chatProviders = map[string]ChatProvider{
		"openai":    providers.NewOpenAIProvider(openaiKey),
		"anthropic": providers.NewAnthropicProvider(anthropicKey),
		"ollama":    providers.NewOllamaProvider(ollamaHost),
		"gemini":    providers.NewGeminiProvider(geminiKey),
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

func isSensitiveRequest(r *http.Request) bool {
	// Check header first (X-Sensitive)
	if h := r.Header.Get("X-Sensitive"); h != "" {
		return h != "false"
	}
	// Check form value
	if s := r.FormValue("sensitive"); s != "" {
		return s != "false"
	}
	// Default to sensitive (local) for safety
	return true
}

func handleWhisperTranscription(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	// Parse multipart form (max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Get the audio file
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing or invalid file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read file content
	fileContent, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Get optional parameters
	model := r.FormValue("model")
	language := r.FormValue("language")
	prompt := r.FormValue("prompt") // Context hints for transcription (names, jargon, etc.)

	// Route based on sensitive flag
	sensitive := isSensitiveRequest(r)

	// Prepare log entry
	logEntry := &RequestLog{
		Timestamp:   startTime,
		RequestType: "stt",
		Sensitive:   sensitive,
		InputChars:  len(fileContent), // Store file size in InputChars field
		ClientIP:    getClientIP(r),
	}

	var resp *WhisperTranscriptionResponse
	var provider string

	if sensitive {
		// Use local whisper server
		resp, err = callLocalWhisper(fileContent, header.Filename, model, language, prompt)
		provider = "local"
		logEntry.Provider = "local"
		logEntry.Model = "whisper-large-v3" // Local server uses whisper-large-v3
	} else {
		// Use OpenAI Whisper API
		resp, err = callOpenAIWhisper(fileContent, header.Filename, model, language, prompt)
		provider = "openai"
		logEntry.Provider = "openai"
		logEntry.Model = "whisper-1"
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs

	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logRequest(logEntry)

		log.Printf("Whisper transcription failed (%s): %v", provider, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	logEntry.Success = true
	logEntry.OutputTokens = len(resp.Text) // Store output text length
	// Store transcribed text in ResponseBody for history display
	logEntry.ResponseBody = []byte(resp.Text)
	logRequest(logEntry)

	log.Printf("Whisper transcription complete (%s, %dms): %d chars", provider, latencyMs, len(resp.Text))

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", provider)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", latencyMs))
	json.NewEncoder(w).Encode(resp)
}

func callLocalWhisper(fileContent []byte, filename, model, language, prompt string) (*WhisperTranscriptionResponse, error) {
	// Create multipart form for local whisper server
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(fileContent); err != nil {
		return nil, fmt.Errorf("failed to write file content: %w", err)
	}

	// Add optional fields
	if model != "" {
		writer.WriteField("model", model)
	}
	if language != "" {
		writer.WriteField("language", language)
	}
	if prompt != "" {
		writer.WriteField("prompt", prompt)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Make request to local whisper server
	url := whisperServerURL + "/v1/audio/transcriptions"
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("local whisper request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("local whisper error %d: %s", resp.StatusCode, string(respBody))
	}

	var result WhisperTranscriptionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func callOpenAIWhisper(fileContent []byte, filename, model, language, prompt string) (*WhisperTranscriptionResponse, error) {
	if openaiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}

	// Create multipart form for OpenAI
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(fileContent); err != nil {
		return nil, fmt.Errorf("failed to write file content: %w", err)
	}

	// Add model (default to whisper-1)
	if model == "" {
		model = "whisper-1"
	}
	writer.WriteField("model", model)

	// Add optional language
	if language != "" {
		writer.WriteField("language", language)
	}

	// Add optional prompt for context hints
	if prompt != "" {
		writer.WriteField("prompt", prompt)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Make request to OpenAI
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/transcriptions", &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+openaiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenAI whisper request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI whisper error %d: %s", resp.StatusCode, string(respBody))
	}

	var result WhisperTranscriptionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func handleWhisperStream(w http.ResponseWriter, r *http.Request) {
	// Streaming transcription - ONLY local, refuse cloud requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	// Check sensitive flag - streaming MUST be local only
	if !isSensitiveRequest(r) {
		http.Error(w, "Streaming transcription only available locally (OpenAI API does not support streaming). Set sensitive=true or remove X-Sensitive header.", http.StatusBadRequest)
		return
	}

	// Parse multipart form (max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Get the audio file
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing or invalid file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read file content
	fileContent, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Get optional parameters
	model := r.FormValue("model")
	language := r.FormValue("language")

	// Prepare log entry
	logEntry := &RequestLog{
		Timestamp:   startTime,
		RequestType: "stt",
		Provider:    "local",
		Model:       "whisper-local-stream",
		Sensitive:   true, // Streaming is always local/sensitive
		InputChars:  len(fileContent),
		ClientIP:    getClientIP(r),
	}

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "Streaming not supported"
		logRequest(logEntry)
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Forward to local whisper server streaming endpoint
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to create form file"
		logRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to create form file\"}\n\n")
		flusher.Flush()
		return
	}
	if _, err := part.Write(fileContent); err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to write file content"
		logRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to write file content\"}\n\n")
		flusher.Flush()
		return
	}

	if model != "" {
		writer.WriteField("model", model)
	}
	if language != "" {
		writer.WriteField("language", language)
	}
	writer.Close()

	// Make streaming request to local whisper server
	whisperURL := whisperServerURL + "/v1/audio/transcriptions/stream"
	req, err := http.NewRequest("POST", whisperURL, &buf)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to create request"
		logRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to create request\"}\n\n")
		flusher.Flush()
		return
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 300 * time.Second} // Longer timeout for streaming
	resp, err := client.Do(req)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("local whisper stream request failed: %s", err.Error())
		logRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"local whisper stream request failed: %s\"}\n\n", err.Error())
		flusher.Flush()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("local whisper error %d: %s", resp.StatusCode, string(respBody))
		logRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"local whisper error %d: %s\"}\n\n", resp.StatusCode, string(respBody))
		flusher.Flush()
		return
	}

	// Stream the response back to client
	buf2 := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf2)
		if n > 0 {
			w.Write(buf2[:n])
			flusher.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(w, "data: {\"error\": \"stream read error: %s\"}\n\n", err.Error())
			flusher.Flush()
			break
		}
	}

	// Log successful stream completion
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.Success = true
	logRequest(logEntry)
}

// TTS handler - proxy to local Kokoro TTS server

// Map OpenAI voice names to Kokoro voices
var ttsVoiceMap = map[string]string{
	"alloy":   "af_nicole", // American female
	"echo":    "am_adam",   // American male
	"fable":   "bf_emma",   // British female
	"onyx":    "bm_george", // British male
	"nova":    "af_sky",    // American female
	"shimmer": "af_bella",  // American female
}

func handleTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req TTSRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Input == "" {
		http.Error(w, "Missing required field: input", http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.Voice == "" {
		req.Voice = "af_nicole" // American female - same as appdaemon default
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "mp3"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}

	// Map OpenAI voice to Kokoro voice
	kokoroVoice, ok := ttsVoiceMap[req.Voice]
	if !ok {
		// If not in map, use directly (allows passing Kokoro voice names)
		kokoroVoice = req.Voice
	}

	// Build Kokoro TTS server URL with query params
	ttsURL := fmt.Sprintf("%s/tts?text=%s&voice=%s&format=%s&speed=%.2f",
		ttsServerURL,
		urlEncode(req.Input),
		kokoroVoice,
		req.ResponseFormat,
		req.Speed,
	)

	// Prepare log entry
	logEntry := &RequestLog{
		Timestamp:   startTime,
		RequestType: "tts",
		Provider:    "kokoro",
		Model:       "kokoro-tts",
		Voice:       kokoroVoice,
		InputChars:  len(req.Input),
		RequestBody: body,
		ClientIP:    getClientIP(r),
	}

	// Forward request to Kokoro TTS server
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(ttsURL)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		logRequest(logEntry)

		log.Printf("TTS request failed: %v", err)
		http.Error(w, "TTS server unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		logRequest(logEntry)

		log.Printf("TTS error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, fmt.Sprintf("TTS error: %s", string(respBody)), resp.StatusCode)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true
	logRequest(logEntry)

	log.Printf("TTS complete (%dms): voice=%s, len=%d chars", latencyMs, kokoroVoice, len(req.Input))

	// Stream audio response back to client
	contentType := "audio/mpeg"
	if req.ResponseFormat == "wav" {
		contentType = "audio/wav"
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=\"speech.%s\"", req.ResponseFormat))
	io.Copy(w, resp.Body)
}

// TTSCompatRequest is the voice_cloning format (different from OpenAI's TTSRequest)
type TTSCompatRequest struct {
	Text  string  `json:"text"`
	Voice string  `json:"voice"`
	Speed float64 `json:"speed"`
}

// handleTTSCompat handles the voice_cloning API format (/tts endpoint)
// This allows tts.lan to route through llm-proxy for logging while maintaining
// compatibility with existing clients that use the voice_cloning format.
func handleTTSCompat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req TTSCompatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Text == "" {
		http.Error(w, "Missing required field: text", http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.Voice == "" {
		req.Voice = "af_nicole"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}

	// Build Kokoro TTS server URL with query params (mp3 is default format)
	ttsURL := fmt.Sprintf("%s/tts?text=%s&voice=%s&format=mp3&speed=%.2f",
		ttsServerURL,
		urlEncode(req.Text),
		req.Voice,
		req.Speed,
	)

	// Prepare log entry
	logEntry := &RequestLog{
		Timestamp:   startTime,
		RequestType: "tts",
		Provider:    "kokoro",
		Model:       "kokoro-tts",
		Voice:       req.Voice,
		InputChars:  len(req.Text),
		RequestBody: body,
		ClientIP:    getClientIP(r),
	}

	// Forward request to Kokoro TTS server
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(ttsURL)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		logRequest(logEntry)

		log.Printf("TTS compat request failed: %v", err)
		http.Error(w, "TTS server unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		logRequest(logEntry)

		log.Printf("TTS compat error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, fmt.Sprintf("TTS error: %s", string(respBody)), resp.StatusCode)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true
	logRequest(logEntry)

	log.Printf("TTS compat complete (%dms): voice=%s, len=%d chars", latencyMs, req.Voice, len(req.Text))

	// Stream audio response back to client
	w.Header().Set("Content-Type", "audio/mpeg")
	io.Copy(w, resp.Body)
}

// urlEncode encodes a string for use in a URL query parameter
func urlEncode(s string) string {
	return url.QueryEscape(s)
}

// WebSearchRequest represents a web search request
type WebSearchRequest struct {
	Query          string   `json:"query"`
	Provider       string   `json:"provider"` // "anthropic" or "openai"
	Model          string   `json:"model"`
	MaxUses        int      `json:"max_uses"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	Usecase        string   `json:"usecase,omitempty"`
}

// WebSearchResponse represents a web search response
type WebSearchResponse struct {
	Response    string              `json:"response"`
	Model       string              `json:"model"`
	Provider    string              `json:"provider"`
	SearchCount int                 `json:"search_count"`
	Sources     []WebSearchSource   `json:"sources,omitempty"`
	CostUSD     float64             `json:"cost_usd,omitempty"`
	Error       string              `json:"error,omitempty"`
	// Internal fields for logging (not sent to client)
	InputTokens  int    `json:"-"`
	OutputTokens int    `json:"-"`
	RawRequest   []byte `json:"-"`
	RawResponse  []byte `json:"-"`
}

type WebSearchSource struct {
	URL     string `json:"url"`
	Title   string `json:"title"`
	Snippet string `json:"snippet,omitempty"`
}

func handleWebSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req WebSearchRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Missing required field: query", http.StatusBadRequest)
		return
	}

	if req.MaxUses == 0 {
		req.MaxUses = 5
	}

	// Prepare log entry
	provider := req.Provider
	if provider == "" {
		provider = "anthropic"
	}
	model := req.Model
	if model == "" {
		if provider == "openai" {
			model = "gpt-4o"
		} else {
			model = "claude-sonnet-4-5-20250929"
		}
	}

	logEntry := &RequestLog{
		Timestamp:      startTime,
		RequestType:    "llm",
		Provider:       provider,
		Model:          model,
		RequestedModel: model,
		Usecase:        req.Usecase,
		RequestBody:    body,
		ClientIP:       getClientIP(r),
	}

	w.Header().Set("Content-Type", "application/json")

	var resp WebSearchResponse
	if provider == "openai" {
		resp = doOpenAIWebSearch(req)
	} else {
		resp = doAnthropicWebSearch(req)
	}

	// Update log entry with response
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.Success = resp.Error == ""
	logEntry.Error = resp.Error
	logEntry.CostUSD = resp.CostUSD
	logEntry.InputTokens = resp.InputTokens
	logEntry.OutputTokens = resp.OutputTokens
	if resp.Model != "" {
		logEntry.Model = resp.Model
	}

	// Store raw API request/response for request detail view
	if resp.RawRequest != nil {
		logEntry.RequestBody = resp.RawRequest
	}
	if resp.RawResponse != nil {
		logEntry.ResponseBody = resp.RawResponse
	}

	logRequest(logEntry)

	log.Printf("WebSearch complete (%dms): provider=%s, model=%s, searches=%d, cost=$%.6f",
		logEntry.LatencyMs, provider, logEntry.Model, resp.SearchCount, resp.CostUSD)

	json.NewEncoder(w).Encode(resp)
}

func doAnthropicWebSearch(req WebSearchRequest) WebSearchResponse {
	if anthropicKey == "" {
		return WebSearchResponse{Error: "Anthropic API key not configured"}
	}

	model := req.Model
	if model == "" {
		model = "claude-sonnet-4-5-20250929"
	}

	// Build web search tool
	webSearchTool := map[string]interface{}{
		"type":     "web_search_20250305",
		"name":     "web_search",
		"max_uses": req.MaxUses,
	}
	if len(req.AllowedDomains) > 0 {
		webSearchTool["allowed_domains"] = req.AllowedDomains
	}

	// Build Anthropic request
	anthropicReq := map[string]interface{}{
		"model":      model,
		"max_tokens": 4096,
		"messages": []map[string]interface{}{
			{"role": "user", "content": req.Query},
		},
		"tools": []map[string]interface{}{webSearchTool},
	}

	reqBody, _ := json.Marshal(anthropicReq)

	httpReq, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return WebSearchResponse{Error: "Failed to create request: " + err.Error()}
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", anthropicKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{Timeout: 120 * time.Second}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return WebSearchResponse{Error: "Request failed: " + err.Error()}
	}
	defer httpResp.Body.Close()

	respBody, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode != 200 {
		return WebSearchResponse{Error: fmt.Sprintf("API error %d: %s", httpResp.StatusCode, string(respBody))}
	}

	var anthropicResp struct {
		Content []struct {
			Type      string `json:"type"`
			Text      string `json:"text"`
			Name      string `json:"name"`
			Input     struct {
				Query string `json:"query"`
			} `json:"input"`
			Content []struct {
				Type  string `json:"type"`
				URL   string `json:"url"`
				Title string `json:"title"`
			} `json:"content"`
			Citations []struct {
				URL       string `json:"url"`
				Title     string `json:"title"`
				CitedText string `json:"cited_text"`
			} `json:"citations"`
		} `json:"content"`
		Model string `json:"model"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			ServerToolUse struct {
				WebSearchRequests int `json:"web_search_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &anthropicResp); err != nil {
		return WebSearchResponse{Error: "Failed to parse response: " + err.Error()}
	}

	// Extract response text and sources
	var textParts []string
	var sources []WebSearchSource
	searchCount := 0

	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			textParts = append(textParts, block.Text)
			// Add citations as sources
			for _, cite := range block.Citations {
				sources = append(sources, WebSearchSource{
					URL:     cite.URL,
					Title:   cite.Title,
					Snippet: cite.CitedText,
				})
			}
		} else if block.Type == "web_search_tool_result" {
			searchCount++
			for _, result := range block.Content {
				if result.Type == "web_search_result" {
					sources = append(sources, WebSearchSource{
						URL:   result.URL,
						Title: result.Title,
					})
				}
			}
		}
	}

	if anthropicResp.Usage.ServerToolUse.WebSearchRequests > 0 {
		searchCount = anthropicResp.Usage.ServerToolUse.WebSearchRequests
	}

	// Calculate cost - pricing is [2]float64{inputPerMillion, outputPerMillion}
	pricing, ok := modelPricing[model]
	if !ok {
		pricing = modelPricing["claude-sonnet-4-5-20250929"]
	}
	tokenCost := (float64(anthropicResp.Usage.InputTokens)*pricing[0] +
		float64(anthropicResp.Usage.OutputTokens)*pricing[1]) / 1000000
	searchCost := float64(searchCount) * 0.01 // $10 per 1000 searches = $0.01 per search
	totalCost := tokenCost + searchCost

	return WebSearchResponse{
		Response:     strings.Join(textParts, "\n"),
		Model:        anthropicResp.Model,
		Provider:     "anthropic",
		SearchCount:  searchCount,
		Sources:      sources,
		CostUSD:      totalCost,
		InputTokens:  anthropicResp.Usage.InputTokens,
		OutputTokens: anthropicResp.Usage.OutputTokens,
		RawRequest:   reqBody,
		RawResponse:  respBody,
	}
}

func doOpenAIWebSearch(req WebSearchRequest) WebSearchResponse {
	if openaiKey == "" {
		return WebSearchResponse{Error: "OpenAI API key not configured"}
	}

	model := req.Model
	if model == "" {
		model = "gpt-4o"
	}

	// Build web search tool
	webSearchTool := map[string]interface{}{
		"type": "web_search",
	}
	if len(req.AllowedDomains) > 0 {
		webSearchTool["filters"] = map[string]interface{}{
			"allowed_domains": req.AllowedDomains,
		}
	}

	// Build OpenAI Responses API request
	openaiReq := map[string]interface{}{
		"model": model,
		"input": req.Query,
		"tools": []map[string]interface{}{webSearchTool},
	}

	reqBody, _ := json.Marshal(openaiReq)

	httpReq, err := http.NewRequest("POST", "https://api.openai.com/v1/responses", bytes.NewReader(reqBody))
	if err != nil {
		return WebSearchResponse{Error: "Failed to create request: " + err.Error()}
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+openaiKey)

	client := &http.Client{Timeout: 120 * time.Second}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return WebSearchResponse{Error: "Request failed: " + err.Error()}
	}
	defer httpResp.Body.Close()

	respBody, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode != 200 {
		return WebSearchResponse{Error: fmt.Sprintf("API error %d: %s", httpResp.StatusCode, string(respBody))}
	}

	var openaiResp struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Annotations []struct {
					Type  string `json:"type"`
					URL   string `json:"url"`
					Title string `json:"title"`
				} `json:"annotations"`
			} `json:"content"`
			Action struct {
				Type    string `json:"type"`
				Sources []struct {
					URL   string `json:"url"`
					Title string `json:"title"`
				} `json:"sources"`
			} `json:"action"`
		} `json:"output"`
		Model string `json:"model"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return WebSearchResponse{Error: "Failed to parse response: " + err.Error()}
	}

	// Extract response text and sources
	var textParts []string
	var sources []WebSearchSource
	searchCount := 0

	for _, output := range openaiResp.Output {
		if output.Type == "message" {
			for _, content := range output.Content {
				if content.Type == "output_text" {
					textParts = append(textParts, content.Text)
				}
				for _, ann := range content.Annotations {
					if ann.Type == "url_citation" {
						sources = append(sources, WebSearchSource{
							URL:   ann.URL,
							Title: ann.Title,
						})
					}
				}
			}
		} else if output.Type == "web_search_call" {
			searchCount++
			for _, src := range output.Action.Sources {
				sources = append(sources, WebSearchSource{
					URL:   src.URL,
					Title: src.Title,
				})
			}
		}
	}

	// Calculate cost (approximate - OpenAI charges per search)
	// pricing is [2]float64{inputPerMillion, outputPerMillion}
	pricing, ok := modelPricing[model]
	if !ok {
		pricing = modelPricing["gpt-4o"]
	}
	tokenCost := (float64(openaiResp.Usage.InputTokens)*pricing[0] +
		float64(openaiResp.Usage.OutputTokens)*pricing[1]) / 1000000
	searchCost := float64(searchCount) * 0.03 // $30 per 1000 searches = $0.03 per search for gpt-4o
	totalCost := tokenCost + searchCost

	return WebSearchResponse{
		Response:     strings.Join(textParts, "\n"),
		Model:        openaiResp.Model,
		Provider:     "openai",
		SearchCount:  searchCount,
		Sources:      sources,
		CostUSD:      totalCost,
		InputTokens:  openaiResp.Usage.InputTokens,
		OutputTokens: openaiResp.Usage.OutputTokens,
		RawRequest:   reqBody,
		RawResponse:  respBody,
	}
}

// handleChatCompletions is now handled by httphandlers.ChatHandler
// handleResponses is now handled by httphandlers.ResponsesHandler

func handleModels(w http.ResponseWriter, r *http.Request) {
	models := []map[string]interface{}{}
	seen := make(map[string]bool)

	// Add routing models
	models = append(models, map[string]interface{}{
		"id":       "auto",
		"object":   "model",
		"owned_by": "llm-proxy",
	})
	seen["auto"] = true

	// Add known models from pricing
	for model := range modelPricing {
		if !seen[model] {
			models = append(models, map[string]interface{}{
				"id":       model,
				"object":   "model",
				"owned_by": "llm-proxy",
			})
			seen[model] = true
		}
	}

	// Add all Ollama models dynamically
	for _, model := range getOllamaModels() {
		if !seen[model] {
			models = append(models, map[string]interface{}{
				"id":       model,
				"object":   "model",
				"owned_by": "llm-proxy",
			})
			seen[model] = true
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// ==================== Anthropic API Compatibility ====================
// These types and handlers implement the Anthropic /v1/messages API format
// to allow Claude Code to use llm-proxy with any backend (including Ollama).

// AnthropicMessagesRequest represents an Anthropic API request
type AnthropicMessagesRequest struct {
	Model       string                    `json:"model"`
	MaxTokens   int                       `json:"max_tokens"`
	System      interface{}               `json:"system,omitempty"` // string or []AnthropicSystemBlock
	Messages    []AnthropicMessage        `json:"messages"`
	Tools       []AnthropicTool           `json:"tools,omitempty"`
	ToolChoice  interface{}               `json:"tool_choice,omitempty"`
	Stream      bool                      `json:"stream,omitempty"`
	Temperature float64                   `json:"temperature,omitempty"`
	TopP        float64                   `json:"top_p,omitempty"`
	TopK        int                       `json:"top_k,omitempty"`
	StopSeqs    []string                  `json:"stop_sequences,omitempty"`
	Metadata    *AnthropicRequestMetadata `json:"metadata,omitempty"`
}

// AnthropicRequestMetadata contains request metadata
type AnthropicRequestMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

// AnthropicSystemBlock represents a system block with cache control
type AnthropicSystemBlock struct {
	Type         string                    `json:"type"`
	Text         string                    `json:"text,omitempty"`
	CacheControl *AnthropicCacheControl    `json:"cache_control,omitempty"`
}

// AnthropicCacheControl for prompt caching
type AnthropicCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// AnthropicMessage represents a message in Anthropic format
type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []AnthropicContentBlock
}

// AnthropicContentBlock represents a content block
type AnthropicContentBlock struct {
	Type         string                    `json:"type"`
	Text         string                    `json:"text,omitempty"`
	ID           string                    `json:"id,omitempty"`
	Name         string                    `json:"name,omitempty"`
	Input        map[string]interface{}    `json:"input,omitempty"`
	ToolUseID    string                    `json:"tool_use_id,omitempty"`
	Content      interface{}               `json:"content,omitempty"` // for tool_result
	Source       *AnthropicImageSource     `json:"source,omitempty"`
	CacheControl *AnthropicCacheControl    `json:"cache_control,omitempty"`
}

// AnthropicImageSource for image content
type AnthropicImageSource struct {
	Type      string `json:"type"`       // "base64" or "url"
	MediaType string `json:"media_type"` // e.g., "image/png"
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// AnthropicTool represents a tool definition (regular tools or server tools like web_search)
type AnthropicTool struct {
	// Regular tool fields
	Name        string                 `json:"name,omitempty"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema,omitempty"`
	// Server tool fields (e.g., web_search_20250305)
	Type           string   `json:"type,omitempty"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

// AnthropicMessagesResponse represents an Anthropic API response
type AnthropicMessagesResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        AnthropicUsage          `json:"usage"`
}

// AnthropicUsage contains token usage
type AnthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// AnthropicError represents an API error
type AnthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// AnthropicErrorResponse wraps an error
type AnthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error AnthropicError `json:"error"`
}

// handleAnthropicMessages handles /v1/messages requests in Anthropic format
func handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "invalid_request_error",
				Message: "Method not allowed",
			},
		})
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "invalid_request_error",
				Message: "Failed to read request body",
			},
		})
		return
	}

	var antReq AnthropicMessagesRequest
	if err := json.Unmarshal(body, &antReq); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "invalid_request_error",
				Message: "Invalid JSON: " + err.Error(),
			},
		})
		return
	}

	// Convert Anthropic request to OpenAI format for internal routing
	openaiReq := convertAnthropicToOpenAI(&antReq)

	// Allow model override via X-Model-Override header
	// This is useful when Claude Code always sends claude-* models but you want to use Ollama
	modelOverride := r.Header.Get("X-Model-Override")
	if modelOverride != "" {
		openaiReq.Model = modelOverride
		antReq.Model = modelOverride // Update for logging
	}

	// Use X-Usecase header - require it to be set
	usecase := r.Header.Get("X-Usecase")
	// if usecase == "" {
	// 	usecase = "claude-code"  // Commented out - require explicit usecase
	// }
	openaiReq.Usecase = usecase

	// Default to non-sensitive for claude-code (allows cloud routing)
	// Can be overridden via X-Sensitive header
	sensitive := false
	if r.Header.Get("X-Sensitive") == "true" {
		sensitive = true
	}
	openaiReq.Sensitive = &sensitive

	// Default precision
	precision := r.Header.Get("X-Precision")
	if precision == "" {
		precision = "medium"
	}
	openaiReq.Precision = precision

	startTime := time.Now()

	// Resolve route
	route, err := router.ResolveRoute(openaiReq)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "invalid_request_error",
				Message: "Route resolution failed: " + err.Error(),
			},
		})
		return
	}

	// Check if model is disabled
	if isModelDisabled(route.Model) {
		// Log the blocked request
		logEntry := &domain.RequestLog{
			Timestamp:      startTime,
			Provider:       route.Provider,
			Model:          route.Model,
			RequestedModel: antReq.Model,
			Sensitive:      sensitive,
			Precision:      precision,
			Usecase:        usecase,
			Success:        false,
			Error:          fmt.Sprintf("model_disabled: %s", route.Model),
			LatencyMs:      time.Since(startTime).Milliseconds(),
			RequestBody:    body,
			ClientIP:       getClientIP(r),
		}
		requestLogger.LogRequest(logEntry)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "api_error",
				Message: fmt.Sprintf("Model %s is currently disabled", route.Model),
			},
		})
		return
	}

	// Handle streaming vs non-streaming
	if antReq.Stream {
		handleAnthropicMessagesStreaming(w, r, &antReq, openaiReq, route, body, usecase, sensitive, precision, startTime)
		return
	}

	// Get provider
	provider, ok := chatProviders[route.Provider]
	if !ok {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "api_error",
				Message: "Provider not available: " + route.Provider,
			},
		})
		return
	}

	// Execute request
	resp, err := provider.Chat(openaiReq, route.Model)
	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		// Log the failed request
		logEntry := &domain.RequestLog{
			Timestamp:      startTime,
			RequestType:    "anthropic-messages",
			Provider:       route.Provider,
			Model:          route.Model,
			RequestedModel: antReq.Model,
			Usecase:        usecase,
			HasImages:      openaiReq.HasImages(),
			Sensitive:      sensitive,
			Precision:      precision,
			Success:        false,
			Error:          err.Error(),
			LatencyMs:      latencyMs,
			RequestBody:    body,
		}
		requestLogger.LogRequest(logEntry)
		metrics.recordRequest(route.Provider, route.Model, "error", latencyMs, 0, 0, 0, false)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(AnthropicErrorResponse{
			Type: "error",
			Error: AnthropicError{
				Type:    "api_error",
				Message: err.Error(),
			},
		})
		return
	}

	// Convert OpenAI response to Anthropic format
	antResp := convertOpenAIToAnthropic(resp, antReq.Model)

	// Log successful request
	inputTokens := 0
	outputTokens := 0
	if resp.Usage != nil {
		inputTokens = resp.Usage.PromptTokens
		outputTokens = resp.Usage.CompletionTokens
	}
	cost := calculateCost(route.Model, inputTokens, outputTokens)

	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestType:    "anthropic-messages",
		Provider:       route.Provider,
		Model:          route.Model,
		RequestedModel: antReq.Model,
		Usecase:        usecase,
		HasImages:      openaiReq.HasImages(),
		Sensitive:      sensitive,
		Precision:      precision,
		Success:        true,
		LatencyMs:      latencyMs,
		InputTokens:    inputTokens,
		OutputTokens:   outputTokens,
		CostUSD:        cost,
		RequestBody:    body,
	}
	respBody, _ := json.Marshal(antResp)
	logEntry.ResponseBody = respBody
	requestID := requestLogger.LogRequest(logEntry)
	metrics.recordRequest(route.Provider, route.Model, "success", latencyMs, inputTokens, outputTokens, cost, false)

	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Request-ID", fmt.Sprintf("%d", requestID))
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", latencyMs))
	json.NewEncoder(w).Encode(antResp)
}

// handleAnthropicMessagesStreaming handles streaming responses in Anthropic SSE format
func handleAnthropicMessagesStreaming(w http.ResponseWriter, r *http.Request, antReq *AnthropicMessagesRequest, openaiReq *domain.ChatCompletionRequest, route *domain.RouteConfig, body []byte, usecase string, sensitive bool, precision string, startTime time.Time) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Generate message ID
	msgID := fmt.Sprintf("msg_%d", time.Now().UnixNano())

	// Helper to write SSE event
	writeEvent := func(event string, data interface{}) {
		jsonData, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, jsonData)
		flusher.Flush()
	}

	// Estimate input tokens
	inputTokens := 0
	for _, msg := range openaiReq.Messages {
		if content, ok := msg.Content.(string); ok {
			inputTokens += len(content) / 4
		}
	}

	// Send message_start event
	writeEvent("message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":            msgID,
			"type":          "message",
			"role":          "assistant",
			"content":       []interface{}{},
			"model":         antReq.Model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]interface{}{
				"input_tokens":  inputTokens,
				"output_tokens": 1,
			},
		},
	})

	// For Ollama, we need to stream from the Ollama API
	if route.Provider == "ollama" {
		streamAnthropicFromOllama(w, flusher, writeEvent, openaiReq, route.Model, antReq.Model, msgID, body, usecase, sensitive, precision, startTime, inputTokens)
		return
	}

	// For non-Ollama providers, get the full response and simulate streaming
	provider, ok := chatProviders[route.Provider]
	if !ok {
		writeEvent("error", map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "api_error",
				"message": "Provider not available: " + route.Provider,
			},
		})
		return
	}

	resp, err := provider.Chat(openaiReq, route.Model)
	if err != nil {
		writeEvent("error", map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "api_error",
				"message": err.Error(),
			},
		})
		return
	}

	// Simulate streaming the response
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]

		// Handle text content
		if content, ok := choice.Message.Content.(string); ok && content != "" {
			// Send content_block_start
			writeEvent("content_block_start", map[string]interface{}{
				"type":  "content_block_start",
				"index": 0,
				"content_block": map[string]interface{}{
					"type": "text",
					"text": "",
				},
			})

			// Stream text in chunks
			chunkSize := 20
			for i := 0; i < len(content); i += chunkSize {
				end := i + chunkSize
				if end > len(content) {
					end = len(content)
				}
				chunk := content[i:end]
				writeEvent("content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": 0,
					"delta": map[string]interface{}{
						"type": "text_delta",
						"text": chunk,
					},
				})
			}

			// Send content_block_stop
			writeEvent("content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": 0,
			})
		}

		// Handle tool calls
		for i, tc := range choice.Message.ToolCalls {
			idx := i
			if choice.Message.Content != nil {
				idx = i + 1
			}

			var input map[string]interface{}
			json.Unmarshal([]byte(tc.Function.Arguments), &input)

			// Send content_block_start for tool_use
			writeEvent("content_block_start", map[string]interface{}{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]interface{}{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": map[string]interface{}{},
				},
			})

			// Send input as delta
			writeEvent("content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]interface{}{
					"type":         "input_json_delta",
					"partial_json": tc.Function.Arguments,
				},
			})

			// Send content_block_stop
			writeEvent("content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": idx,
			})
		}
	}

	// Calculate output tokens
	outputTokens := 0
	if resp.Usage != nil {
		outputTokens = resp.Usage.CompletionTokens
	}

	// Determine stop reason
	stopReason := "end_turn"
	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop":
			stopReason = "end_turn"
		case "length":
			stopReason = "max_tokens"
		case "tool_calls":
			stopReason = "tool_use"
		}
	}

	// Send message_delta
	writeEvent("message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": outputTokens,
		},
	})

	// Send message_stop
	writeEvent("message_stop", map[string]interface{}{
		"type": "message_stop",
	})

	// Log the request
	latencyMs := time.Since(startTime).Milliseconds()
	cost := calculateCost(route.Model, inputTokens, outputTokens)
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestType:    "anthropic-messages-stream",
		Provider:       route.Provider,
		Model:          route.Model,
		RequestedModel: antReq.Model,
		Usecase:        usecase,
		HasImages:      openaiReq.HasImages(),
		Sensitive:      sensitive,
		Precision:      precision,
		Success:        true,
		LatencyMs:      latencyMs,
		InputTokens:    inputTokens,
		OutputTokens:   outputTokens,
		CostUSD:        cost,
		RequestBody:    body,
	}
	requestLogger.LogRequest(logEntry)
	metrics.recordRequest(route.Provider, route.Model, "success", latencyMs, inputTokens, outputTokens, cost, false)
}

// streamAnthropicFromOllama streams Ollama responses in Anthropic SSE format
func streamAnthropicFromOllama(w http.ResponseWriter, flusher http.Flusher, writeEvent func(string, interface{}), openaiReq *domain.ChatCompletionRequest, model string, requestedModel string, msgID string, body []byte, usecase string, sensitive bool, precision string, startTime time.Time, inputTokens int) {
	// Convert messages to Ollama format
	type ollamaMessage struct {
		Role    string   `json:"role"`
		Content string   `json:"content"`
		Images  []string `json:"images,omitempty"`
	}

	var messages []ollamaMessage
	for _, msg := range openaiReq.Messages {
		ollamaMsg := ollamaMessage{Role: msg.Role}
		switch c := msg.Content.(type) {
		case string:
			ollamaMsg.Content = c
		case []interface{}:
			var textParts []string
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "text" {
						textParts = append(textParts, m["text"].(string))
					} else if m["type"] == "image_url" {
						if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
							url := imgURL["url"].(string)
							if strings.HasPrefix(url, "data:") {
								parts := strings.SplitN(url, ",", 2)
								if len(parts) == 2 {
									ollamaMsg.Images = append(ollamaMsg.Images, parts[1])
								}
							}
						}
					}
				}
			}
			ollamaMsg.Content = strings.Join(textParts, "\n")
		}
		messages = append(messages, ollamaMsg)
	}

	ollamaReq := map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   true,
	}
	if len(openaiReq.Tools) > 0 {
		ollamaReq["tools"] = openaiReq.Tools
	}

	reqBody, _ := json.Marshal(ollamaReq)
	httpReq, _ := http.NewRequest("POST", "http://"+ollamaHost+"/api/chat", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 240 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		writeEvent("error", map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "api_error",
				"message": "Ollama request failed: " + err.Error(),
			},
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		writeEvent("error", map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "api_error",
				"message": fmt.Sprintf("Ollama error %d: %s", resp.StatusCode, string(respBody)),
			},
		})
		return
	}

	// Send content_block_start for text
	writeEvent("content_block_start", map[string]interface{}{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]interface{}{
			"type": "text",
			"text": "",
		},
	})

	// Stream from Ollama
	decoder := json.NewDecoder(resp.Body)
	var fullContent strings.Builder
	var toolCalls []map[string]interface{}

	for {
		var chunk struct {
			Model   string `json:"model"`
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id,omitempty"`
					Function struct {
						Name      string                 `json:"name"`
						Arguments map[string]interface{} `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"message"`
			Done bool `json:"done"`
		}

		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			break
		}

		// Stream text content
		if chunk.Message.Content != "" {
			fullContent.WriteString(chunk.Message.Content)
			writeEvent("content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]interface{}{
					"type": "text_delta",
					"text": chunk.Message.Content,
				},
			})
		}

		// Collect tool calls
		for _, tc := range chunk.Message.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Function.Arguments)
			toolID := tc.ID
			if toolID == "" {
				toolID = fmt.Sprintf("toolu_%d", time.Now().UnixNano())
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":        toolID,
				"name":      tc.Function.Name,
				"arguments": string(argsJSON),
			})
		}

		if chunk.Done {
			break
		}
	}

	// Send content_block_stop for text
	writeEvent("content_block_stop", map[string]interface{}{
		"type":  "content_block_stop",
		"index": 0,
	})

	// Send tool use blocks
	stopReason := "end_turn"
	for i, tc := range toolCalls {
		stopReason = "tool_use"
		idx := i + 1

		writeEvent("content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]interface{}{
				"type":  "tool_use",
				"id":    tc["id"],
				"name":  tc["name"],
				"input": map[string]interface{}{},
			},
		})

		writeEvent("content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]interface{}{
				"type":         "input_json_delta",
				"partial_json": tc["arguments"],
			},
		})

		writeEvent("content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": idx,
		})
	}

	// Estimate output tokens
	outputTokens := len(fullContent.String()) / 4

	// Send message_delta
	writeEvent("message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": outputTokens,
		},
	})

	// Send message_stop
	writeEvent("message_stop", map[string]interface{}{
		"type": "message_stop",
	})

	// Log the request
	latencyMs := time.Since(startTime).Milliseconds()
	cost := calculateCost(model, inputTokens, outputTokens)
	logEntry := &domain.RequestLog{
		Timestamp:      startTime,
		RequestType:    "anthropic-messages-stream",
		Provider:       "ollama",
		Model:          model,
		RequestedModel: requestedModel,
		Usecase:        usecase,
		HasImages:      openaiReq.HasImages(),
		Sensitive:      sensitive,
		Precision:      precision,
		Success:        true,
		LatencyMs:      latencyMs,
		InputTokens:    inputTokens,
		OutputTokens:   outputTokens,
		CostUSD:        cost,
		RequestBody:    body,
	}
	requestLogger.LogRequest(logEntry)
	metrics.recordRequest("ollama", model, "success", latencyMs, inputTokens, outputTokens, cost, false)
}

// convertAnthropicToOpenAI converts an Anthropic request to OpenAI format
func convertAnthropicToOpenAI(antReq *AnthropicMessagesRequest) *domain.ChatCompletionRequest {
	var messages []domain.Message

	// Handle system message(s)
	switch sys := antReq.System.(type) {
	case string:
		if sys != "" {
			messages = append(messages, domain.Message{
				Role:    "system",
				Content: sys,
			})
		}
	case []interface{}:
		// Array of system blocks
		var systemText strings.Builder
		for _, block := range sys {
			if m, ok := block.(map[string]interface{}); ok {
				if m["type"] == "text" {
					if text, ok := m["text"].(string); ok {
						if systemText.Len() > 0 {
							systemText.WriteString("\n\n")
						}
						systemText.WriteString(text)
					}
				}
			}
		}
		if systemText.Len() > 0 {
			messages = append(messages, domain.Message{
				Role:    "system",
				Content: systemText.String(),
			})
		}
	}

	// Convert messages
	for _, msg := range antReq.Messages {
		openaiMsg := domain.Message{Role: msg.Role}

		switch content := msg.Content.(type) {
		case string:
			openaiMsg.Content = content
		case []interface{}:
			// Array of content blocks
			var contentParts []domain.ContentPart
			var toolCalls []domain.ToolCall

			for _, block := range content {
				if m, ok := block.(map[string]interface{}); ok {
					blockType, _ := m["type"].(string)
					switch blockType {
					case "text":
						text, _ := m["text"].(string)
						contentParts = append(contentParts, domain.ContentPart{
							Type: "text",
							Text: text,
						})
					case "image":
						// Anthropic image format -> OpenAI format
						if source, ok := m["source"].(map[string]interface{}); ok {
							mediaType, _ := source["media_type"].(string)
							data, _ := source["data"].(string)
							contentParts = append(contentParts, domain.ContentPart{
								Type: "image_url",
								ImageURL: &domain.ImageURL{
									URL: fmt.Sprintf("data:%s;base64,%s", mediaType, data),
								},
							})
						}
					case "tool_use":
						// Assistant's tool calls
						id, _ := m["id"].(string)
						name, _ := m["name"].(string)
						input, _ := m["input"].(map[string]interface{})
						inputJSON, _ := json.Marshal(input)
						toolCalls = append(toolCalls, domain.ToolCall{
							ID:   id,
							Type: "function",
							Function: domain.ToolCallFunction{
								Name:      name,
								Arguments: string(inputJSON),
							},
						})
					case "tool_result":
						// This is a user message with tool results - convert to OpenAI tool role
						toolUseID, _ := m["tool_use_id"].(string)
						var resultContent string
						switch c := m["content"].(type) {
						case string:
							resultContent = c
						case []interface{}:
							// Array of text blocks
							for _, part := range c {
								if pm, ok := part.(map[string]interface{}); ok {
									if pm["type"] == "text" {
										if text, ok := pm["text"].(string); ok {
											resultContent += text
										}
									}
								}
							}
						}
						// Add as a separate tool message
						messages = append(messages, domain.Message{
							Role:       "tool",
							Content:    resultContent,
							ToolCallID: toolUseID,
						})
						continue
					}
				}
			}

			// Set content
			if len(contentParts) > 0 {
				// Convert to []interface{} for JSON marshaling
				var parts []interface{}
				for _, cp := range contentParts {
					part := map[string]interface{}{"type": cp.Type}
					if cp.Type == "text" {
						part["text"] = cp.Text
					} else if cp.Type == "image_url" && cp.ImageURL != nil {
						part["image_url"] = map[string]interface{}{"url": cp.ImageURL.URL}
					}
					parts = append(parts, part)
				}
				if len(parts) == 1 {
					if parts[0].(map[string]interface{})["type"] == "text" {
						openaiMsg.Content = parts[0].(map[string]interface{})["text"]
					} else {
						openaiMsg.Content = parts
					}
				} else {
					openaiMsg.Content = parts
				}
			}
			if len(toolCalls) > 0 {
				openaiMsg.ToolCalls = toolCalls
				// OpenAI requires content to be a string (can be empty) when tool_calls present
				if openaiMsg.Content == nil {
					openaiMsg.Content = ""
				}
			}
		}

		// Ensure content is never nil for OpenAI compatibility
		if openaiMsg.Content == nil {
			openaiMsg.Content = ""
		}

		messages = append(messages, openaiMsg)
	}

	// Convert tools
	var tools []domain.Tool
	for _, t := range antReq.Tools {
		// Debug: log each tool
		log.Printf("[DEBUG] Tool received: name=%q type=%q has_input_schema=%v", t.Name, t.Type, t.InputSchema != nil)

		// Check if this is a server tool (like web_search_20250305)
		if strings.Contains(t.Type, "web_search") {
			log.Printf("[DEBUG] Detected web_search server tool, converting to OpenAI format")
			// Convert Anthropic web_search server tool to OpenAI web_search tool
			tools = append(tools, domain.Tool{
				Type: "web_search",
				// Function is nil for server tools
			})
		} else {
			// Regular function tool
			tools = append(tools, domain.Tool{
				Type: "function",
				Function: &domain.ToolFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.InputSchema,
				},
			})
		}
	}

	return &domain.ChatCompletionRequest{
		Model:       antReq.Model,
		Messages:    messages,
		MaxTokens:   antReq.MaxTokens,
		Temperature: antReq.Temperature,
		Tools:       tools,
		ToolChoice:  antReq.ToolChoice,
	}
}

// convertOpenAIToAnthropic converts an OpenAI response to Anthropic format
func convertOpenAIToAnthropic(resp *domain.ChatCompletionResponse, requestedModel string) *AnthropicMessagesResponse {
	var contentBlocks []AnthropicContentBlock

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]

		// Add text content
		if content, ok := choice.Message.Content.(string); ok && content != "" {
			contentBlocks = append(contentBlocks, AnthropicContentBlock{
				Type: "text",
				Text: content,
			})
		}

		// Add tool use blocks
		for _, tc := range choice.Message.ToolCalls {
			var input map[string]interface{}
			json.Unmarshal([]byte(tc.Function.Arguments), &input)
			contentBlocks = append(contentBlocks, AnthropicContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	// If no content, add empty text block
	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, AnthropicContentBlock{
			Type: "text",
			Text: "",
		})
	}

	// Map OpenAI finish reasons to Anthropic stop reasons
	stopReason := "end_turn"
	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop":
			stopReason = "end_turn"
		case "length":
			stopReason = "max_tokens"
		case "tool_calls":
			stopReason = "tool_use"
		case "content_filter":
			stopReason = "end_turn"
		}
	}

	// Use requested model in response (Claude Code expects this)
	model := requestedModel
	if model == "" {
		model = resp.Model
	}

	usage := AnthropicUsage{}
	if resp.Usage != nil {
		usage.InputTokens = resp.Usage.PromptTokens
		usage.OutputTokens = resp.Usage.CompletionTokens
	}

	return &AnthropicMessagesResponse{
		ID:         resp.ID,
		Type:       "message",
		Role:       "assistant",
		Content:    contentBlocks,
		Model:      model,
		StopReason: stopReason,
		Usage:      usage,
	}
}

// ==================== End Anthropic API Compatibility ====================

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

// Request history for replay feature
type RequestHistoryEntry struct {
	ID           int64   `json:"id"`
	Timestamp    string  `json:"timestamp"`
	RequestType  string  `json:"request_type"` // "llm", "tts", "stt"
	Provider     string  `json:"provider"`
	Model        string  `json:"model"`
	Sensitive    bool    `json:"sensitive"`
	Precision    string  `json:"precision"`
	Usecase      string  `json:"usecase"`
	HasImages    bool    `json:"has_images"`
	LatencyMs    int64   `json:"latency_ms"`
	CostUSD      float64 `json:"cost_usd"`
	Success      bool    `json:"success"`
	CacheKey     string  `json:"cache_key"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	IsReplay     bool    `json:"is_replay"`
	ClientIP     string  `json:"client_ip,omitempty"`
	// TTS/STT specific
	Voice           string `json:"voice,omitempty"`
	AudioDurationMs int64  `json:"audio_duration_ms,omitempty"`
	InputChars      int    `json:"input_chars,omitempty"`
	// Tool info
	Tools       []string `json:"tools,omitempty"`        // Tool names from request
	HasWebSearch bool     `json:"has_web_search,omitempty"` // Convenience flag for web_search tool
}

type RequestHistoryResponse struct {
	Requests []RequestHistoryEntry `json:"requests"`
	Total    int                   `json:"total"`
	Page     int                   `json:"page"`
	PageSize int                   `json:"page_size"`
}

func handleRequestHistory(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	pageSize := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &pageSize)
	}

	page := 1
	if p := r.URL.Query().Get("page"); p != "" {
		fmt.Sscanf(p, "%d", &page)
		if page < 1 {
			page = 1
		}
	}
	offset := (page - 1) * pageSize

	// Build WHERE conditions (shared between count and data queries)
	var conditions []string
	var filterArgs []interface{}

	// Filter by request_type if specified (can be comma-separated like "llm,tts")
	if typeFilter := r.URL.Query().Get("type"); typeFilter != "" {
		types := strings.Split(typeFilter, ",")
		placeholders := make([]string, len(types))
		for i, t := range types {
			placeholders[i] = "?"
			filterArgs = append(filterArgs, strings.TrimSpace(t))
		}
		conditions = append(conditions, "request_type IN ("+strings.Join(placeholders, ",")+")")
	}

	// Filter by usecase if specified
	if usecaseFilter := r.URL.Query().Get("usecase"); usecaseFilter != "" {
		conditions = append(conditions, "usecase = ?")
		filterArgs = append(filterArgs, usecaseFilter)
	}

	// Exclude usecases if specified (comma-separated list)
	if excludeUsecases := r.URL.Query().Get("exclude_usecases"); excludeUsecases != "" {
		usecases := strings.Split(excludeUsecases, ",")
		placeholders := make([]string, len(usecases))
		for i, u := range usecases {
			placeholders[i] = "?"
			filterArgs = append(filterArgs, strings.TrimSpace(u))
		}
		conditions = append(conditions, "usecase NOT IN ("+strings.Join(placeholders, ",")+")")
	}

	// Filter by model if specified
	if modelFilter := r.URL.Query().Get("model"); modelFilter != "" {
		conditions = append(conditions, "model = ?")
		filterArgs = append(filterArgs, modelFilter)
	}

	// Filter by sensitive if specified
	if sensitiveFilter := r.URL.Query().Get("sensitive"); sensitiveFilter != "" {
		conditions = append(conditions, "sensitive = ?")
		if sensitiveFilter == "1" || sensitiveFilter == "true" {
			filterArgs = append(filterArgs, 1)
		} else {
			filterArgs = append(filterArgs, 0)
		}
	}

	// Filter by precision if specified
	if precisionFilter := r.URL.Query().Get("precision"); precisionFilter != "" {
		conditions = append(conditions, "precision = ?")
		filterArgs = append(filterArgs, precisionFilter)
	}

	// Filter by client_ip if specified
	if clientIPFilter := r.URL.Query().Get("client_ip"); clientIPFilter != "" {
		conditions = append(conditions, "client_ip = ?")
		filterArgs = append(filterArgs, clientIPFilter)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = " WHERE " + strings.Join(conditions, " AND ")
	}

	// Get total count for pagination
	countQuery := "SELECT COUNT(*) FROM requests" + whereClause
	var total int
	if err := db.QueryRow(countQuery, filterArgs...).Scan(&total); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Build data query
	query := `
		SELECT id, timestamp, request_type, provider, model, sensitive, precision, usecase, has_images,
		       latency_ms, cost_usd, success, cache_key, input_tokens, output_tokens, is_replay,
		       voice, audio_duration_ms, input_chars, client_ip, request_body
		FROM requests
	` + whereClause + " ORDER BY id DESC LIMIT ? OFFSET ?"

	args := append(filterArgs, pageSize, offset)

	rows, err := db.Query(query, args...)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var history []RequestHistoryEntry
	for rows.Next() {
		var entry RequestHistoryEntry
		var requestType sql.NullString
		var precision sql.NullString
		var usecase sql.NullString
		var cacheKey sql.NullString
		var voice sql.NullString
		var isReplay sql.NullBool
		var clientIP sql.NullString
		var requestBody sql.NullString
		rows.Scan(&entry.ID, &entry.Timestamp, &requestType, &entry.Provider, &entry.Model,
			&entry.Sensitive, &precision, &usecase, &entry.HasImages, &entry.LatencyMs,
			&entry.CostUSD, &entry.Success, &cacheKey, &entry.InputTokens, &entry.OutputTokens,
			&isReplay, &voice, &entry.AudioDurationMs, &entry.InputChars, &clientIP, &requestBody)
		if requestType.Valid {
			entry.RequestType = requestType.String
		} else {
			entry.RequestType = "llm" // Default for old entries
		}
		if precision.Valid {
			entry.Precision = precision.String
		}
		if usecase.Valid {
			entry.Usecase = usecase.String
		}
		if cacheKey.Valid {
			entry.CacheKey = cacheKey.String
		}
		if voice.Valid {
			entry.Voice = voice.String
		}
		if isReplay.Valid {
			entry.IsReplay = isReplay.Bool
		}
		if clientIP.Valid {
			entry.ClientIP = clientIP.String
		}
		// Extract tool names from request body (handles both Chat Completions and Responses API formats)
		if requestBody.Valid && requestBody.String != "" {
			var reqData struct {
				Tools []struct {
					Type     string `json:"type"`
					Name     string `json:"name"`
					Function *struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tools"`
			}
			if json.Unmarshal([]byte(requestBody.String), &reqData) == nil && len(reqData.Tools) > 0 {
				for _, tool := range reqData.Tools {
					// Get tool name from different formats
					toolName := tool.Name
					if toolName == "" && tool.Function != nil && tool.Function.Name != "" {
						toolName = tool.Function.Name
					}
					if toolName == "" && tool.Type != "" {
						// Use type as name for server tools like web_search, code_interpreter, etc.
						toolName = tool.Type
					}
					if toolName != "" {
						entry.Tools = append(entry.Tools, toolName)
						// Check for web search tools
						if strings.Contains(toolName, "web_search") {
							entry.HasWebSearch = true
						}
					}
				}
			} else {
				// Try parsing as generic map to handle unexpected tool formats
				var genericReq map[string]interface{}
				if json.Unmarshal([]byte(requestBody.String), &genericReq) == nil {
					if tools, ok := genericReq["tools"].([]interface{}); ok {
						for _, t := range tools {
							if tool, ok := t.(map[string]interface{}); ok {
								var toolName string
								// Try direct name
								if name, ok := tool["name"].(string); ok && name != "" {
									toolName = name
								}
								// Try function.name
								if toolName == "" {
									if fn, ok := tool["function"].(map[string]interface{}); ok {
										if name, ok := fn["name"].(string); ok && name != "" {
											toolName = name
										}
									}
								}
								// Fall back to type
								if toolName == "" {
									if t, ok := tool["type"].(string); ok && t != "" {
										toolName = t
									}
								}
								if toolName != "" {
									entry.Tools = append(entry.Tools, toolName)
									if strings.Contains(toolName, "web_search") {
										entry.HasWebSearch = true
									}
								}
							}
						}
					}
				}
			}
		}
		history = append(history, entry)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(RequestHistoryResponse{
		Requests: history,
		Total:    total,
		Page:     page,
		PageSize: pageSize,
	})
}

// Pending requests API - returns currently in-flight requests
func handlePendingRequests(w http.ResponseWriter, r *http.Request) {
	pending := getPendingRequests()

	// Add elapsed time to each
	type PendingWithElapsed struct {
		*PendingRequest
		ElapsedMs int64 `json:"elapsed_ms"`
	}

	result := make([]PendingWithElapsed, len(pending))
	now := time.Now()
	for i, p := range pending {
		result[i] = PendingWithElapsed{
			PendingRequest: p,
			ElapsedMs:      now.Sub(p.StartTime).Milliseconds(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// Usecases history API - returns list of all distinct usecases from requests history
func handleUsecasesHistory(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	rows, err := db.Query(`
		SELECT DISTINCT usecase FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		ORDER BY usecase
	`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var usecases []string
	for rows.Next() {
		var usecase string
		rows.Scan(&usecase)
		usecases = append(usecases, usecase)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(usecases)
}

// handleUsecaseDistribution returns the distribution of sensitivity/precision combinations for a usecase
func handleUsecaseDistribution(w http.ResponseWriter, r *http.Request) {
	usecase := r.URL.Query().Get("usecase")
	timeRange := r.URL.Query().Get("range")

	// Convert range to SQL interval
	var interval string
	switch timeRange {
	case "24h":
		interval = "-1 day"
	case "7d":
		interval = "-7 days"
	default:
		interval = "-30 days"
		timeRange = "30d"
	}

	dbMutex.Lock()
	defer dbMutex.Unlock()

	type Distribution struct {
		Sensitive string `json:"sensitive"`
		Precision string `json:"precision"`
		RouteType string `json:"route_type"`
		Model     string `json:"model"`
		Count     int    `json:"count"`
	}

	var rows *sql.Rows
	var err error

	// Get distribution grouped by model to show overrides
	if usecase == "" {
		// Base config: show all requests (gives overall distribution)
		rows, err = db.Query(`
			SELECT
				CASE WHEN sensitive = 1 THEN 'true' ELSE 'false' END as sensitive,
				COALESCE(precision, 'medium') as precision,
				CASE WHEN has_images = 1 THEN 'vision' ELSE 'text' END as route_type,
				model,
				COUNT(*) as count
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '`+interval+`')
			GROUP BY sensitive, precision, route_type, model
			ORDER BY count DESC
		`)
	} else {
		// Specific usecase: show distribution for that usecase
		rows, err = db.Query(`
			SELECT
				CASE WHEN sensitive = 1 THEN 'true' ELSE 'false' END as sensitive,
				COALESCE(precision, 'medium') as precision,
				CASE WHEN has_images = 1 THEN 'vision' ELSE 'text' END as route_type,
				model,
				COUNT(*) as count
			FROM requests
			WHERE usecase = ?
				AND timestamp >= datetime('now', 'localtime', '`+interval+`')
			GROUP BY sensitive, precision, route_type, model
			ORDER BY count DESC
		`, usecase)
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var distribution []Distribution
	totalCount := 0
	for rows.Next() {
		var d Distribution
		rows.Scan(&d.Sensitive, &d.Precision, &d.RouteType, &d.Model, &d.Count)
		distribution = append(distribution, d)
		totalCount += d.Count
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"usecase":      usecase,
		"distribution": distribution,
		"total":        totalCount,
		"range":        timeRange,
	})
}

// handleModelStats returns model volume statistics, optionally filtered by usecase
func handleModelStats(w http.ResponseWriter, r *http.Request) {
	usecase := r.URL.Query().Get("usecase")
	timeRange := r.URL.Query().Get("range")

	// Convert range to SQL interval
	var interval string
	switch timeRange {
	case "24h":
		interval = "-1 day"
	case "7d":
		interval = "-7 days"
	default:
		interval = "-30 days"
	}

	dbMutex.Lock()
	defer dbMutex.Unlock()

	var rows *sql.Rows
	var err error

	if usecase == "" {
		// All usecases
		rows, err = db.Query(`
			SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '` + interval + `')
			GROUP BY model ORDER BY COUNT(*) DESC LIMIT 20
		`)
	} else {
		// Specific usecase
		rows, err = db.Query(`
			SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE usecase = ?
				AND timestamp >= datetime('now', 'localtime', '` + interval + `')
			GROUP BY model ORDER BY COUNT(*) DESC LIMIT 20
		`, usecase)
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var models []map[string]interface{}
	for rows.Next() {
		var model string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&model, &count, &cost, &avgLatency)
		models = append(models, map[string]interface{}{
			"model":          model,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

// Request detail API - returns full request/response data for a specific request
func handleRequestDetail(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	if idStr == "" {
		http.Error(w, "id parameter required", http.StatusBadRequest)
		return
	}

	var id int64
	if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil {
		http.Error(w, "Invalid id", http.StatusBadRequest)
		return
	}

	dbMutex.Lock()
	var entry struct {
		ID             int64          `json:"id"`
		Timestamp      string         `json:"timestamp"`
		Provider       string         `json:"provider"`
		Model          string         `json:"model"`
		RequestedModel sql.NullString `json:"-"`
		Sensitive      bool           `json:"sensitive"`
		Precision      sql.NullString `json:"-"`
		Usecase        sql.NullString `json:"-"`
		Cached         bool           `json:"cached"`
		InputTokens    int            `json:"input_tokens"`
		OutputTokens   int            `json:"output_tokens"`
		LatencyMs      int64          `json:"latency_ms"`
		CostUSD        float64        `json:"cost_usd"`
		Success        bool           `json:"success"`
		Error          sql.NullString `json:"-"`
		CacheKey       sql.NullString `json:"-"`
		HasImages      bool           `json:"has_images"`
		RequestBody    sql.NullString `json:"-"`
		ResponseBody   sql.NullString `json:"-"`
		ClientIP       sql.NullString `json:"-"`
	}

	err := db.QueryRow(`
		SELECT id, timestamp, provider, model, requested_model, sensitive, precision, usecase,
		       cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images,
		       request_body, response_body, client_ip
		FROM requests WHERE id = ?
	`, id).Scan(&entry.ID, &entry.Timestamp, &entry.Provider, &entry.Model,
		&entry.RequestedModel, &entry.Sensitive, &entry.Precision, &entry.Usecase, &entry.Cached,
		&entry.InputTokens, &entry.OutputTokens, &entry.LatencyMs, &entry.CostUSD,
		&entry.Success, &entry.Error, &entry.CacheKey, &entry.HasImages,
		&entry.RequestBody, &entry.ResponseBody, &entry.ClientIP)
	dbMutex.Unlock()

	if err == sql.ErrNoRows {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Build response with nullable fields
	response := map[string]interface{}{
		"id":            entry.ID,
		"timestamp":     entry.Timestamp,
		"provider":      entry.Provider,
		"model":         entry.Model,
		"sensitive":     entry.Sensitive,
		"cached":        entry.Cached,
		"input_tokens":  entry.InputTokens,
		"output_tokens": entry.OutputTokens,
		"latency_ms":    entry.LatencyMs,
		"cost_usd":      entry.CostUSD,
		"success":       entry.Success,
		"has_images":    entry.HasImages,
	}

	if entry.RequestedModel.Valid {
		response["requested_model"] = entry.RequestedModel.String
	}
	if entry.Precision.Valid {
		response["precision"] = entry.Precision.String
	}
	if entry.Usecase.Valid {
		response["usecase"] = entry.Usecase.String
	}
	if entry.Error.Valid && entry.Error.String != "" {
		response["error"] = entry.Error.String
	}
	if entry.ClientIP.Valid {
		response["client_ip"] = entry.ClientIP.String
	}

	// Try to get cached request/response content (cache first, then DB fallback)
	if entry.CacheKey.Valid {
		response["cache_key"] = entry.CacheKey.String
	}

	// Get request body - try cache first, then DB
	var reqBody []byte
	if entry.CacheKey.Valid {
		reqBody, _ = getCachedRequest(entry.CacheKey.String)
	}
	if reqBody == nil && entry.RequestBody.Valid {
		reqBody = []byte(entry.RequestBody.String)
	}
	if reqBody != nil {
		// Try parsing as Chat Completions format first
		var req ChatCompletionRequest
		if json.Unmarshal(reqBody, &req) == nil && len(req.Messages) > 0 {
			// Extract messages for display (but truncate images)
			var displayMessages []map[string]interface{}
			for _, msg := range req.Messages {
				displayMsg := map[string]interface{}{
					"role": msg.Role,
				}
				if msg.Content != nil {
					// Check if content is a string or array
					if s, ok := msg.Content.(string); ok {
						displayMsg["content"] = s
					} else if arr, ok := msg.Content.([]interface{}); ok {
						// Array of content parts - handle images
						var parts []map[string]interface{}
						for _, part := range arr {
							if p, ok := part.(map[string]interface{}); ok {
								partCopy := make(map[string]interface{})
								for k, v := range p {
									// Keep full image_url data for preview in dashboard
									partCopy[k] = v
								}
								parts = append(parts, partCopy)
							}
						}
						displayMsg["content"] = parts
					} else {
						displayMsg["content"] = msg.Content
					}
				}
				displayMessages = append(displayMessages, displayMsg)
			}
			response["request"] = map[string]interface{}{
				"model":       req.Model,
				"messages":    displayMessages,
				"sensitive":   req.Sensitive,
				"precision":   req.Precision,
				"usecase":     req.Usecase,
				"no_cache":    req.NoCache,
				"tools":       req.Tools,
				"tool_choice": req.ToolChoice,
			}
		} else {
			// Try parsing as Responses API format (handles both standard and Codex CLI formats)
			var respReq struct {
				Model        string      `json:"model"`
				Input        interface{} `json:"input"`
				Instructions string      `json:"instructions,omitempty"`
				Tools        []struct {
					Type string `json:"type"`
					// Codex CLI format: fields at top level
					Name        string                 `json:"name,omitempty"`
					Description string                 `json:"description,omitempty"`
					Parameters  map[string]interface{} `json:"parameters,omitempty"`
					// Standard OpenAI format: fields nested in function
					Function *struct {
						Name        string                 `json:"name"`
						Description string                 `json:"description,omitempty"`
						Parameters  map[string]interface{} `json:"parameters,omitempty"`
					} `json:"function,omitempty"`
				} `json:"tools,omitempty"`
				ToolChoice interface{} `json:"tool_choice,omitempty"`
				Sensitive  *bool       `json:"sensitive,omitempty"`
				Precision  string      `json:"precision,omitempty"`
				Usecase    string      `json:"usecase,omitempty"`
			}
			if json.Unmarshal(reqBody, &respReq) == nil && (respReq.Input != nil || len(respReq.Tools) > 0) {
				// Build messages from Responses API input
				var displayMessages []map[string]interface{}
				if respReq.Instructions != "" {
					displayMessages = append(displayMessages, map[string]interface{}{
						"role":    "system",
						"content": respReq.Instructions,
					})
				}
				switch input := respReq.Input.(type) {
				case string:
					displayMessages = append(displayMessages, map[string]interface{}{
						"role":    "user",
						"content": input,
					})
				case []interface{}:
					// Array of input items
					for _, item := range input {
						if m, ok := item.(map[string]interface{}); ok {
							displayMsg := map[string]interface{}{}
							if role, ok := m["role"].(string); ok {
								displayMsg["role"] = role
							}
							if content, ok := m["content"]; ok {
								displayMsg["content"] = content
							}
							if len(displayMsg) > 0 {
								displayMessages = append(displayMessages, displayMsg)
							}
						}
					}
				}

				// Convert tools to display format (handle both Codex CLI and standard formats)
				var displayTools []map[string]interface{}
				for _, tool := range respReq.Tools {
					displayTool := map[string]interface{}{
						"type": tool.Type,
					}

					// Get name, description, parameters from either top-level (Codex) or nested (standard)
					name := tool.Name
					desc := tool.Description
					params := tool.Parameters

					if tool.Function != nil {
						// Standard format - prefer nested values if present
						if tool.Function.Name != "" {
							name = tool.Function.Name
						}
						if tool.Function.Description != "" {
							desc = tool.Function.Description
						}
						if tool.Function.Parameters != nil {
							params = tool.Function.Parameters
						}
					}

					// Build function object for UI compatibility (UI expects tool.function.name, tool.function.description)
					if name != "" || desc != "" || params != nil {
						displayTool["function"] = map[string]interface{}{
							"name":        name,
							"description": desc,
							"parameters":  params,
						}
					}
					// Also set name at top level for simpler access
					if name != "" {
						displayTool["name"] = name
					}
					displayTools = append(displayTools, displayTool)
				}

				sensitive := false
				if respReq.Sensitive != nil {
					sensitive = *respReq.Sensitive
				}

				response["request"] = map[string]interface{}{
					"model":       respReq.Model,
					"messages":    displayMessages,
					"sensitive":   sensitive,
					"precision":   respReq.Precision,
					"usecase":     respReq.Usecase,
					"tools":       displayTools,
					"tool_choice": respReq.ToolChoice,
				}
			}
		}
	}

	// Get response - try cache first, then DB
	var respBody []byte
	if entry.CacheKey.Valid {
		respBody, _ = getCachedResponse(entry.CacheKey.String)
	}
	if respBody == nil && entry.ResponseBody.Valid {
		respBody = []byte(entry.ResponseBody.String)
	}
	if respBody != nil {
		var resp map[string]interface{}
		if json.Unmarshal(respBody, &resp) == nil {
			response["response"] = resp
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
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
	internalReq, _ := http.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqBytes))
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

func handleClearCache(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" && r.Method != "DELETE" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	requestCache.Clear()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Cache cleared",
	})
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	stats := map[string]interface{}{}

	// Total requests
	var total int
	db.QueryRow("SELECT COUNT(*) FROM requests").Scan(&total)
	stats["total_requests"] = total

	// Cached requests
	var cached int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE cached = 1").Scan(&cached)
	stats["cached_requests"] = cached
	stats["cache_hit_rate"] = 0.0
	if total > 0 {
		stats["cache_hit_rate"] = float64(cached) / float64(total)
	}

	// Total cost
	var totalCost float64
	db.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests").Scan(&totalCost)
	stats["total_cost_usd"] = totalCost

	// By provider
	rows, _ := db.Query(`
		SELECT provider, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests GROUP BY provider
	`)
	defer rows.Close()

	byProvider := map[string]interface{}{}
	for rows.Next() {
		var provider string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&provider, &count, &cost, &avgLatency)
		byProvider[provider] = map[string]interface{}{
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		}
	}
	stats["by_provider"] = byProvider

	// By model - return all models, sorted by cost (for cost charts)
	rows, _ = db.Query(`
		SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests GROUP BY model ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	byModel := []map[string]interface{}{}
	for rows.Next() {
		var model string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&model, &count, &cost, &avgLatency)
		byModel = append(byModel, map[string]interface{}{
			"model":          model,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_model"] = byModel

	// By usecase - return all usecases, sorted by cost
	rows, _ = db.Query(`
		SELECT COALESCE(usecase, ''), COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests WHERE usecase IS NOT NULL AND usecase != '' GROUP BY usecase ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	byUsecase := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&usecase, &count, &cost, &avgLatency)
		byUsecase = append(byUsecase, map[string]interface{}{
			"usecase":        usecase,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_usecase"] = byUsecase

	// By usecase + sensitivity + precision
	rows, _ = db.Query(`
		SELECT COALESCE(usecase, ''), sensitive, COALESCE(precision, ''), COUNT(*), AVG(latency_ms)
		FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		GROUP BY usecase, sensitive, precision
		ORDER BY COUNT(*) DESC
	`)
	defer rows.Close()

	byUsecaseDetail := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var sensitive bool
		var precision string
		var count int
		var avgLatency float64
		rows.Scan(&usecase, &sensitive, &precision, &count, &avgLatency)
		byUsecaseDetail = append(byUsecaseDetail, map[string]interface{}{
			"usecase":        usecase,
			"sensitive":      sensitive,
			"precision":      precision,
			"count":          count,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_usecase_detail"] = byUsecaseDetail

	// Recent requests - with optional filtering
	modelFilter := r.URL.Query().Get("model")
	sensitiveFilter := r.URL.Query().Get("sensitive")
	precisionFilter := r.URL.Query().Get("precision")

	recentQuery := `
		SELECT id, timestamp, provider, model, cached, latency_ms, cost_usd, success, error,
		       sensitive, precision, has_images, input_tokens, output_tokens, usecase
		FROM requests WHERE 1=1`
	var args []interface{}

	if modelFilter != "" {
		recentQuery += " AND model = ?"
		args = append(args, modelFilter)
	}
	if sensitiveFilter != "" {
		recentQuery += " AND sensitive = ?"
		if sensitiveFilter == "1" || sensitiveFilter == "true" {
			args = append(args, 1)
		} else {
			args = append(args, 0)
		}
	}
	if precisionFilter != "" {
		recentQuery += " AND precision = ?"
		args = append(args, precisionFilter)
	}
	recentQuery += " ORDER BY id DESC LIMIT 50"

	rows, _ = db.Query(recentQuery, args...)
	defer rows.Close()

	recent := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var cached, success bool
		var latencyMs int64
		var costUsd float64
		var errStr sql.NullString
		var sensitive bool
		var precision sql.NullString
		var hasImages bool
		var inputTokens, outputTokens int
		var usecase sql.NullString
		rows.Scan(&id, &timestamp, &provider, &model, &cached, &latencyMs, &costUsd, &success, &errStr,
			&sensitive, &precision, &hasImages, &inputTokens, &outputTokens, &usecase)

		entry := map[string]interface{}{
			"id":            id,
			"timestamp":     timestamp,
			"provider":      provider,
			"model":         model,
			"cached":        cached,
			"latency_ms":    latencyMs,
			"cost_usd":      costUsd,
			"success":       success,
			"sensitive":     sensitive,
			"has_images":    hasImages,
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if precision.Valid {
			entry["precision"] = precision.String
		}
		if usecase.Valid {
			entry["usecase"] = usecase.String
		}
		recent = append(recent, entry)
	}
	stats["recent_requests"] = recent

	// Include active filters in response
	if modelFilter != "" || sensitiveFilter != "" || precisionFilter != "" {
		stats["filters"] = map[string]string{
			"model":     modelFilter,
			"sensitive": sensitiveFilter,
			"precision": precisionFilter,
		}
	}

	// Today's stats
	today := time.Now().Format("2006-01-02")
	var todayCount int
	var todayCost float64
	db.QueryRow("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM requests WHERE timestamp LIKE ?", today+"%").Scan(&todayCount, &todayCost)
	stats["today"] = map[string]interface{}{
		"requests": todayCount,
		"cost_usd": todayCost,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleTiming returns timing estimates for requests, useful for progress bars
func handleTiming(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")

	timing := map[string]interface{}{}

	// Vision requests by precision (for screenshot description)
	// Use p75 percentile for better estimates (not skewed by outliers)
	rows, _ := db.Query(`
		SELECT precision, AVG(latency_ms), COUNT(*), MIN(latency_ms), MAX(latency_ms)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0 AND precision IS NOT NULL
		GROUP BY precision
	`)
	defer rows.Close()

	visionByPrecision := map[string]interface{}{}
	for rows.Next() {
		var precision string
		var avgLatency, minLatency, maxLatency float64
		var count int
		rows.Scan(&precision, &avgLatency, &count, &minLatency, &maxLatency)

		// Get p75 percentile for this precision (better estimate than mean)
		var p75Latency float64
		db.QueryRow(`
			SELECT latency_ms FROM (
				SELECT latency_ms, ROW_NUMBER() OVER (ORDER BY latency_ms) as rn,
				       COUNT(*) OVER () as total
				FROM requests
				WHERE has_images = 1 AND success = 1 AND cached = 0 AND precision = ?
			) WHERE rn = CAST(total * 0.75 AS INTEGER) + 1
		`, precision).Scan(&p75Latency)

		if p75Latency == 0 {
			p75Latency = avgLatency // Fallback to average
		}

		visionByPrecision[precision] = map[string]interface{}{
			"avg_ms":   int64(avgLatency),
			"p75_ms":   int64(p75Latency),
			"min_ms":   int64(minLatency),
			"max_ms":   int64(maxLatency),
			"count":    count,
			"avg_secs": avgLatency / 1000.0,
			"p75_secs": p75Latency / 1000.0,
		}
	}
	timing["vision_by_precision"] = visionByPrecision

	// Vision requests by sensitive flag
	rows, _ = db.Query(`
		SELECT sensitive, AVG(latency_ms), COUNT(*)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0
		GROUP BY sensitive
	`)
	defer rows.Close()

	visionBySensitive := map[string]interface{}{}
	for rows.Next() {
		var sensitive bool
		var avgLatency float64
		var count int
		rows.Scan(&sensitive, &avgLatency, &count)
		key := "cloud"
		if sensitive {
			key = "local"
		}
		visionBySensitive[key] = map[string]interface{}{
			"avg_ms":   int64(avgLatency),
			"count":    count,
			"avg_secs": avgLatency / 1000.0,
		}
	}
	timing["vision_by_sensitive"] = visionBySensitive

	// Overall vision average
	var overallAvg float64
	var overallCount int
	db.QueryRow(`
		SELECT AVG(latency_ms), COUNT(*)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0
	`).Scan(&overallAvg, &overallCount)
	timing["vision_overall"] = map[string]interface{}{
		"avg_ms":   int64(overallAvg),
		"count":    overallCount,
		"avg_secs": overallAvg / 1000.0,
	}

	// Default estimates if no data
	timing["defaults"] = map[string]interface{}{
		"low":       map[string]interface{}{"avg_secs": 15.0},
		"medium":    map[string]interface{}{"avg_secs": 20.0},
		"high":      map[string]interface{}{"avg_secs": 30.0},
		"very_high": map[string]interface{}{"avg_secs": 45.0},
	}

	json.NewEncoder(w).Encode(timing)
}

// handleAnalytics returns analytics data for charts
func handleAnalytics(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")

	analytics := map[string]interface{}{}

	// Cost breakdown by usecase (for pie chart)
	rows, _ := db.Query(`
		SELECT COALESCE(usecase, 'unspecified'), COALESCE(SUM(cost_usd), 0), COUNT(*)
		FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		GROUP BY usecase
		ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	costByUsecase := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var cost float64
		var count int
		rows.Scan(&usecase, &cost, &count)
		costByUsecase = append(costByUsecase, map[string]interface{}{
			"usecase": usecase,
			"cost":    cost,
			"count":   count,
		})
	}
	analytics["cost_by_usecase"] = costByUsecase

	// Query count matrix by model, sensitivity, precision
	rows, _ = db.Query(`
		SELECT model, sensitive, COALESCE(precision, 'unspecified'), COUNT(*), COALESCE(SUM(cost_usd), 0)
		FROM requests
		GROUP BY model, sensitive, precision
		ORDER BY model, sensitive, precision
	`)
	defer rows.Close()

	queryMatrix := []map[string]interface{}{}
	for rows.Next() {
		var model string
		var sensitive bool
		var precision string
		var count int
		var cost float64
		rows.Scan(&model, &sensitive, &precision, &count, &cost)
		queryMatrix = append(queryMatrix, map[string]interface{}{
			"model":     model,
			"sensitive": sensitive,
			"precision": precision,
			"count":     count,
			"cost":      cost,
		})
	}
	analytics["query_matrix"] = queryMatrix

	// Get all distinct models for filtering
	rows, _ = db.Query(`SELECT DISTINCT model FROM requests ORDER BY model`)
	defer rows.Close()

	models := []string{}
	for rows.Next() {
		var model string
		rows.Scan(&model)
		models = append(models, model)
	}
	analytics["models"] = models

	// Volume over time - configurable time range
	timeRange := r.URL.Query().Get("range")
	if timeRange == "" {
		timeRange = "7d"
	}

	var volumeQuery string
	switch timeRange {
	case "1h":
		// Last hour, grouped by 5-minute blocks
		volumeQuery = `
			SELECT
				strftime('%Y-%m-%d %H:', timestamp) || printf('%02d', (CAST(strftime('%M', timestamp) AS INTEGER) / 5) * 5) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '-1 hour')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	case "1d":
		// Last day, grouped by 1-hour blocks
		volumeQuery = `
			SELECT
				strftime('%Y-%m-%d %H:00', timestamp) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '-1 day')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	case "30d":
		// Last 30 days, grouped by day
		volumeQuery = `
			SELECT
				DATE(timestamp) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= DATE('now', 'localtime', '-30 days')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	default: // 7d
		// Last 7 days, grouped by 4-hour blocks
		volumeQuery = `
			SELECT
				DATE(timestamp) || ' ' || printf('%02d', (CAST(strftime('%H', timestamp) AS INTEGER) / 4) * 4) || ':00' as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= DATE('now', 'localtime', '-7 days')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	}

	rows, _ = db.Query(volumeQuery)
	defer rows.Close()

	volumeOverTime := []map[string]interface{}{}
	for rows.Next() {
		var timeBlock, model string
		var count int
		var cost float64
		rows.Scan(&timeBlock, &model, &count, &cost)
		volumeOverTime = append(volumeOverTime, map[string]interface{}{
			"day":   timeBlock,
			"model": model,
			"count": count,
			"cost":  cost,
		})
	}
	analytics["volume_over_time"] = volumeOverTime
	analytics["time_range"] = timeRange

	// Hourly distribution (for the last 7 days)
	rows, _ = db.Query(`
		SELECT strftime('%H', timestamp) as hour, COUNT(*)
		FROM requests
		WHERE timestamp >= DATE('now', 'localtime', '-7 days')
		GROUP BY hour
		ORDER BY hour
	`)
	defer rows.Close()

	hourlyDistribution := []map[string]interface{}{}
	for rows.Next() {
		var hour string
		var count int
		rows.Scan(&hour, &count)
		hourlyDistribution = append(hourlyDistribution, map[string]interface{}{
			"hour":  hour,
			"count": count,
		})
	}
	analytics["hourly_distribution"] = hourlyDistribution

	// Model-specific stats (for detail view)
	modelParam := r.URL.Query().Get("model")
	if modelParam != "" {
		// Daily volume for specific model
		rows, _ = db.Query(`
			SELECT DATE(timestamp) as day, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE model = ? AND timestamp >= DATE('now', 'localtime', '-30 days')
			GROUP BY DATE(timestamp)
			ORDER BY day ASC
		`, modelParam)
		defer rows.Close()

		modelDailyVolume := []map[string]interface{}{}
		for rows.Next() {
			var day string
			var count int
			var cost, avgLatency float64
			rows.Scan(&day, &count, &cost, &avgLatency)
			modelDailyVolume = append(modelDailyVolume, map[string]interface{}{
				"day":         day,
				"count":       count,
				"cost":        cost,
				"avg_latency": avgLatency,
			})
		}
		analytics["model_daily_volume"] = modelDailyVolume

		// Model summary stats
		var totalCount int
		var totalCost, avgLatency float64
		var minLatency, maxLatency int64
		db.QueryRow(`
			SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms), MIN(latency_ms), MAX(latency_ms)
			FROM requests WHERE model = ?
		`, modelParam).Scan(&totalCount, &totalCost, &avgLatency, &minLatency, &maxLatency)

		analytics["model_summary"] = map[string]interface{}{
			"model":       modelParam,
			"total_count": totalCount,
			"total_cost":  totalCost,
			"avg_latency": avgLatency,
			"min_latency": minLatency,
			"max_latency": maxLatency,
		}
	}

	json.NewEncoder(w).Encode(analytics)
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
			"claude-opus-4-5-20251101",
			"claude-sonnet-4-5-20250929",
			"claude-haiku-4-5-20251001",
		},
		"openai": {
			"gpt-5.1",
			"gpt-4o",
			"gpt-4o-mini",
			"gpt-4-turbo",
			"gpt-3.5-turbo",
			"o1",
			"o1-mini",
		},
		"ollama": getOllamaModels(), // Fetch dynamically from Ollama
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
			"claude-opus-4-5-20251101",
			"claude-sonnet-4-5-20250929",
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
			"gpt-5.1",
			"gpt-4o",
			"gpt-4o-mini",
			"gpt-4-turbo",
			"gpt-3.5-turbo",
			"o1",
			"o1-mini",
		}
		for _, m := range openaiModels {
			configs = append(configs, ModelConfig{
				Provider: "openai",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
		}

		// Ollama models
		ollamaModels := getOllamaModels()
		for _, m := range ollamaModels {
			configs = append(configs, ModelConfig{
				Provider: "ollama",
				Model:    m,
				Enabled:  !isModelDisabled(m),
			})
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

func handleTTSHistory(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	rows, _ := db.Query(`
		SELECT id, timestamp, provider, model, latency_ms, success, error, voice, input_chars
		FROM requests WHERE request_type = 'tts' ORDER BY id DESC LIMIT 50
	`)
	defer rows.Close()

	history := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var latencyMs int64
		var success bool
		var errStr, voice sql.NullString
		var inputChars int
		rows.Scan(&id, &timestamp, &provider, &model, &latencyMs, &success, &errStr, &voice, &inputChars)

		entry := map[string]interface{}{
			"id":          id,
			"timestamp":   timestamp,
			"provider":    provider,
			"model":       model,
			"latency_ms":  latencyMs,
			"success":     success,
			"input_chars": inputChars,
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if voice.Valid {
			entry["voice"] = voice.String
		}
		history = append(history, entry)
	}

	// Get stats
	var totalRequests int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'tts'").Scan(&totalRequests)
	var successCount int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'tts' AND success = 1").Scan(&successCount)
	var avgLatency float64
	db.QueryRow("SELECT COALESCE(AVG(latency_ms), 0) FROM requests WHERE request_type = 'tts'").Scan(&avgLatency)
	var totalChars int
	db.QueryRow("SELECT COALESCE(SUM(input_chars), 0) FROM requests WHERE request_type = 'tts'").Scan(&totalChars)

	result := map[string]interface{}{
		"history":        history,
		"total_requests": totalRequests,
		"success_count":  successCount,
		"avg_latency_ms": avgLatency,
		"total_chars":    totalChars,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func handleSTTHistory(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	rows, _ := db.Query(`
		SELECT id, timestamp, provider, model, latency_ms, success, error, sensitive, input_chars, output_tokens, response_body
		FROM requests WHERE request_type = 'stt' ORDER BY id DESC LIMIT 50
	`)
	defer rows.Close()

	history := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var latencyMs int64
		var success, sensitive bool
		var errStr, responseBody sql.NullString
		var inputChars, outputTokens int
		rows.Scan(&id, &timestamp, &provider, &model, &latencyMs, &success, &errStr, &sensitive, &inputChars, &outputTokens, &responseBody)

		entry := map[string]interface{}{
			"id":           id,
			"timestamp":    timestamp,
			"provider":     provider,
			"model":        model,
			"latency_ms":   latencyMs,
			"success":      success,
			"sensitive":    sensitive,
			"file_size":    inputChars,   // input_chars stores file size for STT
			"output_chars": outputTokens, // output_tokens stores transcription length
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if responseBody.Valid && responseBody.String != "" {
			entry["transcription"] = responseBody.String
		}
		history = append(history, entry)
	}

	// Get stats
	var totalRequests int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt'").Scan(&totalRequests)
	var successCount int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND success = 1").Scan(&successCount)
	var avgLatency float64
	db.QueryRow("SELECT COALESCE(AVG(latency_ms), 0) FROM requests WHERE request_type = 'stt'").Scan(&avgLatency)
	var localCount, cloudCount int
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND provider = 'local'").Scan(&localCount)
	db.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND provider = 'openai'").Scan(&cloudCount)

	result := map[string]interface{}{
		"history":        history,
		"total_requests": totalRequests,
		"success_count":  successCount,
		"avg_latency_ms": avgLatency,
		"local_count":    localCount,
		"cloud_count":    cloudCount,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}


func handleAnalyticsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.AnalyticsTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}


func handleStatsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.StatsTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}


func handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.DashboardTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

// handleRequestPage handles /request/{id} URLs and redirects to /?request={id}
func handleRequestPage(w http.ResponseWriter, r *http.Request) {
	// Extract ID from path like /request/123
	path := strings.TrimPrefix(r.URL.Path, "/request/")
	if path == "" || path == r.URL.Path {
		http.Error(w, "Request ID required", http.StatusBadRequest)
		return
	}
	// Redirect to dashboard with request param
	http.Redirect(w, r, "/?request="+path, http.StatusFound)
}


func handleTestPlayground(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.PlaygroundTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "service": "llm-proxy"})
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

	// Service info
	sb.WriteString("\n# HELP llm_proxy_info Service information\n")
	sb.WriteString("# TYPE llm_proxy_info gauge\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_info{version=\"1.0\",ollama_host=\"%s\"} 1\n", ollamaHost))

	w.Write([]byte(sb.String()))
}

// validateRoutingTableSecurity ensures sensitive data never goes to cloud providers
// This is a critical security check that runs on startup
func validateRoutingTableSecurity() {
	violations := []string{}

	// Check text routing table
	for sensitive, precisions := range routingTable {
		if sensitive == "true" {
			for precision, config := range precisions {
				if config != nil && config.Provider != "ollama" {
					violations = append(violations,
						fmt.Sprintf("TEXT sensitive=true precision=%s routes to %s/%s (must be ollama)",
							precision, config.Provider, config.Model))
				}
			}
		}
	}

	// Check vision routing table
	for sensitive, precisions := range visionRoutingTable {
		if sensitive == "true" {
			for precision, config := range precisions {
				if config != nil && config.Provider != "ollama" {
					violations = append(violations,
						fmt.Sprintf("VISION sensitive=true precision=%s routes to %s/%s (must be ollama)",
							precision, config.Provider, config.Model))
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

	log.Println("Security check passed: all sensitive routes use local Ollama")
}

// handleAIDAProxy proxies requests to Google's AIDA API (used by Jules CLI)
// This allows centralizing API tokens on the server and logging all requests
func handleAIDAProxy(w http.ResponseWriter, r *http.Request) {
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

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
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

	client := &http.Client{Timeout: 300 * time.Second}
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

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
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

	client := &http.Client{Timeout: 300 * time.Second}
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
		Timestamp       float64           `json:"timestamp"`
		Method          string            `json:"method"`
		URL             string            `json:"url"`
		RequestHeaders  map[string]string `json:"request_headers"`
		RequestBody     string            `json:"request_body"`
		ResponseStatus  int               `json:"response_status"`
		ResponseBody    string            `json:"response_body"`
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
	requestLogger = repository.NewSQLiteLogger(db)

	// Initialize router with routing tables
	router = app.NewRouter(routingTable, visionRoutingTable)

	// Load usecase route overrides from database
	loadRouteOverrides()

	// Load disabled models from database
	loadDisabledModels()

	// Initialize cache
	initCache()

	// SECURITY ASSERTION: Verify sensitive routes only use Ollama (local)
	// This runs on startup to catch configuration errors before serving requests
	validateRoutingTableSecurity()

	// Initialize provider adapters (ports -> implementations)
	initChatProviders()

	// Initialize chat handler (primary adapter)
	chatHandler = &httphandlers.ChatHandler{
		Router:          router,
		Providers:       chatProviders,
		Cache:           requestCache,
		Logger:          requestLogger,
		Metrics:         metrics,
		GenerateKey:     generateCacheKey,
		CalculateCost:   calculateCost,
		AddPending:      addPendingRequest,
		RemovePending:   removePendingRequest,
		IsModelDisabled: isModelDisabled,
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

	// Routes
	http.HandleFunc("/", handleDashboard)
	http.HandleFunc("/request/", handleRequestPage)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/metrics", handleMetrics)
	http.Handle("/v1/chat/completions", chatHandler)
	http.HandleFunc("/v1/messages", handleAnthropicMessages) // Anthropic API compatibility for Claude Code
	http.HandleFunc("/v1/estimate", handleEstimate)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/api/stats", withCORS(handleStats))
	http.HandleFunc("/api/stats/models", withCORS(handleModelStats))
	http.HandleFunc("/api/timing", withCORS(handleTiming))
	http.HandleFunc("/api/routes", withCORS(handleRoutes))
	http.HandleFunc("/api/routes/override", withCORS(handleRouteOverride))
	http.HandleFunc("/api/usecases", withCORS(handleUsecases))
	http.HandleFunc("/api/usecases/history", withCORS(handleUsecasesHistory))
	http.HandleFunc("/api/usecases/distribution", withCORS(handleUsecaseDistribution))
	http.HandleFunc("/api/models", withCORS(handleAvailableModels))
	http.HandleFunc("/api/models/config", withCORS(handleModelsConfig))
	http.HandleFunc("/api/models/vision", withCORS(handleVisionModels))
	http.HandleFunc("/api/history", withCORS(handleRequestHistory))
	http.HandleFunc("/api/request", withCORS(handleRequestDetail))
	http.HandleFunc("/api/replay", withCORS(handleReplayRequest))
	http.HandleFunc("/api/cache/clear", withCORS(handleClearCache))
	http.HandleFunc("/api/pending", withCORS(handlePendingRequests))
	http.HandleFunc("/api/tts-history", withCORS(handleTTSHistory))
	http.HandleFunc("/api/stt-history", withCORS(handleSTTHistory))
	http.HandleFunc("/api/analytics", withCORS(handleAnalytics))
	http.HandleFunc("/api/system-metrics", withCORS(handleSystemMetrics))
	http.HandleFunc("/api/system/truncate-logs", withCORS(handleTruncateLogs))
	http.HandleFunc("/api/system/db-stats", withCORS(handleDbStats))
	http.HandleFunc("/analytics", handleAnalyticsPage)
	http.HandleFunc("/stats", handleStatsPage)
	http.HandleFunc("/test", handleTestPlayground)
	http.HandleFunc("/v1/audio/transcriptions", handleWhisperTranscription)
	http.HandleFunc("/v1/audio/transcriptions/stream", handleWhisperStream)
	http.HandleFunc("/v1/audio/speech", handleTTS)
	http.HandleFunc("/tts", handleTTSCompat) // Compat endpoint for voice_cloning format (tts.lan)
	http.HandleFunc("/v1/websearch", handleWebSearch)
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

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
