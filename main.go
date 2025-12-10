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
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
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

	_ "github.com/mattn/go-sqlite3"
)

// Configuration
var (
	port             = getEnv("PORT", "8080")
	openaiKey        = getEnv("OPENAI_API_KEY", "")
	anthropicKey     = getEnv("ANTHROPIC_API_KEY", "")
	ollamaHost       = getEnv("OLLAMA_HOST", "localhost:11434")
	dataDir          = getEnv("DATA_DIR", "./data")
	cacheTTLHours    = 24 * 7                                                // 1 week cache
	whisperServerURL = getEnv("WHISPER_SERVER_URL", "http://localhost:8890") // Local whisper server
	ttsServerURL     = getEnv("TTS_SERVER_URL", "http://localhost:7788")     // Local TTS server (Kokoro)
)

// Model pricing per 1M tokens (input, output)
var modelPricing = map[string][2]float64{
	// OpenAI
	"gpt-5.1":       {5.00, 20.00},  // GPT-5.1
	"gpt-4o":        {2.50, 10.00},
	"gpt-4o-mini":   {0.15, 0.60},
	"gpt-4-turbo":   {10.00, 30.00},
	"gpt-4":         {30.00, 60.00},
	"gpt-3.5-turbo": {0.50, 1.50},
	"o1":            {15.00, 60.00},
	"o1-mini":       {3.00, 12.00},
	// Anthropic - Claude 4.5 models only
	"claude-opus-4-5-20251101":  {5.00, 25.00},
	"claude-sonnet-4-5-20250929": {3.00, 15.00},
	"claude-haiku-4-5-20251001": {1.00, 5.00},
	// Ollama (free)
	"qwen3-vl:30b":  {0, 0},
	"llama3:latest": {0, 0},
	"llama3.3:70b":  {0, 0},
	"gemma3:latest": {0, 0},
	"llava":         {0, 0},
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
		"low":       {Provider: "ollama", Model: "llama3:latest"},
	},
	// sensitive: true (text only, local)
	"true": {
		"very_high": nil, // Not available - Claude requires cloud
		"high":      {Provider: "ollama", Model: "llama3.3:70b"},
		"medium":    {Provider: "ollama", Model: "gemma3:latest"},
		"low":       {Provider: "ollama", Model: "llama3:latest"},
	},
}

// Vision routing (requests with images)
// Precision levels: very_high, medium, low (no "high" level)
var visionRoutingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (can use cloud)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-5-20250929"}, // Claude has great vision
		"medium":    {Provider: "openai", Model: "gpt-4o"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
	// sensitive: true (local only)
	"true": {
		"very_high": {Provider: "ollama", Model: "qwen3-vl:235b"}, // Largest local vision model (143GB)
		"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
}

// Router service for request routing decisions
var router *app.Router

// HTTP handler for chat completions (primary adapter)
var chatHandler *httphandlers.ChatHandler

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
		return []string{"llama3:latest", "gemma3:latest"} // fallback
	}
	defer resp.Body.Close()

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		log.Printf("Failed to decode Ollama models: %v", err)
		return []string{"llama3:latest", "gemma3:latest"} // fallback
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

	// Fix empty provider values from routing failures
	db.Exec(`UPDATE requests SET provider = 'routing_failed' WHERE provider = '' OR provider IS NULL`)
	// Default existing rows to 'llm' type
	db.Exec(`UPDATE requests SET request_type = 'llm' WHERE request_type IS NULL OR request_type = ''`)

	// Create indexes AFTER migrations ensure columns exist
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_provider ON requests(provider)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_model ON requests(model)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_request_type ON requests(request_type)`)

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
	}
}

// Whisper transcription handlers

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

	// Route based on sensitive flag
	sensitive := isSensitiveRequest(r)

	// Prepare log entry
	logEntry := &RequestLog{
		Timestamp:   startTime,
		RequestType: "stt",
		Sensitive:   sensitive,
		InputChars:  len(fileContent), // Store file size in InputChars field
	}

	var resp *WhisperTranscriptionResponse
	var provider string

	if sensitive {
		// Use local whisper server
		resp, err = callLocalWhisper(fileContent, header.Filename, model, language)
		provider = "local"
		logEntry.Provider = "local"
		logEntry.Model = "whisper-large-v3" // Local server uses whisper-large-v3
	} else {
		// Use OpenAI Whisper API
		resp, err = callOpenAIWhisper(fileContent, header.Filename, model, language)
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

func callLocalWhisper(fileContent []byte, filename, model, language string) (*WhisperTranscriptionResponse, error) {
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

func callOpenAIWhisper(fileContent []byte, filename, model, language string) (*WhisperTranscriptionResponse, error) {
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

// urlEncode encodes a string for use in a URL query parameter
func urlEncode(s string) string {
	return url.QueryEscape(s)
}

// handleChatCompletions is now handled by httphandlers.ChatHandler

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
	// TTS/STT specific
	Voice           string `json:"voice,omitempty"`
	AudioDurationMs int64  `json:"audio_duration_ms,omitempty"`
	InputChars      int    `json:"input_chars,omitempty"`
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
		       voice, audio_duration_ms, input_chars
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
		rows.Scan(&entry.ID, &entry.Timestamp, &requestType, &entry.Provider, &entry.Model,
			&entry.Sensitive, &precision, &usecase, &entry.HasImages, &entry.LatencyMs,
			&entry.CostUSD, &entry.Success, &cacheKey, &entry.InputTokens, &entry.OutputTokens,
			&isReplay, &voice, &entry.AudioDurationMs, &entry.InputChars)
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
	}

	err := db.QueryRow(`
		SELECT id, timestamp, provider, model, requested_model, sensitive, precision, usecase,
		       cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images,
		       request_body, response_body
		FROM requests WHERE id = ?
	`, id).Scan(&entry.ID, &entry.Timestamp, &entry.Provider, &entry.Model,
		&entry.RequestedModel, &entry.Sensitive, &entry.Precision, &entry.Usecase, &entry.Cached,
		&entry.InputTokens, &entry.OutputTokens, &entry.LatencyMs, &entry.CostUSD,
		&entry.Success, &entry.Error, &entry.CacheKey, &entry.HasImages,
		&entry.RequestBody, &entry.ResponseBody)
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
		var req ChatCompletionRequest
		if json.Unmarshal(reqBody, &req) == nil {
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
	w.Header().Set("Access-Control-Allow-Origin", "*")

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
	w.Header().Set("Access-Control-Allow-Origin", "*")

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

// Analytics HTML
const analyticsHTML = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Proxy - Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { color: #6366f1; margin-bottom: 8px; font-size: 28px; }
        .subtitle { color: #888; margin-bottom: 16px; }
        a { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }

        .nav-tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            background: #1a1a2e;
            padding: 4px;
            border-radius: 12px;
        }
        .nav-tab {
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-tab:hover { color: #e0e0e0; background: #252540; }
        .nav-tab.active { background: #6366f1; color: #fff; }

        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }
        @media (max-width: 1200px) {
            .charts-grid { grid-template-columns: 1fr; }
        }

        .chart-card {
            background: #1a1a2e;
            border-radius: 16px;
            padding: 24px;
            position: relative;
        }
        .chart-card.full-width {
            grid-column: 1 / -1;
        }
        .chart-title {
            font-size: 18px;
            font-weight: 600;
            color: #a5b4fc;
            margin-bottom: 16px;
        }
        .chart-container {
            position: relative;
            height: 320px;
        }
        .chart-container.tall {
            height: 400px;
        }

        .matrix-container {
            overflow-x: auto;
        }
        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            border-radius: 12px;
            overflow: hidden;
        }
        .matrix-table th, .matrix-table td {
            padding: 12px 14px;
            text-align: center;
            border: 1px solid #2d2d44;
        }
        .matrix-table th {
            background: #252540;
            color: #a5b4fc;
            font-weight: 600;
        }
        .matrix-table .sensitive-header {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border-bottom: 2px solid #ef4444;
        }
        .matrix-table .not-sensitive-header {
            background: rgba(34, 197, 94, 0.15);
            color: #86efac;
            border-bottom: 2px solid #22c55e;
        }
        .matrix-table .precision-header {
            font-size: 11px;
            text-transform: capitalize;
            font-weight: 500;
        }
        .matrix-table td {
            background: #1e1e30;
        }
        .matrix-table .model-cell {
            text-align: left;
            font-weight: 500;
            color: #6366f1;
            cursor: pointer;
            background: #1a1a2e;
        }
        .matrix-table .model-cell:hover {
            color: #818cf8;
            text-decoration: underline;
        }
        .matrix-cell {
            cursor: pointer;
            transition: all 0.15s;
        }
        .matrix-cell.cell-sensitive {
            background: rgba(239, 68, 68, 0.08);
        }
        .matrix-cell.cell-sensitive:hover {
            background: rgba(239, 68, 68, 0.2);
        }
        .matrix-cell.cell-not-sensitive {
            background: rgba(34, 197, 94, 0.08);
        }
        .matrix-cell.cell-not-sensitive:hover {
            background: rgba(34, 197, 94, 0.2);
        }
        .matrix-cell.clickable-cell {
            cursor: pointer;
        }
        .matrix-cell.clickable-cell:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .matrix-count {
            font-size: 16px;
            font-weight: 600;
            color: #e0e0e0;
        }
        .matrix-cost {
            font-size: 11px;
            color: #22c55e;
            margin-top: 2px;
        }

        .filter-row {
            display: flex;
            gap: 32px;
            align-items: center;
            margin-bottom: 16px;
            padding: 12px 16px;
            background: #252540;
            border-radius: 10px;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .filter-group label {
            color: #a5b4fc;
            font-size: 13px;
            font-weight: 500;
        }
        .filter-group input[type="range"] {
            width: 120px;
            accent-color: #6366f1;
            cursor: pointer;
        }
        .filter-group span {
            min-width: 50px;
            color: #e0e0e0;
            font-size: 13px;
            font-weight: 600;
        }
        .filter-info {
            color: #888;
            font-size: 12px;
            margin-left: auto;
        }

        .time-range-selector {
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
            background: #252540;
            padding: 4px;
            border-radius: 8px;
            width: fit-content;
        }
        .time-btn {
            padding: 6px 14px;
            border-radius: 6px;
            border: none;
            background: transparent;
            color: #888;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .time-btn:hover {
            color: #e0e0e0;
        }
        .time-btn.active {
            background: #6366f1;
            color: #fff;
        }

        .model-selector {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .model-btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid #374151;
            background: #1a1a2e;
            color: #888;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .model-btn:hover {
            border-color: #6366f1;
            color: #fff;
        }
        .model-btn.active {
            background: #6366f1;
            color: #fff;
            border-color: #6366f1;
        }

        .model-detail {
            display: none;
            animation: fadeIn 0.3s ease;
        }
        .model-detail.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-box {
            background: #252540;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .stat-box-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .stat-box-value {
            font-size: 24px;
            font-weight: bold;
            color: #6366f1;
        }
        .stat-box-value.cost { color: #22c55e; }
        .stat-box-value.latency { color: #f59e0b; }

        .back-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #6366f1;
            font-size: 14px;
            margin-bottom: 20px;
            cursor: pointer;
        }
        .back-link:hover {
            color: #818cf8;
        }

        .legend-inline {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 12px;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #888;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Proxy Analytics</h1>
        <p class="subtitle">Cost breakdown, usage patterns, and model performance</p>

        <div class="nav-tabs">
            <button class="nav-tab" onclick="window.location.href='/'">Requests</button>
            <button class="nav-tab" onclick="window.location.href='/test'">Playground</button>
            <button class="nav-tab" onclick="window.location.href='/#routing'">Routing</button>
            <button class="nav-tab active">Analytics</button>
        </div>

        <div id="overview-view">
            <div class="charts-grid">
                <!-- Cost by Model Bar Chart -->
                <div class="chart-card">
                    <div class="chart-title">Cost Breakdown by Model</div>
                    <div class="chart-container">
                        <canvas id="costByModelChart"></canvas>
                    </div>
                </div>

                <!-- Cost by Provider Pie Chart -->
                <div class="chart-card">
                    <div class="chart-title">Cost Breakdown by Provider</div>
                    <div class="chart-container">
                        <canvas id="costByProviderChart"></canvas>
                    </div>
                </div>

                <!-- Cost by Usecase Pie Chart -->
                <div class="chart-card">
                    <div class="chart-title">Cost Breakdown by Use Case</div>
                    <div class="chart-container">
                        <canvas id="costPieChart"></canvas>
                    </div>
                </div>

                <!-- Hourly Distribution -->
                <div class="chart-card">
                    <div class="chart-title">Request Volume by Hour (Last 7 Days)</div>
                    <div class="chart-container">
                        <canvas id="hourlyChart"></canvas>
                    </div>
                </div>

                <!-- Query Matrix by Model/Sensitivity/Precision -->
                <div class="chart-card full-width">
                    <div class="chart-title">Query Count Matrix (Model × Sensitivity × Precision)</div>
                    <p style="color: #888; font-size: 13px; margin-bottom: 12px;">Click on a model name to see detailed stats</p>
                    <div class="filter-row">
                        <div class="filter-group">
                            <label>Min Requests:</label>
                            <input type="range" id="minCountSlider" min="0" max="100" value="1" oninput="updateMatrixFilters()">
                            <span id="minCountValue">1</span>
                        </div>
                        <div class="filter-group">
                            <label>Min Cost ($):</label>
                            <input type="range" id="minCostSlider" min="0" max="100" value="0" step="1" oninput="updateMatrixFilters()">
                            <span id="minCostValue">$0.00</span>
                        </div>
                        <div class="filter-info" id="filterInfo">Showing all models</div>
                    </div>
                    <div class="matrix-container" id="matrixContainer">
                        <!-- Matrix will be generated here -->
                    </div>
                </div>

                <!-- Volume Over Time -->
                <div class="chart-card full-width">
                    <div class="chart-title">Request Volume Over Time</div>
                    <div class="time-range-selector">
                        <button class="time-btn" data-range="1h" onclick="selectTimeRange('1h')">1 Hour</button>
                        <button class="time-btn" data-range="1d" onclick="selectTimeRange('1d')">1 Day</button>
                        <button class="time-btn active" data-range="7d" onclick="selectTimeRange('7d')">7 Days</button>
                        <button class="time-btn" data-range="30d" onclick="selectTimeRange('30d')">30 Days</button>
                    </div>
                    <div class="model-selector" id="modelSelector">
                        <button class="model-btn active" data-model="all">All Models</button>
                    </div>
                    <div class="chart-container tall">
                        <canvas id="volumeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Detail View -->
        <div id="model-detail-view" class="model-detail">
            <div class="back-link" onclick="showOverview()">
                ← Back to Overview
            </div>
            <div class="chart-card">
                <div class="chart-title" id="modelDetailTitle">Model Details</div>
                <div class="stats-row" id="modelStatsRow">
                    <!-- Stats will be populated here -->
                </div>
                <div class="chart-container tall">
                    <canvas id="modelDetailChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let analyticsData = null;
        let statsData = null;
        let costPieChart = null;
        let costByModelChart = null;
        let costByProviderChart = null;
        let hourlyChart = null;
        let volumeChart = null;
        let modelDetailChart = null;
        let selectedVolumeModel = 'all';
        let selectedTimeRange = '7d';

        const colors = [
            '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#22c55e',
            '#14b8a6', '#06b6d4', '#3b82f6', '#ef4444', '#84cc16',
            '#f97316', '#a855f7', '#eab308', '#10b981', '#0ea5e9'
        ];

        async function loadAnalytics() {
            const [analyticsResponse, statsResponse] = await Promise.all([
                fetch('/api/analytics?range=' + selectedTimeRange),
                fetch('/api/stats')
            ]);
            analyticsData = await analyticsResponse.json();
            statsData = await statsResponse.json();
            renderCharts();
        }

        async function selectTimeRange(range) {
            selectedTimeRange = range;
            document.querySelectorAll('.time-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.range === range);
            });
            // Reload data with new range
            const response = await fetch('/api/analytics?range=' + range);
            const newData = await response.json();
            analyticsData.volume_over_time = newData.volume_over_time;
            analyticsData.time_range = newData.time_range;
            renderVolumeChart();
            renderModelSelector();
        }

        function renderCharts() {
            renderCostByModelChart();
            renderCostByProviderChart();
            renderCostPieChart();
            renderHourlyChart();
            initMatrixSliders();
            renderMatrix();
            renderVolumeChart();
            renderModelSelector();
        }

        function renderCostByModelChart() {
            const ctx = document.getElementById('costByModelChart').getContext('2d');
            // Filter models with cost > 0 and sort by cost descending
            const data = (statsData.by_model || [])
                .filter(d => d.cost_usd > 0)
                .sort((a, b) => b.cost_usd - a.cost_usd)
                .slice(0, 10); // Top 10 models

            if (costByModelChart) costByModelChart.destroy();

            costByModelChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.map(d => d.model),
                    datasets: [{
                        label: 'Cost ($)',
                        data: data.map(d => d.cost_usd),
                        backgroundColor: '#22c55e',
                        borderColor: '#16a34a',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    const item = data[context.dataIndex];
                                    return [
                                        'Cost: $' + item.cost_usd.toFixed(4),
                                        'Requests: ' + item.count.toLocaleString()
                                    ];
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2d2d44' },
                            ticks: {
                                color: '#888',
                                callback: function(value) { return '$' + value.toFixed(2); }
                            }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#888', font: { size: 11 } }
                        }
                    }
                }
            });
        }

        function renderCostByProviderChart() {
            const ctx = document.getElementById('costByProviderChart').getContext('2d');
            const providerData = statsData.by_provider || {};
            // Convert object to array and filter providers with cost > 0
            const data = Object.entries(providerData)
                .map(([name, stats]) => ({ name, ...stats }))
                .filter(d => d.cost_usd > 0)
                .sort((a, b) => b.cost_usd - a.cost_usd);

            if (costByProviderChart) costByProviderChart.destroy();

            const providerColors = {
                'anthropic': '#d97706',
                'openai': '#10b981',
                'ollama': '#6366f1',
                'local': '#8b5cf6',
                'kokoro': '#ec4899'
            };

            costByProviderChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.map(d => d.name.charAt(0).toUpperCase() + d.name.slice(1)),
                    datasets: [{
                        data: data.map(d => d.cost_usd),
                        backgroundColor: data.map(d => providerColors[d.name] || colors[0]),
                        borderColor: '#0f0f1a',
                        borderWidth: 2,
                        hoverOffset: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: '#888',
                                padding: 12,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    const item = data[context.dataIndex];
                                    const total = data.reduce((sum, d) => sum + d.cost_usd, 0);
                                    const percentage = ((item.cost_usd / total) * 100).toFixed(1);
                                    return [
                                        'Cost: $' + item.cost_usd.toFixed(4),
                                        'Requests: ' + item.count.toLocaleString(),
                                        'Share: ' + percentage + '%'
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        }

        function renderCostPieChart() {
            const ctx = document.getElementById('costPieChart').getContext('2d');
            const data = analyticsData.cost_by_usecase || [];

            if (costPieChart) costPieChart.destroy();

            costPieChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.map(d => d.usecase),
                    datasets: [{
                        data: data.map(d => d.cost),
                        backgroundColor: colors.slice(0, data.length),
                        borderColor: '#0f0f1a',
                        borderWidth: 2,
                        hoverOffset: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: '#888',
                                padding: 12,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    const item = data[context.dataIndex];
                                    const total = data.reduce((sum, d) => sum + d.cost, 0);
                                    const percentage = ((item.cost / total) * 100).toFixed(1);
                                    return [
                                        'Cost: $' + item.cost.toFixed(4),
                                        'Requests: ' + item.count.toLocaleString(),
                                        'Share: ' + percentage + '%'
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        }

        function renderHourlyChart() {
            const ctx = document.getElementById('hourlyChart').getContext('2d');
            const data = analyticsData.hourly_distribution || [];

            // Fill in missing hours
            const hourlyData = new Array(24).fill(0);
            data.forEach(d => {
                const hour = parseInt(d.hour);
                hourlyData[hour] = d.count;
            });

            if (hourlyChart) hourlyChart.destroy();

            hourlyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Array.from({length: 24}, (_, i) => i.toString().padStart(2, '0') + ':00'),
                    datasets: [{
                        label: 'Requests',
                        data: hourlyData,
                        backgroundColor: '#6366f1',
                        borderRadius: 4,
                        borderSkipped: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2d2d44' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            grid: { color: '#2d2d44' },
                            ticks: { color: '#888' }
                        }
                    }
                }
            });
        }

        let matrixMinCount = 1;
        let matrixMinCost = 0;
        let maxModelCount = 100;
        let maxModelCost = 1;

        function initMatrixSliders() {
            const data = analyticsData.query_matrix || [];

            // Calculate per-model totals
            const modelStats = {};
            data.forEach(d => {
                if (!modelStats[d.model]) {
                    modelStats[d.model] = { count: 0, cost: 0 };
                }
                modelStats[d.model].count += d.count;
                modelStats[d.model].cost += d.cost;
            });

            // Find max values for slider ranges
            maxModelCount = Math.max(...Object.values(modelStats).map(s => s.count), 1);
            maxModelCost = Math.max(...Object.values(modelStats).map(s => s.cost), 0.01);

            // Set slider max values
            document.getElementById('minCountSlider').max = maxModelCount;
            document.getElementById('minCostSlider').max = Math.ceil(maxModelCost * 100); // cents
        }

        function updateMatrixFilters() {
            matrixMinCount = parseInt(document.getElementById('minCountSlider').value);
            const costCents = parseInt(document.getElementById('minCostSlider').value);
            matrixMinCost = costCents / 100;

            document.getElementById('minCountValue').textContent = matrixMinCount;
            document.getElementById('minCostValue').textContent = '$' + matrixMinCost.toFixed(2);

            renderMatrix();
        }

        function renderMatrix() {
            const container = document.getElementById('matrixContainer');
            const data = analyticsData.query_matrix || [];

            // Calculate per-model totals for filtering
            const modelStats = {};
            data.forEach(d => {
                if (!modelStats[d.model]) {
                    modelStats[d.model] = { count: 0, cost: 0 };
                }
                modelStats[d.model].count += d.count;
                modelStats[d.model].cost += d.cost;
            });

            // Filter models based on sliders
            const allModels = [...new Set(data.map(d => d.model))].sort();
            const models = allModels.filter(m => {
                const stats = modelStats[m] || { count: 0, cost: 0 };
                return stats.count >= matrixMinCount && stats.cost >= matrixMinCost;
            });

            // Update filter info
            document.getElementById('filterInfo').textContent =
                'Showing ' + models.length + ' of ' + allModels.length + ' models';

            if (models.length === 0) {
                container.innerHTML = '<p style="color:#888;text-align:center;padding:40px;">No models match the current filters</p>';
                return;
            }

            const precisions = ['very_high', 'high', 'medium', 'low', 'unspecified'];
            const hasUnspecified = data.some(d => d.precision === 'unspecified');
            const activePrecisions = precisions.filter(p => p !== 'unspecified' || hasUnspecified);

            let html = '<table class="matrix-table"><thead>';

            // First header row: Sensitivity groups
            html += '<tr><th rowspan="2" style="vertical-align:middle;">Model</th>';
            html += '<th colspan="' + activePrecisions.length + '" class="sensitive-header">Sensitive</th>';
            html += '<th colspan="' + activePrecisions.length + '" class="not-sensitive-header">Not Sensitive</th>';
            html += '<th rowspan="2" style="vertical-align:middle;">Total</th>';
            html += '</tr>';

            // Second header row: Precision levels
            html += '<tr>';
            [true, false].forEach(sensitive => {
                activePrecisions.forEach(prec => {
                    const cls = sensitive ? 'sensitive-header' : 'not-sensitive-header';
                    html += '<th class="' + cls + ' precision-header">' + prec.replace('_', ' ') + '</th>';
                });
            });
            html += '</tr></thead><tbody>';

            models.forEach(model => {
                const stats = modelStats[model];
                html += '<tr><td class="model-cell" onclick="showModelDetail(\'' + model + '\')">' + model + '</td>';

                [true, false].forEach(sensitive => {
                    activePrecisions.forEach(prec => {
                        const item = data.find(d => d.model === model && d.sensitive === sensitive && d.precision === prec);
                        const bgClass = sensitive ? 'cell-sensitive' : 'cell-not-sensitive';
                        if (item && item.count > 0) {
                            const clickHandler = 'showMatrixRequests(\'' + encodeURIComponent(model) + '\',' + sensitive + ',\'' + prec + '\')';
                            html += '<td class="matrix-cell ' + bgClass + ' clickable-cell" onclick="' + clickHandler + '" title="Click to view ' + item.count + ' requests">';
                            html += '<div class="matrix-count">' + item.count.toLocaleString() + '</div>';
                            if (item.cost > 0) {
                                html += '<div class="matrix-cost">$' + item.cost.toFixed(4) + '</div>';
                            }
                            html += '</td>';
                        } else {
                            html += '<td class="matrix-cell ' + bgClass + '" style="color:#555;">-</td>';
                        }
                    });
                });

                // Total column
                html += '<td style="background:#252540;font-weight:600;">';
                html += '<div class="matrix-count">' + stats.count.toLocaleString() + '</div>';
                if (stats.cost > 0) {
                    html += '<div class="matrix-cost">$' + stats.cost.toFixed(4) + '</div>';
                }
                html += '</td>';

                html += '</tr>';
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        }

        function renderModelSelector() {
            const container = document.getElementById('modelSelector');
            const volumeData = analyticsData.volume_over_time || [];

            // Only show models with more than 1 request total
            const modelCounts = {};
            volumeData.forEach(d => {
                modelCounts[d.model] = (modelCounts[d.model] || 0) + d.count;
            });
            const activeModels = Object.keys(modelCounts).filter(m => modelCounts[m] > 1).sort();

            let html = '<button class="model-btn active" data-model="all" onclick="selectVolumeModel(\'all\')">All Models</button>';
            activeModels.forEach(model => {
                html += '<button class="model-btn" data-model="' + model + '" onclick="selectVolumeModel(\'' + model + '\')">' + model + ' <span style="color:#888;font-size:11px;">(' + modelCounts[model] + ')</span></button>';
            });
            container.innerHTML = html;
        }

        function selectVolumeModel(model) {
            selectedVolumeModel = model;
            document.querySelectorAll('#modelSelector .model-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.model === model);
            });
            renderVolumeChart();
        }

        function renderVolumeChart() {
            const ctx = document.getElementById('volumeChart').getContext('2d');
            const data = analyticsData.volume_over_time || [];

            if (volumeChart) volumeChart.destroy();

            // Get unique dates
            const dates = [...new Set(data.map(d => d.day))].sort();

            // Calculate model totals for filtering
            const modelCounts = {};
            data.forEach(d => {
                modelCounts[d.model] = (modelCounts[d.model] || 0) + d.count;
            });

            let datasets = [];

            if (selectedVolumeModel === 'all') {
                // Group by model, only include models with > 1 request
                const models = [...new Set(data.map(d => d.model))].filter(m => modelCounts[m] > 1);
                models.forEach((model, i) => {
                    const modelData = dates.map(date => {
                        const item = data.find(d => d.day === date && d.model === model);
                        return item ? item.count : 0;
                    });
                    datasets.push({
                        label: model,
                        data: modelData,
                        borderColor: colors[i % colors.length],
                        backgroundColor: colors[i % colors.length] + '20',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        borderWidth: 2
                    });
                });
            } else {
                // Single model
                const modelData = dates.map(date => {
                    const item = data.find(d => d.day === date && d.model === selectedVolumeModel);
                    return item ? item.count : 0;
                });
                datasets.push({
                    label: selectedVolumeModel,
                    data: modelData,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    borderWidth: 2
                });
            }

            volumeChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates.map(d => {
                        // Format: "2024-12-09 08:00" -> "Dec 9 08:00"
                        const parts = d.split(' ');
                        const datePart = new Date(parts[0]);
                        const timePart = parts[1] || '00:00';
                        return datePart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' + timePart;
                    }),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: selectedVolumeModel === 'all',
                            position: 'top',
                            labels: {
                                color: '#888',
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1,
                            filter: function(tooltipItem) {
                                // Only show models with > 0 requests at this point
                                return tooltipItem.raw > 0;
                            },
                            callbacks: {
                                label: function(context) {
                                    if (context.raw === 0) return null;
                                    return context.dataset.label + ': ' + context.raw + ' requests';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2d2d44' },
                            ticks: {
                                color: '#888',
                                maxRotation: 45,
                                minRotation: 45,
                                autoSkip: true,
                                maxTicksLimit: 20
                            }
                        },
                        y: {
                            grid: { color: '#2d2d44' },
                            ticks: { color: '#888' },
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function showMatrixRequests(model, sensitive, precision) {
            // Redirect to the main dashboard with filters applied
            const params = new URLSearchParams();
            params.set('model', decodeURIComponent(model));
            params.set('sensitive', sensitive ? '1' : '0');
            if (precision && precision !== 'unspecified') {
                params.set('precision', precision);
            }
            window.location.href = '/?matrix=' + encodeURIComponent(params.toString());
        }

        async function showModelDetail(model) {
            document.getElementById('overview-view').style.display = 'none';
            document.getElementById('model-detail-view').classList.add('active');
            document.getElementById('modelDetailTitle').textContent = model + ' - Detailed Analytics';

            // Fetch model-specific data
            const response = await fetch('/api/analytics?model=' + encodeURIComponent(model));
            const data = await response.json();

            // Render stats
            const summary = data.model_summary || {};
            document.getElementById('modelStatsRow').innerHTML =
                '<div class="stat-box"><div class="stat-box-label">Total Requests</div><div class="stat-box-value">' + (summary.total_count || 0).toLocaleString() + '</div></div>' +
                '<div class="stat-box"><div class="stat-box-label">Total Cost</div><div class="stat-box-value cost">$' + (summary.total_cost || 0).toFixed(4) + '</div></div>' +
                '<div class="stat-box"><div class="stat-box-label">Avg Latency</div><div class="stat-box-value latency">' + Math.round(summary.avg_latency || 0).toLocaleString() + 'ms</div></div>' +
                '<div class="stat-box"><div class="stat-box-label">Min Latency</div><div class="stat-box-value latency">' + (summary.min_latency || 0).toLocaleString() + 'ms</div></div>' +
                '<div class="stat-box"><div class="stat-box-label">Max Latency</div><div class="stat-box-value latency">' + (summary.max_latency || 0).toLocaleString() + 'ms</div></div>';

            // Render chart
            const ctx = document.getElementById('modelDetailChart').getContext('2d');
            const dailyData = data.model_daily_volume || [];

            if (modelDetailChart) modelDetailChart.destroy();

            modelDetailChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dailyData.map(d => {
                        const date = new Date(d.day);
                        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }),
                    datasets: [{
                        label: 'Requests',
                        data: dailyData.map(d => d.count),
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.2)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    }, {
                        label: 'Avg Latency (ms)',
                        data: dailyData.map(d => d.avg_latency),
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: false,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#888',
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1a1a2e',
                            titleColor: '#a5b4fc',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2d2d44',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2d2d44' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: { color: '#2d2d44' },
                            ticks: { color: '#6366f1' },
                            title: {
                                display: true,
                                text: 'Requests',
                                color: '#6366f1'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: { drawOnChartArea: false },
                            ticks: { color: '#f59e0b' },
                            title: {
                                display: true,
                                text: 'Latency (ms)',
                                color: '#f59e0b'
                            }
                        }
                    }
                }
            });
        }

        function showOverview() {
            document.getElementById('model-detail-view').classList.remove('active');
            document.getElementById('overview-view').style.display = 'block';
        }

        // Load on page load
        loadAnalytics();
    </script>
</body>
</html>`

func handleAnalyticsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(analyticsHTML))
}

// Dashboard HTML
const dashboardHTML = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Proxy</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #6366f1; margin-bottom: 8px; font-size: 28px; }
        .subtitle { color: #888; margin-bottom: 16px; }
        a { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Top Navigation Tabs */
        .nav-tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            background: #1a1a2e;
            padding: 4px;
            border-radius: 12px;
        }
        .nav-tab {
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-tab:hover { color: #e0e0e0; background: #252540; }
        .nav-tab.active { background: #6366f1; color: #fff; }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
        }
        .stat-label { color: #888; font-size: 14px; margin-bottom: 4px; }
        .stat-value { font-size: 32px; font-weight: bold; color: #6366f1; }
        .stat-value.cost { color: #22c55e; }
        .stat-value.latency { color: #f59e0b; }

        .section { margin-bottom: 24px; }
        .section-title { font-size: 18px; margin-bottom: 12px; color: #a5b4fc; }

        table {
            width: 100%;
            border-collapse: collapse;
            background: #1a1a2e;
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #2d2d44;
        }
        th { background: #252540; color: #a5b4fc; font-weight: 600; }
        tr.clickable { cursor: pointer; }
        tr.clickable:hover { background: #252540; }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge.openai { background: #10b981; color: #000; }
        .badge.anthropic { background: #f59e0b; color: #000; }
        .badge.ollama { background: #6366f1; color: #fff; }
        .badge.local { background: #8b5cf6; color: #fff; }
        .badge.kokoro { background: #ec4899; color: #fff; }
        .badge.cached { background: #22c55e; color: #000; }
        .badge.error { background: #ef4444; color: #fff; }
        .badge.success { background: #22c55e; color: #000; }
        .badge.sensitive { background: #ef4444; color: #fff; }
        .badge.not-sensitive { background: #374151; color: #9ca3af; }
        .badge.precision { background: #8b5cf6; color: #fff; }
        .badge.usecase { background: #f59e0b; color: #000; }
        .badge.images { background: #06b6d4; color: #000; }
        .badge.replay { background: #ec4899; color: #fff; }
        .badge.type-llm { background: #6366f1; color: #fff; }
        .badge.type-tts { background: #ec4899; color: #fff; }
        .badge.type-stt { background: #14b8a6; color: #000; }

        .type-filter {
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #374151;
            background: #1a1a2e;
            color: #888;
            cursor: pointer;
            font-size: 13px;
        }
        .type-filter:hover { border-color: #6366f1; color: #fff; }
        .type-filter.active { background: #6366f1; color: #fff; border-color: #6366f1; }

        .routes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
        }
        .route-card {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 16px;
            border-left: 4px solid #6366f1;
        }
        .route-card.unavailable { border-left-color: #ef4444; opacity: 0.6; }
        .route-card.text { border-left-color: #6366f1; }
        .route-card.vision { border-left-color: #06b6d4; }
        .route-card.tts { border-left-color: #ec4899; }
        .route-card.stt { border-left-color: #22c55e; }
        .route-card.overridden { border-left-color: #ef4444 !important; background: rgba(239, 68, 68, 0.1); }
        .route-card.overridden .route-target { color: #ef4444; }
        .route-card.editable { cursor: pointer; transition: transform 0.1s, box-shadow 0.1s; }
        .route-card.editable:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .route-flags { font-size: 13px; color: #888; margin-bottom: 8px; }
        .route-target { font-size: 15px; font-weight: 600; }
        .route-type { font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }

        /* STT History Cards */
        .stt-card {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .stt-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #2d2d44;
        }
        .stt-card-meta {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            font-size: 13px;
        }
        .stt-card-meta span { color: #888; }
        .stt-card-meta strong { color: #a5b4fc; }
        .stt-transcription {
            background: #252540;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 150px;
            overflow-y: auto;
        }
        .stt-transcription:empty::before {
            content: '(No transcription available)';
            color: #666;
            font-style: italic;
        }

        .refresh-btn {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #6366f1;
            color: #fff;
            border: none;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
            z-index: 100;
        }
        .refresh-btn:hover { background: #4f46e5; }

        @keyframes spin { 100% { transform: rotate(360deg); } }
        .spinning { animation: spin 1s linear infinite; }

        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            overflow-y: auto;
            padding: 40px 20px;
        }
        .modal-overlay.active { display: block; }
        .modal {
            background: #1a1a2e;
            border-radius: 16px;
            max-width: 900px;
            margin: 0 auto;
            position: relative;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 24px;
            border-bottom: 1px solid #2d2d44;
        }
        .modal-header h2 { color: #a5b4fc; font-size: 20px; }
        .modal-close {
            background: none;
            border: none;
            color: #888;
            font-size: 28px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        .modal-close:hover { color: #fff; }
        .modal-body { padding: 24px; }
        .modal-section { margin-bottom: 24px; }
        .modal-section:last-child { margin-bottom: 0; }
        .modal-section h3 {
            color: #6366f1;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }
        .detail-item { }
        .detail-label { color: #888; font-size: 12px; margin-bottom: 4px; }
        .detail-value { color: #e0e0e0; font-size: 14px; font-weight: 500; }

        .code-block {
            background: #252540;
            border-radius: 8px;
            padding: 16px;
            font-family: 'SF Mono', Monaco, 'Consolas', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }
        .message-block {
            background: #252540;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }
        .message-block:last-child { margin-bottom: 0; }
        .message-role {
            color: #6366f1;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .message-content {
            color: #e0e0e0;
            font-size: 14px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .tokens-small { font-size: 11px; color: #888; }

        /* Image preview styles */
        .message-text { margin-bottom: 8px; }
        .message-text:last-child { margin-bottom: 0; }
        .message-image-container {
            margin: 12px 0;
            text-align: center;
        }
        .message-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            border: 1px solid #3d3d5c;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .message-image:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }
        .image-label {
            color: #888;
            font-size: 11px;
            margin-top: 4px;
        }
        .image-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.95);
            z-index: 2000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        .full-image {
            max-width: 95vw;
            max-height: 90vh;
            border-radius: 8px;
        }
        .image-hint {
            color: #888;
            font-size: 12px;
            margin-top: 16px;
        }

        /* Pending requests styles */
        #pending-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 16px;
        }
        #pending-section .section-title {
            color: #f59e0b;
        }
        .pending-count {
            background: #f59e0b;
            color: #000;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 8px;
        }
        #pending-table tbody tr {
            background: rgba(245, 158, 11, 0.1);
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { background: rgba(245, 158, 11, 0.1); }
            50% { background: rgba(245, 158, 11, 0.2); }
        }
        .elapsed-time {
            font-family: 'SF Mono', Monaco, 'Consolas', monospace;
            color: #f59e0b;
        }
        .preview-text {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 12px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Proxy</h1>
        <p class="subtitle">Unified AI gateway with routing, caching, and cost tracking</p>

        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('llm')">Requests</button>
            <button class="nav-tab" onclick="window.location.href='/test'">Playground</button>
            <button class="nav-tab" onclick="switchTab('routing')">Routing</button>
            <button class="nav-tab" onclick="window.location.href='/analytics'">Analytics</button>
        </div>

        <!-- LLM Tab -->
        <div id="tab-llm" class="tab-content active">
            <div class="stats-grid" id="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Requests</div>
                    <div class="stat-value" id="total-requests">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Cache Hit Rate</div>
                    <div class="stat-value" id="cache-rate">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Cost</div>
                    <div class="stat-value cost" id="total-cost">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Today's Cost</div>
                    <div class="stat-value cost" id="today-cost">-</div>
                </div>
            </div>

            <div class="section">
                <h2 class="section-title">By Provider</h2>
                <table id="provider-table">
                    <thead><tr><th>Provider</th><th>Requests</th><th>Avg Latency</th><th>Cost</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="section" id="pending-section" style="display:none">
                <h2 class="section-title">Pending Requests <span class="pending-count" id="pending-count"></span></h2>
                <table id="pending-table">
                    <thead><tr><th>Started</th><th>Provider</th><th>Model</th><th>Sensitive</th><th>Precision</th><th>Usecase</th><th>Elapsed</th><th>Preview</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="section">
                <h2 class="section-title">Recent Requests <span style="font-size:12px;color:#888;font-weight:normal">(click row for details)</span></h2>
                <div id="filter-bar" style="display: none; background: #2d2d44; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; color: #a5b4fc; font-size: 14px;"></div>
                <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center;">
                    <button class="type-filter active" data-type="" onclick="filterByType('')">All</button>
                    <button class="type-filter" data-type="llm" onclick="filterByType('llm')">LLM</button>
                    <button class="type-filter" data-type="tts" onclick="filterByType('tts')">TTS</button>
                    <button class="type-filter" data-type="stt" onclick="filterByType('stt')">STT</button>
                    <span style="margin-left: 16px; color: #888;">Usecase:</span>
                    <select id="usecase-filter" onchange="filterByUsecase(this.value)" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #374151; background: #1a1a2e; color: #888; cursor: pointer; font-size: 13px;">
                        <option value="">All</option>
                    </select>
                </div>
                <table id="recent-table">
                    <thead><tr><th>Time</th><th>Type</th><th>Provider</th><th>Model</th><th>Sensitive</th><th>Precision</th><th>Usecase</th><th>Info</th><th>Latency</th><th>Cost</th><th>Status</th></tr></thead>
                    <tbody></tbody>
                </table>
                <div id="pagination-controls" style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding: 12px 16px; background: #1e1e2e; border-radius: 8px;">
                    <button id="prev-page" onclick="goToPrevPage()" style="padding: 8px 16px; border-radius: 6px; border: 1px solid #374151; background: #2d2d44; color: #a5b4fc; cursor: pointer; font-size: 13px;" disabled>&larr; Previous</button>
                    <span id="page-info" style="color: #888; font-size: 13px;">Page 1 of 1 (0 total)</span>
                    <button id="next-page" onclick="goToNextPage()" style="padding: 8px 16px; border-radius: 6px; border: 1px solid #374151; background: #2d2d44; color: #a5b4fc; cursor: pointer; font-size: 13px;" disabled>Next &rarr;</button>
                </div>
            </div>
        </div>

        <!-- Routing Tab -->
        <div id="tab-routing" class="tab-content">
            <div class="section">
                <h2 class="section-title">Usecase Routing Configuration</h2>
                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px;">
                    <label style="color: #a5b4fc;">Usecase:</label>
                    <select id="usecase-select" onchange="loadRoutes()" style="padding: 8px 12px; border-radius: 8px; background: #1e1e2e; border: 1px solid #374151; color: #fff; min-width: 200px;">
                        <option value="">(base config - no overrides)</option>
                    </select>
                    <input type="text" id="new-usecase-input" placeholder="New usecase name..." style="padding: 8px 12px; border-radius: 8px; background: #1e1e2e; border: 1px solid #374151; color: #fff;">
                    <button onclick="addNewUsecase()" style="padding: 8px 16px; border-radius: 8px; background: #6366f1; border: none; color: #fff; cursor: pointer;">Add</button>
                </div>
                <p style="color: #888; font-size: 13px; margin-bottom: 20px;">Click on a route card to change its model. Overridden routes are shown in <span style="color: #ef4444;">red</span>.</p>
            </div>
            <div class="section">
                <h2 class="section-title">Text Routing</h2>
                <div class="routes-grid" id="text-routes-grid"></div>
            </div>
            <div class="section">
                <h2 class="section-title">Vision Routing</h2>
                <div class="routes-grid" id="vision-routes-grid"></div>
            </div>
            <div class="section">
                <h2 class="section-title">TTS Routing</h2>
                <div class="routes-grid" id="tts-routes-grid"></div>
            </div>
            <div class="section">
                <h2 class="section-title">STT Routing</h2>
                <div class="routes-grid" id="stt-routes-grid"></div>
            </div>
            <div class="section" id="usecase-distribution-section" style="display: none;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h2 class="section-title" style="margin: 0;">Traffic Distribution</h2>
                    <div style="display: flex; gap: 8px;">
                        <button id="range-24h" onclick="setTimeRange('24h')" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #374151; background: #1e1e2e; color: #888; cursor: pointer; font-size: 12px;">24h</button>
                        <button id="range-7d" onclick="setTimeRange('7d')" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #374151; background: #1e1e2e; color: #888; cursor: pointer; font-size: 12px;">7d</button>
                        <button id="range-30d" onclick="setTimeRange('30d')" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #6366f1; background: #6366f1; color: #fff; cursor: pointer; font-size: 12px;">30d</button>
                    </div>
                </div>
                <p style="color: #888; font-size: 13px; margin-bottom: 12px;">Shows which sensitivity/precision combinations are used - helps identify what routes matter most</p>
                <div id="usecase-distribution-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px;"></div>
                <div id="usecase-distribution-empty" style="display: none; color: #888; font-style: italic; padding: 20px; text-align: center;">No requests in the selected time range</div>
            </div>
            <div class="section">
                <h2 class="section-title">Model Traffic Volume</h2>
                <p style="color: #888; font-size: 13px; margin-bottom: 12px;">Request counts by model - helps identify high-traffic routes to optimize</p>
                <div id="model-volume-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px;"></div>
            </div>
        </div>

        <!-- Route Edit Modal -->
        <div class="modal-overlay" id="route-modal-overlay" onclick="closeRouteModal(event)" style="display:none">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 500px;">
                <div class="modal-header">
                    <h2>Edit Route</h2>
                    <button class="modal-close" onclick="closeRouteModal()">&times;</button>
                </div>
                <div class="modal-body" id="route-modal-body">
                    <div style="margin-bottom: 16px;">
                        <div style="color: #888; font-size: 13px;">Route</div>
                        <div id="route-edit-info" style="color: #fff; font-size: 15px;"></div>
                    </div>
                    <div style="margin-bottom: 16px;">
                        <div style="color: #888; font-size: 13px; margin-bottom: 4px;">Base Model</div>
                        <div id="route-base-model" style="color: #6366f1; font-size: 14px;"></div>
                    </div>
                    <div id="route-traffic-info" style="margin-bottom: 16px; padding: 10px; background: #1e1e2e; border-radius: 8px; display: none;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 18px;">⚠️</span>
                            <div>
                                <div style="color: #f59e0b; font-size: 13px; font-weight: 600;">Route Traffic (24h)</div>
                                <div id="route-traffic-count" style="color: #fff; font-size: 14px;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-bottom: 16px;">
                        <label style="color: #888; font-size: 13px; display: block; margin-bottom: 4px;">Provider</label>
                        <select id="route-edit-provider" onchange="updateModelOptions()" style="width: 100%; padding: 10px; border-radius: 8px; background: #1e1e2e; border: 1px solid #374151; color: #fff;">
                            <option value="anthropic">anthropic</option>
                            <option value="openai">openai</option>
                            <option value="ollama">ollama</option>
                        </select>
                    </div>
                    <div style="margin-bottom: 16px;">
                        <label style="color: #888; font-size: 13px; display: block; margin-bottom: 4px;">Model</label>
                        <select id="route-edit-model" style="width: 100%; padding: 10px; border-radius: 8px; background: #1e1e2e; border: 1px solid #374151; color: #fff;">
                        </select>
                    </div>
                    <div style="display: flex; gap: 12px; margin-top: 20px;">
                        <button onclick="saveRouteOverride()" style="flex: 1; padding: 12px; border-radius: 8px; background: #22c55e; border: none; color: #000; font-weight: 600; cursor: pointer;">Save Override</button>
                        <button onclick="resetRouteToBase()" id="reset-route-btn" style="flex: 1; padding: 12px; border-radius: 8px; background: #ef4444; border: none; color: #fff; font-weight: 600; cursor: pointer; display: none;">Reset to Base</button>
                        <button onclick="closeRouteModal()" style="flex: 1; padding: 12px; border-radius: 8px; background: #374151; border: none; color: #fff; font-weight: 600; cursor: pointer;">Cancel</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <button class="refresh-btn" id="refresh-btn" onclick="refresh()">&#8635;</button>

    <!-- Request Detail Modal -->
    <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2>Request Details</h2>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                <div style="text-align:center;color:#888;padding:40px;">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let currentTypeFilter = '';
        let currentFilters = {}; // For matrix filtering

        // Parse matrix filters from URL
        function parseMatrixFilters() {
            const urlParams = new URLSearchParams(window.location.search);
            const matrixParam = urlParams.get('matrix');
            console.log('Matrix param:', matrixParam);
            if (matrixParam) {
                const filterParams = new URLSearchParams(matrixParam);
                currentFilters = {
                    model: filterParams.get('model') || '',
                    sensitive: filterParams.get('sensitive') || '',
                    precision: filterParams.get('precision') || ''
                };
                console.log('Parsed filters:', currentFilters);
            }
        }
        parseMatrixFilters();

        function clearFilters() {
            currentFilters = {};
            currentPage = 1; // Reset to first page on filter clear
            window.history.replaceState({}, '', '/');
            updateFilterDisplay();
            loadStats();
        }

        function updateFilterDisplay() {
            const filterBar = document.getElementById('filter-bar');
            if (!filterBar) return;

            if (currentFilters.model || currentFilters.sensitive || currentFilters.precision) {
                let filterText = 'Filtering: ';
                if (currentFilters.model) filterText += 'model=' + currentFilters.model + ' ';
                if (currentFilters.sensitive) filterText += 'sensitive=' + (currentFilters.sensitive === '1' ? 'true' : 'false') + ' ';
                if (currentFilters.precision) filterText += 'precision=' + currentFilters.precision;
                filterBar.innerHTML = filterText + ' <a href="#" onclick="clearFilters(); return false;" style="margin-left: 12px;">Clear filters</a>';
                filterBar.style.display = 'block';
            } else {
                filterBar.style.display = 'none';
            }
        }

        async function loadStats() {
            let url = '/api/stats';
            const params = new URLSearchParams();
            if (currentFilters.model) params.set('model', currentFilters.model);
            if (currentFilters.sensitive) params.set('sensitive', currentFilters.sensitive);
            if (currentFilters.precision) params.set('precision', currentFilters.precision);
            if (params.toString()) url += '?' + params.toString();
            console.log('loadStats URL:', url, 'currentFilters:', currentFilters);

            const resp = await fetch(url);
            const stats = await resp.json();
            console.log('Stats response filters:', stats.filters, 'recent count:', stats.recent_requests?.length);

            updateFilterDisplay();

            document.getElementById('total-requests').textContent = stats.total_requests.toLocaleString();
            document.getElementById('cache-rate').textContent = (stats.cache_hit_rate * 100).toFixed(1) + '%';
            document.getElementById('total-cost').textContent = '$' + stats.total_cost_usd.toFixed(4);
            document.getElementById('today-cost').textContent = '$' + (stats.today?.cost_usd || 0).toFixed(4);

            // Provider table
            const providerBody = document.querySelector('#provider-table tbody');
            providerBody.innerHTML = '';
            for (const [provider, data] of Object.entries(stats.by_provider || {})) {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td><span class="badge ' + provider + '">' + provider + '</span></td>' +
                    '<td>' + data.count + '</td>' +
                    '<td>' + Math.round(data.avg_latency_ms) + 'ms</td>' +
                    '<td>$' + data.cost_usd.toFixed(4) + '</td>';
                providerBody.appendChild(tr);
            }

            // Load recent requests via history API
            await loadRecentRequests();
        }

        async function loadRecentRequests() {
            let url = '/api/history?limit=50&page=' + currentPage;
            if (currentTypeFilter) {
                url += '&type=' + encodeURIComponent(currentTypeFilter);
            }
            if (currentUsecaseFilter) {
                url += '&usecase=' + encodeURIComponent(currentUsecaseFilter);
            }
            // Add matrix filters
            if (currentFilters.model) {
                url += '&model=' + encodeURIComponent(currentFilters.model);
            }
            if (currentFilters.sensitive) {
                url += '&sensitive=' + encodeURIComponent(currentFilters.sensitive);
            }
            if (currentFilters.precision) {
                url += '&precision=' + encodeURIComponent(currentFilters.precision);
            }
            const resp = await fetch(url);
            const data = await resp.json();

            // Update pagination state
            totalRequests = data.total;
            totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
            currentPage = data.page;
            updatePaginationControls();

            const recentBody = document.querySelector('#recent-table tbody');
            recentBody.innerHTML = '';
            for (const req of data.requests || []) {
                const tr = document.createElement('tr');
                tr.className = 'clickable';
                tr.onclick = () => showRequestDetail(req.id);
                const time = new Date(req.timestamp).toLocaleTimeString();
                const status = req.cached ? '<span class="badge cached">cached</span>' :
                    (req.success ? '<span class="badge success">ok</span>' : '<span class="badge error">error</span>');
                const sensitiveClass = req.sensitive ? 'sensitive' : 'not-sensitive';
                const sensitiveText = req.sensitive ? 'YES' : 'no';
                const precision = req.precision || '-';
                const usecase = req.usecase ? '<span class="badge usecase">' + req.usecase + '</span>' : '-';
                const hasImages = req.has_images ? '<span class="badge images">img</span>' : '';
                const replayBadge = req.is_replay ? '<span class="badge replay">replay</span>' : '';
                const reqType = req.request_type || 'llm';
                const typeBadge = '<span class="badge type-' + reqType + '">' + reqType.toUpperCase() + '</span>';

                // Info column: tokens for LLM, voice for TTS, duration for STT
                let info = '';
                if (reqType === 'llm') {
                    const tokens = (req.input_tokens || 0) + (req.output_tokens || 0);
                    info = '<span class="tokens-small">' + tokens + ' tok</span>';
                } else if (reqType === 'tts') {
                    info = req.voice || '-';
                } else if (reqType === 'stt') {
                    info = req.audio_duration_ms ? (req.audio_duration_ms / 1000).toFixed(1) + 's' : '-';
                }

                tr.innerHTML = '<td>' + time + '</td>' +
                    '<td>' + typeBadge + ' ' + replayBadge + '</td>' +
                    '<td><span class="badge ' + req.provider + '">' + req.provider + '</span></td>' +
                    '<td>' + req.model + ' ' + hasImages + '</td>' +
                    '<td><span class="badge ' + sensitiveClass + '">' + sensitiveText + '</span></td>' +
                    '<td><span class="badge precision">' + precision + '</span></td>' +
                    '<td>' + usecase + '</td>' +
                    '<td>' + info + '</td>' +
                    '<td>' + req.latency_ms + 'ms</td>' +
                    '<td>$' + req.cost_usd.toFixed(6) + '</td>' +
                    '<td>' + status + '</td>';
                recentBody.appendChild(tr);
            }
        }

        function filterByType(type) {
            currentTypeFilter = type;
            currentPage = 1; // Reset to first page on filter change
            // Update button states
            document.querySelectorAll('.type-filter').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.type === type);
            });
            loadRecentRequests();
        }

        let currentUsecaseFilter = '';
        let currentPage = 1;
        let totalPages = 1;
        let totalRequests = 0;

        function filterByUsecase(usecase) {
            currentUsecaseFilter = usecase;
            currentPage = 1; // Reset to first page on filter change
            loadRecentRequests();
        }

        function goToPrevPage() {
            if (currentPage > 1) {
                currentPage--;
                loadRecentRequests();
            }
        }

        function goToNextPage() {
            if (currentPage < totalPages) {
                currentPage++;
                loadRecentRequests();
            }
        }

        function updatePaginationControls() {
            const prevBtn = document.getElementById('prev-page');
            const nextBtn = document.getElementById('next-page');
            const pageInfo = document.getElementById('page-info');

            prevBtn.disabled = currentPage <= 1;
            nextBtn.disabled = currentPage >= totalPages;

            prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
            nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';

            pageInfo.textContent = 'Page ' + currentPage + ' of ' + totalPages + ' (' + totalRequests + ' total)';
        }

        async function loadUsecaseFilter() {
            const resp = await fetch('/api/usecases/history');
            const usecases = await resp.json();
            const select = document.getElementById('usecase-filter');
            const currentValue = select.value;

            // Keep the first option (All)
            select.innerHTML = '<option value="">All</option>';
            for (const usecase of usecases || []) {
                const opt = document.createElement('option');
                opt.value = usecase;
                opt.textContent = usecase;
                select.appendChild(opt);
            }

            // Restore selection if it still exists
            if (currentValue && usecases && usecases.includes(currentValue)) {
                select.value = currentValue;
            }
        }

        let availableModels = {};
        let currentEditRoute = null;

        async function loadUsecases() {
            // Fetch all usecases from history (API calls made)
            const resp = await fetch('/api/usecases/history');
            const usecases = await resp.json();
            const select = document.getElementById('usecase-select');
            const currentValue = select.value;

            // Keep the first option (base config)
            select.innerHTML = '<option value="">(base config - no overrides)</option>';
            for (const usecase of usecases || []) {
                const opt = document.createElement('option');
                opt.value = usecase;
                opt.textContent = usecase;
                select.appendChild(opt);
            }

            // Restore selection if it still exists
            if (currentValue && usecases && usecases.includes(currentValue)) {
                select.value = currentValue;
            }
        }

        async function loadAvailableModels() {
            const resp = await fetch('/api/models');
            availableModels = await resp.json();
        }

        function addNewUsecase() {
            const input = document.getElementById('new-usecase-input');
            const name = input.value.trim();
            if (!name) return;

            const select = document.getElementById('usecase-select');
            // Check if already exists
            for (const opt of select.options) {
                if (opt.value === name) {
                    select.value = name;
                    input.value = '';
                    loadRoutes();
                    return;
                }
            }

            // Add new option
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
            select.value = name;
            input.value = '';
            loadRoutes();
        }

        async function loadRoutes() {
            const usecase = document.getElementById('usecase-select').value;
            const url = usecase ? '/api/routes?usecase=' + encodeURIComponent(usecase) : '/api/routes';
            const resp = await fetch(url);
            const routes = await resp.json();

            const textGrid = document.getElementById('text-routes-grid');
            const visionGrid = document.getElementById('vision-routes-grid');
            const ttsGrid = document.getElementById('tts-routes-grid');
            const sttGrid = document.getElementById('stt-routes-grid');
            textGrid.innerHTML = '';
            visionGrid.innerHTML = '';
            ttsGrid.innerHTML = '';
            sttGrid.innerHTML = '';

            const typeColors = {
                'text': '#6366f1',
                'vision': '#06b6d4',
                'tts': '#ec4899',
                'stt': '#22c55e'
            };

            for (const route of routes) {
                const card = document.createElement('div');
                const isEditable = (route.type === 'text' || route.type === 'vision') && route.available && usecase;
                const isOverridden = route.overridden;

                let classes = 'route-card ' + route.type;
                if (!route.available) classes += ' unavailable';
                if (isOverridden) classes += ' overridden';
                if (isEditable) classes += ' editable';
                card.className = classes;

                const flags = route.precision
                    ? 'sensitive: ' + route.sensitive + ', precision: ' + route.precision
                    : 'sensitive: ' + route.sensitive;

                let targetText = route.available ? route.provider + ' / ' + route.model : 'Not Available';
                if (isOverridden) {
                    targetText += ' <span style="font-size:11px;opacity:0.7">(override)</span>';
                }

                card.innerHTML = '<div class="route-type" style="color:' + (typeColors[route.type] || '#6366f1') + '">' + route.type + '</div>' +
                    '<div class="route-flags">' + flags + '</div>' +
                    '<div class="route-target">' + targetText + '</div>';

                if (isEditable) {
                    card.onclick = () => openRouteEditModal(route, usecase);
                }

                if (route.type === 'vision') {
                    visionGrid.appendChild(card);
                } else if (route.type === 'tts') {
                    ttsGrid.appendChild(card);
                } else if (route.type === 'stt') {
                    sttGrid.appendChild(card);
                } else {
                    textGrid.appendChild(card);
                }
            }

            // Load distribution and model volumes for selected usecase
            loadUsecaseDistribution(usecase);
            loadModelVolumes(usecase);
        }

        let currentTimeRange = '30d';
        let lastDistributionData = null;

        function setTimeRange(range) {
            currentTimeRange = range;
            // Update button styles
            ['24h', '7d', '30d'].forEach(r => {
                const btn = document.getElementById('range-' + r);
                if (r === range) {
                    btn.style.background = '#6366f1';
                    btn.style.borderColor = '#6366f1';
                    btn.style.color = '#fff';
                } else {
                    btn.style.background = '#1e1e2e';
                    btn.style.borderColor = '#374151';
                    btn.style.color = '#888';
                }
            });
            // Reload data with new range
            const usecase = document.getElementById('usecase-select').value;
            loadUsecaseDistribution(usecase);
            loadModelVolumes(usecase);
        }

        async function loadUsecaseDistribution(usecase) {
            const section = document.getElementById('usecase-distribution-section');
            const grid = document.getElementById('usecase-distribution-grid');
            const emptyMsg = document.getElementById('usecase-distribution-empty');

            section.style.display = 'block';
            grid.innerHTML = '';
            emptyMsg.style.display = 'none';

            let url = '/api/usecases/distribution?range=' + currentTimeRange;
            if (usecase) {
                url += '&usecase=' + encodeURIComponent(usecase);
            }
            const resp = await fetch(url);
            const data = await resp.json();
            lastDistributionData = data;

            // Highlight route cards that have traffic
            highlightActiveRoutes(data.distribution || []);

            if (!data.distribution || data.distribution.length === 0) {
                emptyMsg.style.display = 'block';
                return;
            }

            const typeColors = {
                'text': '#6366f1',
                'vision': '#06b6d4'
            };

            const maxCount = Math.max(...data.distribution.map(d => d.count));

            for (const d of data.distribution) {
                const card = document.createElement('div');
                card.style.cssText = 'background: #1e1e2e; border: 1px solid #374151; border-radius: 8px; padding: 12px;';

                const barWidth = (d.count / maxCount) * 100;
                const color = typeColors[d.route_type] || '#6366f1';
                const pct = ((d.count / data.total) * 100).toFixed(1);

                card.innerHTML =
                    '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">' +
                        '<span style="font-size: 12px; color: ' + color + '; font-weight: 600;">' + d.route_type.toUpperCase() + '</span>' +
                        '<span style="font-size: 14px; font-weight: 600; color: #fff;">' + d.count + ' <span style="font-size: 11px; color: #888;">(' + pct + '%)</span></span>' +
                    '</div>' +
                    '<div style="font-size: 11px; color: #888; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="' + d.model + '">' + d.model + '</div>' +
                    '<div style="font-size: 12px; color: #a5b4fc; margin-bottom: 8px;">' +
                        'sensitive: ' + d.sensitive + ', precision: ' + d.precision +
                    '</div>' +
                    '<div style="background: #374151; border-radius: 4px; height: 6px; overflow: hidden;">' +
                        '<div style="width: ' + barWidth + '%; height: 100%; background: ' + color + '; border-radius: 4px;"></div>' +
                    '</div>';

                grid.appendChild(card);
            }

            // Add total summary
            const summary = document.createElement('div');
            summary.style.cssText = 'background: #1e1e2e; border: 1px solid #22c55e; border-radius: 8px; padding: 12px; display: flex; align-items: center; justify-content: center;';
            summary.innerHTML = '<span style="font-size: 14px; color: #22c55e; font-weight: 600;">Total: ' + data.total + ' requests (' + currentTimeRange + ')</span>';
            grid.appendChild(summary);
        }

        function highlightActiveRoutes(distribution) {
            // Build a set of active route keys
            const activeRoutes = new Set();
            for (const d of distribution) {
                activeRoutes.add(d.route_type + '|' + d.sensitive + '|' + d.precision);
            }

            // Find all route cards and highlight those with traffic
            document.querySelectorAll('.route-card').forEach(card => {
                const flagsText = card.querySelector('.route-flags')?.textContent || '';
                const typeText = card.querySelector('.route-type')?.textContent || '';

                // Parse sensitive and precision from flags
                const sensitiveMatch = flagsText.match(/sensitive:\s*(true|false)/);
                const precisionMatch = flagsText.match(/precision:\s*(\w+)/);

                if (sensitiveMatch && precisionMatch) {
                    const key = typeText + '|' + sensitiveMatch[1] + '|' + precisionMatch[1];
                    if (activeRoutes.has(key)) {
                        card.style.boxShadow = '0 0 12px rgba(245, 158, 11, 0.6), inset 0 0 20px rgba(245, 158, 11, 0.1)';
                        card.style.borderColor = '#f59e0b';
                    } else {
                        card.style.boxShadow = '';
                        card.style.borderColor = '';
                    }
                }
            });
        }

        async function openRouteEditModal(route, usecase) {
            currentEditRoute = { ...route, usecase };

            document.getElementById('route-edit-info').textContent =
                route.type.toUpperCase() + ' | sensitive=' + route.sensitive + ' | precision=' + route.precision;
            document.getElementById('route-base-model').textContent =
                (route.base_provider || route.provider) + ' / ' + (route.base_model || route.model);

            // Set current values
            document.getElementById('route-edit-provider').value = route.provider;
            updateModelOptions();
            document.getElementById('route-edit-model').value = route.model;

            // Show reset button if overridden
            document.getElementById('reset-route-btn').style.display = route.overridden ? 'block' : 'none';

            // Load traffic count for this specific route
            const trafficInfo = document.getElementById('route-traffic-info');
            const trafficCount = document.getElementById('route-traffic-count');
            trafficInfo.style.display = 'none';

            if (usecase) {
                const resp = await fetch('/api/usecases/distribution?usecase=' + encodeURIComponent(usecase));
                const data = await resp.json();

                // Find the count for this specific route
                const match = (data.distribution || []).find(d =>
                    d.sensitive === route.sensitive &&
                    d.precision === route.precision &&
                    d.route_type === route.type
                );

                if (match && match.count > 0) {
                    trafficInfo.style.display = 'block';
                    const riskLevel = match.count > 100 ? 'high-traffic route!' : match.count > 10 ? 'moderate traffic' : 'low traffic';
                    trafficCount.innerHTML = '<strong>' + match.count + '</strong> requests - ' + riskLevel;
                }
            }

            document.getElementById('route-modal-overlay').style.display = 'flex';
        }

        function closeRouteModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('route-modal-overlay').style.display = 'none';
            currentEditRoute = null;
        }

        function updateModelOptions() {
            const provider = document.getElementById('route-edit-provider').value;
            const modelSelect = document.getElementById('route-edit-model');
            const currentModel = modelSelect.value;

            modelSelect.innerHTML = '';
            const models = availableModels[provider] || [];
            for (const model of models) {
                const opt = document.createElement('option');
                opt.value = model;
                opt.textContent = model;
                modelSelect.appendChild(opt);
            }

            // Try to keep current selection
            if (models.includes(currentModel)) {
                modelSelect.value = currentModel;
            }
        }

        async function saveRouteOverride() {
            if (!currentEditRoute) return;

            const provider = document.getElementById('route-edit-provider').value;
            const model = document.getElementById('route-edit-model').value;

            const resp = await fetch('/api/routes/override', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    usecase: currentEditRoute.usecase,
                    type: currentEditRoute.type,
                    sensitive: currentEditRoute.sensitive,
                    precision: currentEditRoute.precision,
                    provider: provider,
                    model: model
                })
            });

            if (resp.ok) {
                closeRouteModal();
                loadRoutes();
                loadUsecases();
            } else {
                const err = await resp.text();
                alert('Failed to save: ' + err);
            }
        }

        async function resetRouteToBase() {
            if (!currentEditRoute) return;

            const resp = await fetch('/api/routes/override?usecase=' + encodeURIComponent(currentEditRoute.usecase) +
                '&type=' + currentEditRoute.type +
                '&sensitive=' + currentEditRoute.sensitive +
                '&precision=' + currentEditRoute.precision, {
                method: 'DELETE'
            });

            if (resp.ok) {
                closeRouteModal();
                loadRoutes();
            } else {
                const err = await resp.text();
                alert('Failed to reset: ' + err);
            }
        }

        // Tab switching
        let currentTab = 'llm';
        function switchTab(tabName) {
            currentTab = tabName;
            // Update tab buttons
            document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.nav-tab[onclick*="' + tabName + '"]')?.classList.add('active');
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById('tab-' + tabName)?.classList.add('active');
            // Load data for tab
            if (tabName === 'routing') {
                loadUsecases();
                loadAvailableModels();
                loadRoutes();
                loadModelVolumes();
            }
        }

        async function loadModelVolumes(usecase) {
            let url = '/api/stats/models?range=' + currentTimeRange;
            if (usecase) {
                url += '&usecase=' + encodeURIComponent(usecase);
            }
            const resp = await fetch(url);
            const models = await resp.json();
            const grid = document.getElementById('model-volume-grid');
            grid.innerHTML = '';

            if (!models || models.length === 0) {
                grid.innerHTML = '<div style="color: #888; font-style: italic;">No traffic data yet</div>';
                return;
            }

            // Find max count for bar scaling
            const maxCount = Math.max(...models.map(m => m.count));

            for (const m of models) {
                const card = document.createElement('div');
                card.style.cssText = 'background: #1e1e2e; border: 1px solid #374151; border-radius: 8px; padding: 12px;';

                const barWidth = (m.count / maxCount) * 100;
                const costStr = m.cost_usd > 0 ? '$' + m.cost_usd.toFixed(4) : '-';

                card.innerHTML =
                    '<div style="font-size: 13px; color: #a5b4fc; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="' + m.model + '">' + m.model + '</div>' +
                    '<div style="display: flex; align-items: center; gap: 8px;">' +
                        '<div style="flex: 1; background: #374151; border-radius: 4px; height: 8px; overflow: hidden;">' +
                            '<div style="width: ' + barWidth + '%; height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 4px;"></div>' +
                        '</div>' +
                        '<div style="font-size: 14px; font-weight: 600; color: #fff; min-width: 50px; text-align: right;">' + m.count.toLocaleString() + '</div>' +
                    '</div>' +
                    '<div style="font-size: 11px; color: #888; margin-top: 4px;">' + Math.round(m.avg_latency_ms) + 'ms avg | ' + costStr + '</div>';

                grid.appendChild(card);
            }
        }

        async function showRequestDetail(id) {
            const overlay = document.getElementById('modal-overlay');
            const body = document.getElementById('modal-body');
            overlay.classList.add('active');
            body.innerHTML = '<div style="text-align:center;color:#888;padding:40px;">Loading...</div>';
            currentRequestId = id;

            try {
                const resp = await fetch('/api/request?id=' + id);
                const data = await resp.json();
                currentRequestData = data;
                renderRequestDetail(data);
            } catch (err) {
                body.innerHTML = '<div style="color:#ef4444;padding:40px;">Error loading request details: ' + err.message + '</div>';
            }
        }

        function renderRequestDetail(data) {
            const body = document.getElementById('modal-body');
            const time = new Date(data.timestamp).toLocaleString();
            const sensitiveClass = data.sensitive ? 'sensitive' : 'not-sensitive';
            const sensitiveText = data.sensitive ? 'YES (local only)' : 'No (cloud OK)';

            let html = '<div style="display:flex;justify-content:flex-end;margin-bottom:12px">';
            html += '<button onclick="copyRequestJson()" style="padding:8px 16px;background:#374151;border:none;border-radius:6px;color:#e2e8f0;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:6px"><span style="font-size:16px">📋</span> Copy Full JSON</button>';
            html += '</div>';
            html += '<div class="modal-section"><h3>Request Metadata</h3><div class="detail-grid">';
            html += '<div class="detail-item"><div class="detail-label">ID</div><div class="detail-value">#' + data.id + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Timestamp</div><div class="detail-value">' + time + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Provider</div><div class="detail-value"><span class="badge ' + data.provider + '">' + data.provider + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Model Used</div><div class="detail-value">' + data.model + '</div></div>';
            if (data.requested_model) {
                html += '<div class="detail-item"><div class="detail-label">Requested Model</div><div class="detail-value">' + data.requested_model + '</div></div>';
            }
            html += '<div class="detail-item"><div class="detail-label">Sensitive</div><div class="detail-value"><span class="badge ' + sensitiveClass + '">' + sensitiveText + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Precision</div><div class="detail-value"><span class="badge precision">' + (data.precision || '-') + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Usecase</div><div class="detail-value"><span class="badge usecase">' + (data.usecase || '-') + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Has Images</div><div class="detail-value">' + (data.has_images ? '<span class="badge images">Yes</span>' : 'No') + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Cached</div><div class="detail-value">' + (data.cached ? '<span class="badge cached">Yes</span>' : 'No') + '</div></div>';
            html += '</div></div>';

            html += '<div class="modal-section"><h3>Performance</h3><div class="detail-grid">';
            html += '<div class="detail-item"><div class="detail-label">Latency</div><div class="detail-value">' + data.latency_ms + ' ms</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Input Tokens</div><div class="detail-value">' + data.input_tokens + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Output Tokens</div><div class="detail-value">' + data.output_tokens + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Cost</div><div class="detail-value" style="color:#22c55e">$' + data.cost_usd.toFixed(6) + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Status</div><div class="detail-value">' + (data.success ? '<span class="badge success">Success</span>' : '<span class="badge error">Error</span>') + '</div></div>';
            if (data.error) {
                html += '<div class="detail-item" style="grid-column:1/-1"><div class="detail-label">Error</div><div class="detail-value" style="color:#ef4444">' + escapeHtml(data.error) + '</div></div>';
            }
            html += '</div></div>';

            // Request messages
            if (data.request && data.request.messages) {
                html += '<div class="modal-section"><h3>Request Messages</h3>';
                for (const msg of data.request.messages) {
                    html += '<div class="message-block">';
                    html += '<div class="message-role">' + msg.role + '</div>';
                    html += '<div class="message-content">' + formatMessageContent(msg.content) + '</div>';
                    html += '</div>';
                }
                html += '</div>';
            }

            // Response
            if (data.response) {
                let responseContent = '';
                if (data.response.choices && data.response.choices.length > 0) {
                    const choice = data.response.choices[0];
                    if (choice.message && choice.message.content) {
                        responseContent = choice.message.content;
                    }
                }

                // Check for thinking content wrapped in <think> tags
                const thinkMatch = responseContent.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/);
                if (thinkMatch) {
                    const thinkingContent = thinkMatch[1];
                    const actualResponse = thinkMatch[2];

                    // Show thinking section first
                    html += '<div class="modal-section"><h3>💭 Thinking</h3>';
                    html += '<div class="code-block" style="background:#1e1b4b;border-left:3px solid #8b5cf6;max-height:300px;overflow-y:auto">' + escapeHtml(thinkingContent) + '</div>';
                    html += '</div>';

                    // Show actual response
                    html += '<div class="modal-section"><h3>Response</h3>';
                    html += '<div class="code-block">' + escapeHtml(actualResponse) + '</div>';
                    html += '</div>';
                } else if (responseContent) {
                    html += '<div class="modal-section"><h3>Response</h3>';
                    html += '<div class="code-block">' + escapeHtml(responseContent) + '</div>';
                    html += '</div>';
                } else {
                    html += '<div class="modal-section"><h3>Response</h3>';
                    html += '<div class="code-block">' + escapeHtml(JSON.stringify(data.response, null, 2)) + '</div>';
                    html += '</div>';
                }
            }

            // Replay with different model section
            if (data.request && data.request.messages) {
                html += '<div class="modal-section"><h3>Replay with Different Model</h3>';
                html += '<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">';
                html += '<select id="replay-model-select" style="padding:8px 12px;background:#1e1b4b;border:1px solid #4338ca;border-radius:6px;color:#e2e8f0;font-size:14px;min-width:250px">';
                html += '<option value="">Select a model...</option>';
                for (const group of replayModelOptions) {
                    html += '<optgroup label="' + group.group + '">';
                    for (const model of group.models) {
                        const selected = model === data.model ? ' selected' : '';
                        html += '<option value="' + model + '"' + selected + '>' + model + '</option>';
                    }
                    html += '</optgroup>';
                }
                html += '</select>';
                html += '<button id="replay-btn" onclick="replayWithModel()" style="padding:8px 16px;background:#4f46e5;border:none;border-radius:6px;color:white;cursor:pointer;font-size:14px">Replay Request</button>';
                html += '</div>';
                html += '<div id="replay-result" style="margin-top:12px"></div>';
                html += '</div>';
            }

            // Cache key
            if (data.cache_key) {
                html += '<div class="modal-section"><h3>Cache</h3>';
                html += '<div class="detail-item"><div class="detail-label">Cache Key</div><div class="detail-value" style="font-family:monospace;font-size:12px;word-break:break-all">' + data.cache_key + '</div></div>';
                html += '</div>';
            }

            body.innerHTML = html;
        }

        function formatMessageContent(content) {
            if (typeof content === 'string') {
                return escapeHtml(content);
            }
            if (Array.isArray(content)) {
                let result = '';
                for (const part of content) {
                    if (part.type === 'text') {
                        result += '<div class="message-text">' + escapeHtml(part.text || '') + '</div>';
                    } else if (part.type === 'image_url') {
                        const url = part.image_url?.url || '';
                        if (url) {
                            result += '<div class="message-image-container">';
                            result += '<img class="message-image" src="' + escapeHtml(url) + '" onclick="showFullImage(this.src)" title="Click to view full size">';
                            result += '<div class="image-label">Image attachment (click to enlarge)</div>';
                            result += '</div>';
                        } else {
                            result += '<em style="color:#06b6d4">[Image: no data]</em>';
                        }
                    } else {
                        result += escapeHtml(JSON.stringify(part));
                    }
                }
                return result;
            }
            return escapeHtml(JSON.stringify(content));
        }

        function showFullImage(src) {
            const overlay = document.createElement('div');
            overlay.className = 'image-overlay';
            overlay.onclick = () => overlay.remove();
            overlay.innerHTML = '<img src="' + src + '" class="full-image"><div class="image-hint">Click anywhere to close</div>';
            document.body.appendChild(overlay);
        }

        function escapeHtml(str) {
            if (typeof str !== 'string') return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        function closeModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('modal-overlay').classList.remove('active');
        }

        async function copyRequestJson() {
            if (!currentRequestData) return;
            try {
                await navigator.clipboard.writeText(JSON.stringify(currentRequestData, null, 2));
                // Visual feedback
                const btn = event.target.closest('button');
                const originalHtml = btn.innerHTML;
                btn.innerHTML = '<span style="font-size:16px">✓</span> Copied!';
                btn.style.background = '#22c55e';
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.background = '#374151';
                }, 1500);
            } catch (err) {
                alert('Failed to copy: ' + err.message);
            }
        }

        // Store current request ID and data for replay/copy
        let currentRequestId = null;
        let currentRequestData = null;

        // Available models for replay - fetched dynamically
        let replayModelOptions = [
            { group: 'Anthropic', models: ['claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'] },
            { group: 'OpenAI', models: ['gpt-5.1', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1', 'o1-mini'] },
            { group: 'Ollama (Local)', models: [] } // Will be populated dynamically
        ];

        // Fetch Ollama models dynamically
        fetch('/api/models').then(r => r.json()).then(data => {
            if (data.ollama) {
                replayModelOptions[2].models = data.ollama;
                // Refresh the selector if it's already populated
                const select = document.getElementById('replay-model-select');
                if (select && select.options.length > 1) {
                    populateReplayModelSelect();
                }
            }
        }).catch(e => console.error('Failed to load Ollama models:', e));

        async function replayWithModel() {
            const select = document.getElementById('replay-model-select');
            const model = select.value;
            if (!model || !currentRequestId) return;

            const btn = document.getElementById('replay-btn');
            const resultDiv = document.getElementById('replay-result');
            btn.disabled = true;
            btn.textContent = 'Replaying...';
            resultDiv.innerHTML = '<div style="color:#888;padding:10px;">Sending request to ' + model + '...</div>';

            try {
                const resp = await fetch('/api/replay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ request_id: currentRequestId, model: model })
                });

                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }

                const data = await resp.json();
                let content = '';
                if (data.choices && data.choices[0]?.message?.content) {
                    content = data.choices[0].message.content;
                } else {
                    content = JSON.stringify(data, null, 2);
                }

                resultDiv.innerHTML = '<div class="code-block" style="max-height:400px;overflow-y:auto">' + escapeHtml(content) + '</div>';
                resultDiv.innerHTML += '<div style="color:#22c55e;margin-top:8px;font-size:12px">Replayed successfully with ' + model + '</div>';

                // Refresh stats after replay
                loadStats();
            } catch (err) {
                resultDiv.innerHTML = '<div style="color:#ef4444;padding:10px;">Error: ' + escapeHtml(err.message) + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Replay Request';
            }
        }

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        async function loadPending() {
            try {
                const resp = await fetch('/api/pending');
                const pending = await resp.json();

                const section = document.getElementById('pending-section');
                const tbody = document.querySelector('#pending-table tbody');
                const countBadge = document.getElementById('pending-count');

                if (pending.length === 0) {
                    section.style.display = 'none';
                    return;
                }

                section.style.display = 'block';
                countBadge.textContent = pending.length;
                tbody.innerHTML = '';

                for (const req of pending) {
                    const tr = document.createElement('tr');
                    const startTime = new Date(req.start_time).toLocaleTimeString();
                    const sensitiveClass = req.sensitive ? 'sensitive' : 'not-sensitive';
                    const sensitiveText = req.sensitive ? 'YES' : 'no';
                    const hasImages = req.has_images ? '<span class="badge images">img</span>' : '';
                    const elapsed = formatElapsed(req.elapsed_ms);

                    tr.innerHTML = '<td>' + startTime + '</td>' +
                        '<td><span class="badge ' + req.provider + '">' + req.provider + '</span></td>' +
                        '<td>' + req.model + ' ' + hasImages + '</td>' +
                        '<td><span class="badge ' + sensitiveClass + '">' + sensitiveText + '</span></td>' +
                        '<td><span class="badge precision">' + (req.precision || '-') + '</span></td>' +
                        '<td><span class="badge usecase">' + (req.usecase || '-') + '</span></td>' +
                        '<td class="elapsed-time">' + elapsed + '</td>' +
                        '<td class="preview-text" title="' + escapeHtml(req.preview) + '">' + escapeHtml(req.preview) + '</td>';
                    tbody.appendChild(tr);
                }
            } catch (err) {
                console.error('Failed to load pending requests:', err);
            }
        }

        function formatElapsed(ms) {
            if (ms < 1000) return ms + 'ms';
            if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
            return Math.floor(ms / 60000) + 'm ' + Math.floor((ms % 60000) / 1000) + 's';
        }

        async function refresh() {
            const btn = document.getElementById('refresh-btn');
            btn.classList.add('spinning');
            // Load data based on current tab
            if (currentTab === 'llm') {
                await Promise.all([loadStats(), loadPending(), loadUsecaseFilter()]);
            } else if (currentTab === 'tts') {
                await loadTTSData();
            } else if (currentTab === 'stt') {
                await loadSTTData();
            } else if (currentTab === 'routing') {
                await loadRoutes();
            }
            btn.classList.remove('spinning');
        }

        // Handle URL hash navigation
        function handleHash() {
            const hash = window.location.hash.replace('#', '');
            if (hash && ['llm', 'tts', 'stt', 'routing'].includes(hash)) {
                switchTab(hash);
            }
        }
        window.addEventListener('hashchange', handleHash);

        // Initial load
        handleHash(); // Check for hash on page load

        // Check for ?request= param to open a specific request detail
        const urlParams = new URLSearchParams(window.location.search);
        const requestId = urlParams.get('request');
        if (requestId) {
            showRequestDetail(parseInt(requestId));
        }

        if (currentTab === 'llm') {
            loadStats();
            loadPending();
            loadUsecaseFilter();
        }
        setInterval(refresh, 30000);
        // More frequent updates for pending requests (only on LLM tab)
        setInterval(() => { if (currentTab === 'llm') loadPending(); }, 2000);
    </script>
</body>
</html>`

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(dashboardHTML))
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

// Test playground HTML
const testPlaygroundHTML = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Proxy - Test Playground</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #6366f1; margin-bottom: 8px; font-size: 28px; }
        .subtitle { color: #888; margin-bottom: 16px; }
        a { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Top Navigation Tabs */
        .nav-tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            background: #1a1a2e;
            padding: 4px;
            border-radius: 12px;
        }
        .nav-tab {
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-tab:hover { color: #e0e0e0; background: #252540; }
        .nav-tab.active { background: #6366f1; color: #fff; }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
        }
        .tab {
            padding: 12px 24px;
            background: #1a1a2e;
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            font-size: 16px;
        }
        .tab.active { background: #6366f1; color: #fff; }
        .tab:hover:not(.active) { background: #252540; }

        .panel { display: none; }
        .panel.active { display: block; }

        .form-group { margin-bottom: 16px; }
        label { display: block; color: #a5b4fc; margin-bottom: 8px; font-weight: 500; }

        textarea, input[type="text"], select {
            width: 100%;
            padding: 12px;
            background: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 14px;
            font-family: inherit;
        }
        textarea { min-height: 120px; resize: vertical; }
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: #6366f1;
        }

        .row { display: flex; gap: 16px; }
        .row > * { flex: 1; }

        .image-upload {
            border: 2px dashed #2d2d44;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .image-upload:hover { border-color: #6366f1; }
        .image-upload.has-image { border-style: solid; border-color: #22c55e; }
        .image-preview { max-width: 100%; max-height: 300px; margin-top: 16px; border-radius: 8px; }

        .submit-btn {
            background: #6366f1;
            color: #fff;
            border: none;
            padding: 14px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 16px;
        }
        .submit-btn:hover { background: #4f46e5; }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .result-box {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
        }
        .result-header {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid #2d2d44;
        }
        .result-meta { font-size: 14px; }
        .result-meta span { color: #888; }
        .result-meta strong { color: #a5b4fc; }

        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge.openai { background: #10b981; color: #000; }
        .badge.anthropic { background: #f59e0b; color: #000; }
        .badge.ollama { background: #6366f1; color: #fff; }
        .badge.routed { background: #22c55e; color: #000; }
        .badge.override { background: #f59e0b; color: #000; }

        .result-content {
            background: #252540;
            border-radius: 8px;
            padding: 16px;
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            max-height: 400px;
            overflow-y: auto;
        }

        .thinking-box {
            background: #1e1b4b;
            border-left: 3px solid #8b5cf6;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            max-height: 300px;
            overflow-y: auto;
        }
        .thinking-box h4 {
            color: #a78bfa;
            margin: 0 0 12px 0;
            font-size: 14px;
        }
        .thinking-content {
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: #c4b5fd;
        }

        .loading { opacity: 0.6; pointer-events: none; }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #fff;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }

        .error { color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Proxy</h1>
        <p class="subtitle">Unified AI gateway with routing, caching, and cost tracking</p>

        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="nav-tab" onclick="window.location.href='/'">Requests</button>
            <button class="nav-tab active">Playground</button>
            <button class="nav-tab" onclick="window.location.href='/#routing'">Routing</button>
            <button class="nav-tab" onclick="window.location.href='/analytics'">Analytics</button>
        </div>

        <h2 style="color:#a5b4fc;margin-bottom:16px;font-size:20px;">Test Playground</h2>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('chat')">Chat Completion</button>
            <button class="tab" onclick="switchTab('vision')">Vision Analysis</button>
            <button class="tab" onclick="switchTab('whisper')">Speech to Text</button>
            <button class="tab" onclick="switchTab('tts')">Text to Speech</button>
        </div>

        <!-- Chat Panel -->
        <div id="chat-panel" class="panel active">
            <div class="form-group">
                <label>Prompt</label>
                <textarea id="chat-prompt" placeholder="Enter your prompt here...">Tell me a short joke about programming.</textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>Sensitive</label>
                    <select id="chat-sensitive">
                        <option value="false">false (can use cloud)</option>
                        <option value="true">true (local only)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Precision</label>
                    <select id="chat-precision">
                        <option value="low">low</option>
                        <option value="medium" selected>medium</option>
                        <option value="high">high</option>
                        <option value="very_high">very_high</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label>Model Override (optional - leave empty to use routing)</label>
                <select id="chat-model">
                    <option value="">-- Use automatic routing --</option>
                </select>
            </div>

            <button class="submit-btn" id="chat-submit" onclick="submitChat()">Send Request</button>

            <div id="chat-result" class="result-box" style="display:none;">
                <div class="result-header" id="chat-result-header"></div>
                <div class="result-content" id="chat-result-content"></div>
            </div>
        </div>

        <!-- Vision Panel -->
        <div id="vision-panel" class="panel">
            <div class="form-group">
                <label>Image</label>
                <div class="image-upload" id="image-drop" onclick="document.getElementById('image-input').click()">
                    <input type="file" id="image-input" accept="image/*" style="display:none" onchange="handleImageSelect(event)">
                    <p>Click or drag an image here</p>
                    <img id="image-preview" class="image-preview" style="display:none">
                </div>
            </div>

            <div class="form-group">
                <label>Prompt</label>
                <textarea id="vision-prompt" placeholder="What should I analyze in this image?">Describe what you see in this image in detail.</textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>Sensitive</label>
                    <select id="vision-sensitive">
                        <option value="false">false (can use cloud)</option>
                        <option value="true">true (local only)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Precision</label>
                    <select id="vision-precision">
                        <option value="low">low</option>
                        <option value="medium" selected>medium</option>
                        <option value="high">high</option>
                        <option value="very_high">very_high</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label>Model Override (optional - leave empty to use routing)</label>
                <select id="vision-model">
                    <option value="">-- Use automatic routing --</option>
                </select>
            </div>

            <button class="submit-btn" id="vision-submit" onclick="submitVision()">Analyze Image</button>

            <div id="vision-result" class="result-box" style="display:none;">
                <div class="result-header" id="vision-result-header"></div>
                <div class="result-content" id="vision-result-content"></div>
            </div>
        </div>

        <!-- Whisper Panel -->
        <div id="whisper-panel" class="panel">
            <div class="form-group">
                <label>Audio File</label>
                <div class="image-upload" id="audio-drop" onclick="document.getElementById('audio-input').click()">
                    <input type="file" id="audio-input" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm" style="display:none" onchange="handleAudioSelect(event)">
                    <p id="audio-placeholder">Click or drag an audio file here (wav, mp3, m4a, ogg, flac, webm)</p>
                    <p id="audio-filename" style="display:none; color:#22c55e; font-weight:600;"></p>
                </div>
            </div>

            <div class="form-group">
                <label>Or Record Audio</label>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <button class="submit-btn" id="record-btn" onclick="toggleRecording()" style="margin-top:0; background:#ef4444;">
                        🎤 Start Recording
                    </button>
                    <span id="recording-status" style="color:#888;"></span>
                </div>
                <p id="https-warning" style="display:none; color:#f59e0b; font-size:12px; margin-top:8px;">
                    ⚠️ Microphone requires HTTPS. Use file upload instead.
                </p>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>Mode</label>
                    <select id="whisper-mode">
                        <option value="single">Single-shot (full transcription)</option>
                        <option value="stream">Streaming (progressive results)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Routing</label>
                    <select id="whisper-sensitive">
                        <option value="true">Local (sensitive=true)</option>
                        <option value="false">Cloud/OpenAI (sensitive=false)</option>
                    </select>
                </div>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>Model (optional)</label>
                    <input type="text" id="whisper-model" placeholder="e.g. large-v3-turbo, whisper-1">
                </div>
                <div class="form-group">
                    <label>Language (optional)</label>
                    <input type="text" id="whisper-language" placeholder="e.g. en, fr, de, es">
                </div>
            </div>

            <button class="submit-btn" id="whisper-submit" onclick="submitWhisper()">Transcribe</button>

            <div id="whisper-result" class="result-box" style="display:none;">
                <div class="result-header" id="whisper-result-header"></div>
                <div class="result-content" id="whisper-result-content"></div>
            </div>
        </div>

        <!-- TTS Panel -->
        <div id="tts-panel" class="panel">
            <div class="form-group">
                <label>Text to speak</label>
                <textarea id="tts-text" placeholder="Enter text to synthesize...">Hello! This is a test of the text to speech system. How does it sound?</textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>Voice</label>
                    <select id="tts-voice">
                        <option value="alloy">alloy (American Female)</option>
                        <option value="echo">echo (American Male)</option>
                        <option value="fable">fable (British Female)</option>
                        <option value="onyx" selected>onyx (British Male)</option>
                        <option value="nova">nova (American Female)</option>
                        <option value="shimmer">shimmer (American Female)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Speed</label>
                    <input type="range" id="tts-speed" min="0.5" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('tts-speed-val').textContent=this.value">
                    <span id="tts-speed-val" style="color:#22c55e; font-weight:600;">1.0</span>x
                </div>
            </div>

            <div class="form-group">
                <label>Format</label>
                <select id="tts-format">
                    <option value="mp3">MP3</option>
                    <option value="wav">WAV</option>
                </select>
            </div>

            <button class="submit-btn" id="tts-submit" onclick="submitTTS()">Speak</button>

            <div id="tts-result" class="result-box" style="display:none;">
                <div class="result-header" id="tts-result-header"></div>
                <div id="tts-audio-container"></div>
            </div>
        </div>
    </div>

    <script>
        const PLAYGROUND_USECASE = 'test_playground';
        let selectedImageBase64 = null;

        // Load available models for dropdowns
        async function loadModelSelectors() {
            try {
                const resp = await fetch('/v1/models');
                const data = await resp.json();
                const models = (data.data || []).map(m => m.id).filter(id => id !== 'auto').sort();

                const chatSelect = document.getElementById('chat-model');
                const visionSelect = document.getElementById('vision-model');

                models.forEach(model => {
                    const opt1 = document.createElement('option');
                    opt1.value = model;
                    opt1.textContent = model;
                    chatSelect.appendChild(opt1);

                    const opt2 = document.createElement('option');
                    opt2.value = model;
                    opt2.textContent = model;
                    visionSelect.appendChild(opt2);
                });
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        }
        loadModelSelectors();

        // Helper to escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Parse <think> tags from response and render properly
        function renderResponseWithThinking(content, resultContent) {
            const thinkMatch = content.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/);
            if (thinkMatch) {
                const thinkingContent = thinkMatch[1].trim();
                const actualResponse = thinkMatch[2].trim();

                // Clear existing content
                resultContent.innerHTML = '';

                // Add thinking box
                const thinkingBox = document.createElement('div');
                thinkingBox.className = 'thinking-box';
                thinkingBox.innerHTML = '<h4>💭 Thinking</h4><div class="thinking-content">' + escapeHtml(thinkingContent) + '</div>';
                resultContent.appendChild(thinkingBox);

                // Add actual response
                const responseDiv = document.createElement('div');
                responseDiv.textContent = actualResponse;
                resultContent.appendChild(responseDiv);
            } else {
                // No thinking tags, just display normally
                resultContent.textContent = content;
            }
        }

        // Safely parse JSON but preserve raw text for error display
        async function parseJsonSafe(resp) {
            const raw = await resp.text();
            try {
                return { data: JSON.parse(raw), raw };
            } catch (e) {
                return { data: null, raw };
            }
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector('.tab[onclick*="' + tab + '"]').classList.add('active');
            document.getElementById(tab + '-panel').classList.add('active');
        }

        function handleImageSelect(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                selectedImageBase64 = e.target.result;
                const preview = document.getElementById('image-preview');
                preview.src = selectedImageBase64;
                preview.style.display = 'block';
                document.getElementById('image-drop').classList.add('has-image');
            };
            reader.readAsDataURL(file);
        }

        // Drag and drop
        const dropZone = document.getElementById('image-drop');
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#6366f1'; });
        dropZone.addEventListener('dragleave', e => { dropZone.style.borderColor = ''; });
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.style.borderColor = '';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                document.getElementById('image-input').files = e.dataTransfer.files;
                handleImageSelect({target: {files: e.dataTransfer.files}});
            }
        });

        async function submitChat() {
            const btn = document.getElementById('chat-submit');
            const resultBox = document.getElementById('chat-result');
            const resultHeader = document.getElementById('chat-result-header');
            const resultContent = document.getElementById('chat-result-content');

            const prompt = document.getElementById('chat-prompt').value;
            const sensitive = document.getElementById('chat-sensitive').value === 'true';
            const precision = document.getElementById('chat-precision').value;
            const modelOverride = document.getElementById('chat-model').value.trim();

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Processing...';
            resultBox.style.display = 'none';

            const body = {
                model: modelOverride || 'auto',
                messages: [{role: 'user', content: prompt}],
                sensitive: sensitive,
                precision: precision,
                usecase: PLAYGROUND_USECASE,
                no_cache: true
            };

            const startTime = Date.now();
            try {
                const resp = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });

                const latency = Date.now() - startTime;
                const {data, raw} = await parseJsonSafe(resp);

                if (resp.ok && data) {
                    const provider = resp.headers.get('X-LLM-Proxy-Provider') || data.provider || 'unknown';
                    const model = resp.headers.get('X-LLM-Proxy-Model') || data.model || 'unknown';
                    const cost = resp.headers.get('X-LLM-Proxy-Cost-USD') || '0';
                    const isOverride = modelOverride !== '';

                    resultHeader.innerHTML =
                        '<div class="result-meta"><span>Provider:</span> <span class="badge ' + provider + '">' + provider + '</span></div>' +
                        '<div class="result-meta"><span>Model:</span> <strong>' + model + '</strong></div>' +
                        '<div class="result-meta"><span>Routing:</span> <span class="badge ' + (isOverride ? 'override' : 'routed') + '">' + (isOverride ? 'override' : 'auto-routed') + '</span></div>' +
                        '<div class="result-meta"><span>Latency:</span> <strong>' + latency + 'ms</strong></div>' +
                        '<div class="result-meta"><span>Cost:</span> <strong>$' + parseFloat(cost).toFixed(6) + '</strong></div>' +
                        (data.usage ? '<div class="result-meta"><span>Tokens:</span> <strong>' + data.usage.total_tokens + '</strong></div>' : '');

                    renderResponseWithThinking(data.choices[0].message.content, resultContent);
                } else if (resp.ok) {
                    resultHeader.innerHTML = '<div class="result-meta error">Invalid response</div>';
                    resultContent.textContent = raw || 'Empty response from server';
                } else {
                    resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                    resultContent.textContent = data ? JSON.stringify(data, null, 2) : (raw || 'Unknown error');
                }
            } catch (err) {
                resultHeader.innerHTML = '<div class="result-meta error">Request failed</div>';
                resultContent.textContent = err.message;
            }

            resultBox.style.display = 'block';
            btn.disabled = false;
            btn.innerHTML = 'Send Request';
        }

        async function submitVision() {
            if (!selectedImageBase64) {
                alert('Please select an image first');
                return;
            }

            const btn = document.getElementById('vision-submit');
            const resultBox = document.getElementById('vision-result');
            const resultHeader = document.getElementById('vision-result-header');
            const resultContent = document.getElementById('vision-result-content');

            const prompt = document.getElementById('vision-prompt').value;
            const sensitive = document.getElementById('vision-sensitive').value === 'true';
            const precision = document.getElementById('vision-precision').value;
            const modelOverride = document.getElementById('vision-model').value.trim();

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Analyzing...';
            resultBox.style.display = 'none';

            const body = {
                model: modelOverride || 'auto',
                messages: [{
                    role: 'user',
                    content: [
                        {type: 'text', text: prompt},
                        {type: 'image_url', image_url: {url: selectedImageBase64}}
                    ]
                }],
                sensitive: sensitive,
                precision: precision,
                usecase: PLAYGROUND_USECASE,
                no_cache: true
            };

            const startTime = Date.now();
            try {
                const resp = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });

                const latency = Date.now() - startTime;
                const {data, raw} = await parseJsonSafe(resp);

                if (resp.ok && data) {
                    const provider = resp.headers.get('X-LLM-Proxy-Provider') || data.provider || 'unknown';
                    const model = resp.headers.get('X-LLM-Proxy-Model') || data.model || 'unknown';
                    const cost = resp.headers.get('X-LLM-Proxy-Cost-USD') || '0';
                    const isOverride = modelOverride !== '';

                    resultHeader.innerHTML =
                        '<div class="result-meta"><span>Provider:</span> <span class="badge ' + provider + '">' + provider + '</span></div>' +
                        '<div class="result-meta"><span>Model:</span> <strong>' + model + '</strong></div>' +
                        '<div class="result-meta"><span>Routing:</span> <span class="badge ' + (isOverride ? 'override' : 'routed') + '">' + (isOverride ? 'override' : 'auto-routed') + '</span></div>' +
                        '<div class="result-meta"><span>Latency:</span> <strong>' + latency + 'ms</strong></div>' +
                        '<div class="result-meta"><span>Cost:</span> <strong>$' + parseFloat(cost).toFixed(6) + '</strong></div>' +
                        (data.usage ? '<div class="result-meta"><span>Tokens:</span> <strong>' + data.usage.total_tokens + '</strong></div>' : '');

                    renderResponseWithThinking(data.choices[0].message.content, resultContent);
                } else if (resp.ok) {
                    resultHeader.innerHTML = '<div class="result-meta error">Invalid response</div>';
                    resultContent.textContent = raw || 'Empty response from server';
                } else {
                    resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                    resultContent.textContent = data ? JSON.stringify(data, null, 2) : (raw || 'Unknown error');
                }
            } catch (err) {
                resultHeader.innerHTML = '<div class="result-meta error">Request failed</div>';
                resultContent.textContent = err.message;
            }

            resultBox.style.display = 'block';
            btn.disabled = false;
            btn.innerHTML = 'Analyze Image';
        }

        // Whisper/Audio handling
        let selectedAudioFile = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        function handleAudioSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            selectedAudioFile = file;
            document.getElementById('audio-placeholder').style.display = 'none';
            document.getElementById('audio-filename').textContent = '🎵 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
            document.getElementById('audio-filename').style.display = 'block';
            document.getElementById('audio-drop').classList.add('has-image');
        }

        // Audio drag and drop
        const audioDropZone = document.getElementById('audio-drop');
        audioDropZone.addEventListener('dragover', e => { e.preventDefault(); audioDropZone.style.borderColor = '#6366f1'; });
        audioDropZone.addEventListener('dragleave', e => { audioDropZone.style.borderColor = ''; });
        audioDropZone.addEventListener('drop', e => {
            e.preventDefault();
            audioDropZone.style.borderColor = '';
            const file = e.dataTransfer.files[0];
            if (file && (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|ogg|flac|webm)$/i))) {
                document.getElementById('audio-input').files = e.dataTransfer.files;
                handleAudioSelect({target: {files: e.dataTransfer.files}});
            }
        });

        async function toggleRecording() {
            const btn = document.getElementById('record-btn');
            const status = document.getElementById('recording-status');

            if (!isRecording) {
                // Check if mediaDevices is available (requires HTTPS or localhost)
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert('Microphone recording requires HTTPS. Please use file upload instead.');
                    return;
                }

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(audioChunks, { type: 'audio/webm' });
                        selectedAudioFile = new File([blob], 'recording.webm', { type: 'audio/webm' });
                        document.getElementById('audio-placeholder').style.display = 'none';
                        document.getElementById('audio-filename').textContent = '🎤 Recording (' + (blob.size / 1024).toFixed(1) + ' KB)';
                        document.getElementById('audio-filename').style.display = 'block';
                        document.getElementById('audio-drop').classList.add('has-image');
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    btn.innerHTML = '⏹ Stop Recording';
                    btn.style.background = '#22c55e';
                    status.textContent = 'Recording...';

                    // Visual recording indicator
                    let seconds = 0;
                    const timer = setInterval(() => {
                        if (!isRecording) { clearInterval(timer); return; }
                        seconds++;
                        status.textContent = 'Recording... ' + seconds + 's';
                    }, 1000);
                } catch (err) {
                    alert('Could not access microphone: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                btn.innerHTML = '🎤 Start Recording';
                btn.style.background = '#ef4444';
                status.textContent = 'Recording saved';
            }
        }

        async function submitWhisper() {
            if (!selectedAudioFile) {
                alert('Please select or record an audio file first');
                return;
            }

            const btn = document.getElementById('whisper-submit');
            const resultBox = document.getElementById('whisper-result');
            const resultHeader = document.getElementById('whisper-result-header');
            const resultContent = document.getElementById('whisper-result-content');

            const mode = document.getElementById('whisper-mode').value;
            const sensitive = document.getElementById('whisper-sensitive').value;
            const model = document.getElementById('whisper-model').value.trim();
            const language = document.getElementById('whisper-language').value.trim();

            // Check streaming + cloud combo
            if (mode === 'stream' && sensitive === 'false') {
                alert('Streaming mode is only available with local processing (OpenAI API does not support streaming)');
                return;
            }

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>' + (mode === 'stream' ? 'Streaming...' : 'Transcribing...');
            resultBox.style.display = 'block';
            resultHeader.innerHTML = '<div class="result-meta"><span>Status:</span> <strong>Processing...</strong></div>';
            resultContent.textContent = '';

            const formData = new FormData();
            formData.append('file', selectedAudioFile);
            if (model) formData.append('model', model);
            if (language) formData.append('language', language);
            formData.append('sensitive', sensitive);

            const endpoint = mode === 'stream' ? '/v1/audio/transcriptions/stream' : '/v1/audio/transcriptions';
            const startTime = Date.now();

            try {
                if (mode === 'stream') {
                    // Streaming mode - use fetch with reader
                    const resp = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });

                    if (!resp.ok) {
                        const errorText = await resp.text();
                        throw new Error(errorText);
                    }

                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    let fullText = '';

                    resultHeader.innerHTML =
                        '<div class="result-meta"><span>Mode:</span> <span class="badge ollama">streaming</span></div>' +
                        '<div class="result-meta"><span>Provider:</span> <span class="badge ollama">local</span></div>';

                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value, {stream: true});
                        // Parse SSE data
                        const lines = chunk.split('\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.text) {
                                        fullText = data.text;
                                        resultContent.textContent = fullText;
                                    }
                                    if (data.error) {
                                        resultContent.textContent = 'Error: ' + data.error;
                                    }
                                } catch (e) {
                                    // Not valid JSON, might be partial
                                    fullText += line.slice(6);
                                    resultContent.textContent = fullText;
                                }
                            }
                        }
                    }

                    const latency = Date.now() - startTime;
                    resultHeader.innerHTML =
                        '<div class="result-meta"><span>Mode:</span> <span class="badge ollama">streaming</span></div>' +
                        '<div class="result-meta"><span>Provider:</span> <span class="badge ollama">local</span></div>' +
                        '<div class="result-meta"><span>Latency:</span> <strong>' + latency + 'ms</strong></div>';

                } else {
                    // Single-shot mode
                    const resp = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });

                    const latency = Date.now() - startTime;
                    const provider = resp.headers.get('X-LLM-Proxy-Provider') || 'unknown';
                    const serverLatency = resp.headers.get('X-LLM-Proxy-Latency-Ms') || latency;

                    if (resp.ok) {
                        const data = await resp.json();
                        resultHeader.innerHTML =
                            '<div class="result-meta"><span>Mode:</span> <span class="badge routed">single-shot</span></div>' +
                            '<div class="result-meta"><span>Provider:</span> <span class="badge ' + (provider === 'openai' ? 'openai' : 'ollama') + '">' + provider + '</span></div>' +
                            '<div class="result-meta"><span>Latency:</span> <strong>' + serverLatency + 'ms</strong></div>';
                        resultContent.textContent = data.text || '(empty transcription)';
                    } else {
                        const errorText = await resp.text();
                        resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                        resultContent.textContent = errorText;
                    }
                }
            } catch (err) {
                resultHeader.innerHTML = '<div class="result-meta error">Request failed</div>';
                resultContent.textContent = err.message;
            }

            btn.disabled = false;
            btn.innerHTML = 'Transcribe';
        }

        // Check HTTPS on page load and disable recording if not available
        (function() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                const btn = document.getElementById('record-btn');
                const warning = document.getElementById('https-warning');
                if (btn) {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                }
                if (warning) warning.style.display = 'block';
            }
        })();

        // TTS handling
        async function submitTTS() {
            const text = document.getElementById('tts-text').value.trim();
            if (!text) {
                alert('Please enter some text to speak');
                return;
            }

            const btn = document.getElementById('tts-submit');
            const resultBox = document.getElementById('tts-result');
            const resultHeader = document.getElementById('tts-result-header');
            const audioContainer = document.getElementById('tts-audio-container');

            const voice = document.getElementById('tts-voice').value;
            const speed = parseFloat(document.getElementById('tts-speed').value);
            const format = document.getElementById('tts-format').value;

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Generating...';
            resultBox.style.display = 'block';
            resultHeader.innerHTML = '<div class="result-meta"><span>Status:</span> <strong>Processing...</strong></div>';
            audioContainer.innerHTML = '';

            const startTime = Date.now();

            try {
                const resp = await fetch('/v1/audio/speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'tts-1',
                        input: text,
                        voice: voice,
                        response_format: format,
                        speed: speed
                    })
                });

                const latency = Date.now() - startTime;

                if (resp.ok) {
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);

                    resultHeader.innerHTML =
                        '<div class="result-meta"><span>Voice:</span> <span class="badge ollama">' + voice + '</span></div>' +
                        '<div class="result-meta"><span>Speed:</span> <strong>' + speed + 'x</strong></div>' +
                        '<div class="result-meta"><span>Latency:</span> <strong>' + latency + 'ms</strong></div>';

                    audioContainer.innerHTML = '<audio controls autoplay src="' + url + '" style="width:100%; margin-top:12px;"></audio>';
                } else {
                    const errorText = await resp.text();
                    resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                    audioContainer.innerHTML = '<pre style="color:#ef4444;">' + errorText + '</pre>';
                }
            } catch (err) {
                resultHeader.innerHTML = '<div class="result-meta error">Request failed</div>';
                audioContainer.innerHTML = '<pre style="color:#ef4444;">' + err.message + '</pre>';
            }

            btn.disabled = false;
            btn.innerHTML = 'Speak';
        }
    </script>
</body>
</html>`

func handleTestPlayground(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(testPlaygroundHTML))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "service": "llm-proxy"})
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	metrics.mutex.RLock()
	defer metrics.mutex.RUnlock()

	var sb strings.Builder

	// Requests total
	sb.WriteString("# HELP llm_proxy_requests_total Total number of LLM requests\n")
	sb.WriteString("# TYPE llm_proxy_requests_total counter\n")
	for key, count := range metrics.RequestsTotal {
		parts := strings.SplitN(key, "|", 3)
		if len(parts) == 3 {
			sb.WriteString(fmt.Sprintf("llm_proxy_requests_total{provider=\"%s\",model=\"%s\",status=\"%s\"} %d\n",
				parts[0], parts[1], parts[2], count))
		}
	}

	// Tokens total
	sb.WriteString("\n# HELP llm_proxy_tokens_total Total tokens processed\n")
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

	// Cost
	sb.WriteString("\n# HELP llm_proxy_cost_usd_total Total cost in USD\n")
	sb.WriteString("# TYPE llm_proxy_cost_usd_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cost_usd_total %.6f\n", metrics.CostTotal))

	// Cache
	sb.WriteString("\n# HELP llm_proxy_cache_hits_total Total cache hits\n")
	sb.WriteString("# TYPE llm_proxy_cache_hits_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cache_hits_total %d\n", metrics.CacheHits))

	sb.WriteString("\n# HELP llm_proxy_cache_misses_total Total cache misses\n")
	sb.WriteString("# TYPE llm_proxy_cache_misses_total counter\n")
	sb.WriteString(fmt.Sprintf("llm_proxy_cache_misses_total %d\n", metrics.CacheMisses))

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

	// Initialize cache
	initCache()

	// SECURITY ASSERTION: Verify sensitive routes only use Ollama (local)
	// This runs on startup to catch configuration errors before serving requests
	validateRoutingTableSecurity()

	// Initialize provider adapters (ports -> implementations)
	initChatProviders()

	// Initialize chat handler (primary adapter)
	chatHandler = &httphandlers.ChatHandler{
		Router:        router,
		Providers:     chatProviders,
		Cache:         requestCache,
		Logger:        requestLogger,
		Metrics:       metrics,
		GenerateKey:   generateCacheKey,
		CalculateCost: calculateCost,
		AddPending:    addPendingRequest,
		RemovePending: removePendingRequest,
	}

	// Routes
	http.HandleFunc("/", handleDashboard)
	http.HandleFunc("/request/", handleRequestPage)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/metrics", handleMetrics)
	http.Handle("/v1/chat/completions", chatHandler)
	http.HandleFunc("/v1/estimate", handleEstimate)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/api/stats", handleStats)
	http.HandleFunc("/api/stats/models", handleModelStats)
	http.HandleFunc("/api/timing", handleTiming)
	http.HandleFunc("/api/routes", handleRoutes)
	http.HandleFunc("/api/routes/override", handleRouteOverride)
	http.HandleFunc("/api/usecases", handleUsecases)
	http.HandleFunc("/api/usecases/history", handleUsecasesHistory)
	http.HandleFunc("/api/usecases/distribution", handleUsecaseDistribution)
	http.HandleFunc("/api/models", handleAvailableModels)
	http.HandleFunc("/api/models/vision", handleVisionModels)
	http.HandleFunc("/api/history", handleRequestHistory)
	http.HandleFunc("/api/request", handleRequestDetail)
	http.HandleFunc("/api/replay", handleReplayRequest)
	http.HandleFunc("/api/cache/clear", handleClearCache)
	http.HandleFunc("/api/pending", handlePendingRequests)
	http.HandleFunc("/api/tts-history", handleTTSHistory)
	http.HandleFunc("/api/stt-history", handleSTTHistory)
	http.HandleFunc("/api/analytics", handleAnalytics)
	http.HandleFunc("/analytics", handleAnalyticsPage)
	http.HandleFunc("/test", handleTestPlayground)
	http.HandleFunc("/v1/audio/transcriptions", handleWhisperTranscription)
	http.HandleFunc("/v1/audio/transcriptions/stream", handleWhisperStream)
	http.HandleFunc("/v1/audio/speech", handleTTS)

	log.Printf("LLM Proxy starting on port %s", port)
	log.Printf("Whisper server: %s", whisperServerURL)
	log.Printf("TTS server: %s", ttsServerURL)
	log.Printf("OpenAI key: %v", openaiKey != "")
	log.Printf("Anthropic key: %v", anthropicKey != "")
	log.Printf("Ollama host: %s", ollamaHost)

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
