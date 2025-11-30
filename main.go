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
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Configuration
var (
	port             = getEnv("PORT", "8080")
	openaiKey        = getEnv("OPENAI_API_KEY", "")
	anthropicKey     = getEnv("ANTHROPIC_API_KEY", "")
	ollamaHost       = getEnv("OLLAMA_HOST", "localhost:11434")
	dataDir          = getEnv("DATA_DIR", "./data")
	cacheTTLHours    = 24 * 7 // 1 week cache
	whisperServerURL = getEnv("WHISPER_SERVER_URL", "http://localhost:8890") // Local whisper server
)

// Model pricing per 1M tokens (input, output)
var modelPricing = map[string][2]float64{
	// OpenAI
	"gpt-4o":            {2.50, 10.00},
	"gpt-4o-mini":       {0.15, 0.60},
	"gpt-4-turbo":       {10.00, 30.00},
	"gpt-4":             {30.00, 60.00},
	"gpt-3.5-turbo":     {0.50, 1.50},
	"o1":                {15.00, 60.00},
	"o1-mini":           {3.00, 12.00},
	// Anthropic
	"claude-3-5-sonnet-20241022": {3.00, 15.00},
	"claude-3-5-haiku-20241022":  {0.80, 4.00},
	"claude-3-opus-20240229":     {15.00, 75.00},
	"claude-sonnet-4-20250514":   {3.00, 15.00},
	"claude-opus-4-20250514":     {15.00, 75.00},
	// Ollama (free)
	"qwen3-vl:30b":   {0, 0},
	"llama3:latest":  {0, 0},
	"llama3.3:70b":   {0, 0},
	"gemma3:latest":  {0, 0},
	"llava":          {0, 0},
}

// Routing configuration
type RouteConfig struct {
	Provider string
	Model    string
}

// Route based on sensitive + precision + hasImages flags
// Format: routingTable[sensitive][precision] for text
//         visionRoutingTable[sensitive][precision] for vision
// Precision levels: very_high > high > medium > low
var routingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (text only)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-20250514"},
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
var visionRoutingTable = map[string]map[string]*RouteConfig{
	// sensitive: false (can use cloud)
	"false": {
		"very_high": {Provider: "anthropic", Model: "claude-sonnet-4-20250514"}, // Claude has great vision
		"high":      {Provider: "openai", Model: "gpt-4o"},
		"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
	// sensitive: true (local only - no high-quality vision locally)
	"true": {
		"very_high": nil, // Not available - requires cloud
		"high":      nil, // Not available - no local model matches gpt-4o vision quality
		"medium":    {Provider: "ollama", Model: "qwen3-vl:30b"},
		"low":       {Provider: "ollama", Model: "qwen3-vl:30b"},
	},
}

// Database
var db *sql.DB
var dbMutex sync.Mutex

// Cache
var cache = make(map[string]*CacheEntry)
var cacheMutex sync.RWMutex

type CacheEntry struct {
	Request   []byte    // Original request body for replay
	Response  []byte
	CreatedAt time.Time
}

// Pending requests tracker
var pendingRequests = make(map[string]*PendingRequest)
var pendingMutex sync.RWMutex
var pendingCounter int64

type PendingRequest struct {
	ID        string    `json:"id"`
	StartTime time.Time `json:"start_time"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	HasImages bool      `json:"has_images"`
	Sensitive bool      `json:"sensitive"`
	Precision string    `json:"precision"`
	Preview   string    `json:"preview"` // First 100 chars of user message
}

// Prometheus metrics
type Metrics struct {
	RequestsTotal    map[string]int64          // provider:model:status -> count
	TokensTotal      map[string]int64          // provider:model:direction -> count
	DurationSumMs    map[string]int64          // provider:model -> sum of ms
	DurationCount    map[string]int64          // provider:model -> count
	CostTotal        float64
	CacheHits        int64
	CacheMisses      int64
	mutex            sync.RWMutex
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

// OpenAI API types
type ChatCompletionRequest struct {
	Model            string         `json:"model"`
	Messages         []Message      `json:"messages"`
	MaxTokens        int            `json:"max_tokens,omitempty"`
	Temperature      float64        `json:"temperature,omitempty"`
	Stream           bool           `json:"stream,omitempty"`
	// Custom routing fields (non-OpenAI)
	Sensitive        *bool          `json:"sensitive,omitempty"`
	Precision        string         `json:"precision,omitempty"`
	NoCache          bool           `json:"no_cache,omitempty"`
}

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ContentPart
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
	// Custom fields
	Cached   bool   `json:"cached,omitempty"`
	Provider string `json:"provider,omitempty"`
}

type Choice struct {
	Index        int      `json:"index"`
	Message      Message  `json:"message"`
	FinishReason string   `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Anthropic API types
type AnthropicRequest struct {
	Model     string            `json:"model"`
	MaxTokens int               `json:"max_tokens"`
	Messages  []AnthropicMessage `json:"messages"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type AnthropicResponse struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Model     string `json:"model"`
	StopReason string `json:"stop_reason"`
	Usage     struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Ollama API types
type OllamaRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type OllamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type OllamaResponse struct {
	Model   string `json:"model"`
	Message struct {
		Role     string `json:"role"`
		Content  string `json:"content"`
		Thinking string `json:"thinking,omitempty"` // Qwen3 thinking mode
	} `json:"message"`
	Done bool `json:"done"`
}

// Whisper API types (OpenAI-compatible)
type WhisperTranscriptionResponse struct {
	Text string `json:"text"`
}

// Request log entry
type RequestLog struct {
	ID              int64     `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	Provider        string    `json:"provider"`
	Model           string    `json:"model"`
	RequestedModel  string    `json:"requested_model"`
	Sensitive       bool      `json:"sensitive"`
	Precision       string    `json:"precision"`
	Cached          bool      `json:"cached"`
	InputTokens     int       `json:"input_tokens"`
	OutputTokens    int       `json:"output_tokens"`
	LatencyMs       int64     `json:"latency_ms"`
	CostUSD         float64   `json:"cost_usd"`
	Success         bool      `json:"success"`
	Error           string    `json:"error,omitempty"`
	CacheKey        string    `json:"cache_key"`
	HasImages       bool      `json:"has_images"`
	RequestBody     []byte    `json:"-"` // Stored in DB for persistence
	ResponseBody    []byte    `json:"-"` // Stored in DB for persistence
}

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

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
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
			response_body TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp);
		CREATE INDEX IF NOT EXISTS idx_provider ON requests(provider);
		CREATE INDEX IF NOT EXISTS idx_model ON requests(model);
	`)
	if err != nil {
		return err
	}

	// Add columns for existing databases (migration)
	db.Exec(`ALTER TABLE requests ADD COLUMN request_body TEXT`)
	db.Exec(`ALTER TABLE requests ADD COLUMN response_body TEXT`)
	return nil
}

func logRequest(entry *RequestLog) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	_, err := db.Exec(`
		INSERT INTO requests (timestamp, provider, model, requested_model, sensitive, precision, cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images, request_body, response_body)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, entry.Timestamp.Format(time.RFC3339), entry.Provider, entry.Model, entry.RequestedModel,
		entry.Sensitive, entry.Precision, entry.Cached, entry.InputTokens, entry.OutputTokens,
		entry.LatencyMs, entry.CostUSD, entry.Success, entry.Error, entry.CacheKey, entry.HasImages,
		string(entry.RequestBody), string(entry.ResponseBody))

	if err != nil {
		log.Printf("Failed to log request: %v", err)
	}
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

func generateCacheKey(req *ChatCompletionRequest) string {
	// Hash the request content
	h := sha256.New()
	h.Write([]byte(req.Model))
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

func getCached(key string) ([]byte, bool) {
	cacheMutex.RLock()
	defer cacheMutex.RUnlock()

	entry, ok := cache[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(cacheTTLHours)*time.Hour {
		return nil, false
	}

	return entry.Response, true
}

func setCache(key string, request, response []byte) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	cache[key] = &CacheEntry{
		Request:   request,
		Response:  response,
		CreatedAt: time.Now(),
	}
}

func getCachedRequest(key string) ([]byte, bool) {
	cacheMutex.RLock()
	defer cacheMutex.RUnlock()

	entry, ok := cache[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(cacheTTLHours)*time.Hour {
		return nil, false
	}

	return entry.Request, true
}

func getCachedResponse(key string) ([]byte, bool) {
	cacheMutex.RLock()
	defer cacheMutex.RUnlock()

	entry, ok := cache[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(cacheTTLHours)*time.Hour {
		return nil, false
	}

	return entry.Response, true
}

func hasImages(req *ChatCompletionRequest) bool {
	for _, msg := range req.Messages {
		switch c := msg.Content.(type) {
		case []interface{}:
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "image_url" {
						return true
					}
				}
			}
		case []ContentPart:
			for _, part := range c {
				if part.Type == "image_url" {
					return true
				}
			}
		}
	}
	return false
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

	sensitive := false
	if req.Sensitive != nil {
		sensitive = *req.Sensitive
	}

	pendingRequests[id] = &PendingRequest{
		ID:        id,
		StartTime: startTime,
		Provider:  route.Provider,
		Model:     route.Model,
		HasImages: hasImages(req),
		Sensitive: sensitive,
		Precision: req.Precision,
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

func resolveRoute(req *ChatCompletionRequest) (*RouteConfig, error) {
	// If model is explicitly specified (not a routing keyword), use it directly
	if req.Model != "" && req.Model != "auto" && req.Model != "route" {
		// Check if it's a known model and determine provider
		provider := "openai"
		if strings.HasPrefix(req.Model, "claude") {
			provider = "anthropic"
		} else if strings.Contains(req.Model, ":") || req.Model == "llama3" || req.Model == "gemma3" || req.Model == "llava" || strings.HasPrefix(req.Model, "qwen") {
			provider = "ollama"
		}
		return &RouteConfig{Provider: provider, Model: req.Model}, nil
	}

	// Use routing table - choose text or vision based on content
	sensitive := "false"
	if req.Sensitive != nil && *req.Sensitive {
		sensitive = "true"
	}

	precision := req.Precision
	if precision == "" {
		precision = "medium"
	}

	// Select appropriate routing table
	var selectedTable map[string]map[string]*RouteConfig
	if hasImages(req) {
		selectedTable = visionRoutingTable
	} else {
		selectedTable = routingTable
	}

	routes, ok := selectedTable[sensitive]
	if !ok {
		return nil, fmt.Errorf("invalid sensitive value")
	}

	route, ok := routes[precision]
	if !ok {
		return nil, fmt.Errorf("invalid precision value: %s", precision)
	}

	if route == nil {
		return nil, fmt.Errorf("capability not available: sensitive=%s, precision=%s", sensitive, precision)
	}

	return route, nil
}

func callOpenAI(req *ChatCompletionRequest, model string) (*ChatCompletionResponse, error) {
	if openaiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}

	// Convert request
	openaiReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}
	if req.MaxTokens > 0 {
		openaiReq["max_tokens"] = req.MaxTokens
	}
	if req.Temperature > 0 {
		openaiReq["temperature"] = req.Temperature
	}

	body, _ := json.Marshal(openaiReq)

	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Authorization", "Bearer "+openaiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI error %d: %s", resp.StatusCode, string(respBody))
	}

	var result ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}
	result.Provider = "openai"
	return &result, nil
}

func callAnthropic(req *ChatCompletionRequest, model string) (*ChatCompletionResponse, error) {
	if anthropicKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	// Convert messages to Anthropic format
	var messages []AnthropicMessage
	for _, msg := range req.Messages {
		messages = append(messages, AnthropicMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	anthropicReq := AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Messages:  messages,
	}

	body, _ := json.Marshal(anthropicReq)

	httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	httpReq.Header.Set("x-api-key", anthropicKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Anthropic error %d: %s", resp.StatusCode, string(respBody))
	}

	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(respBody, &anthropicResp); err != nil {
		return nil, err
	}

	// Convert to OpenAI format
	content := ""
	for _, c := range anthropicResp.Content {
		if c.Type == "text" {
			content += c.Text
		}
	}

	return &ChatCompletionResponse{
		ID:      anthropicResp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   anthropicResp.Model,
		Provider: "anthropic",
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: "assistant", Content: content},
			FinishReason: anthropicResp.StopReason,
		}},
		Usage: &Usage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
	}, nil
}

func callOllama(req *ChatCompletionRequest, model string) (*ChatCompletionResponse, error) {
	// Convert messages to Ollama format
	var messages []OllamaMessage
	for _, msg := range req.Messages {
		ollamaMsg := OllamaMessage{Role: msg.Role}

		switch c := msg.Content.(type) {
		case string:
			ollamaMsg.Content = c
		case []interface{}:
			// Handle multimodal content
			var textParts []string
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "text" {
						textParts = append(textParts, m["text"].(string))
					} else if m["type"] == "image_url" {
						if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
							url := imgURL["url"].(string)
							// Extract base64 from data URL
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

	ollamaReq := OllamaRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	}

	body, _ := json.Marshal(ollamaReq)

	httpReq, _ := http.NewRequest("POST", "http://"+ollamaHost+"/api/chat", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Ollama error %d: %s", resp.StatusCode, string(respBody))
	}

	var ollamaResp OllamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, err
	}

	// Build response content - include thinking wrapped in <think> tags if present
	responseContent := ollamaResp.Message.Content
	if ollamaResp.Message.Thinking != "" {
		responseContent = "<think>" + ollamaResp.Message.Thinking + "</think>" + responseContent
	}

	// Estimate tokens (rough approximation)
	inputTokens := 0
	for _, m := range messages {
		inputTokens += len(m.Content) / 4
	}
	outputTokens := len(responseContent) / 4

	return &ChatCompletionResponse{
		ID:      fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   ollamaResp.Model,
		Provider: "ollama",
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: "assistant", Content: responseContent},
			FinishReason: "stop",
		}},
		Usage: &Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}, nil
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

	startTime := time.Now()

	// Route based on sensitive flag
	sensitive := isSensitiveRequest(r)

	var resp *WhisperTranscriptionResponse
	var provider string

	if sensitive {
		// Use local whisper server
		resp, err = callLocalWhisper(fileContent, header.Filename, model, language)
		provider = "local"
	} else {
		// Use OpenAI Whisper API
		resp, err = callOpenAIWhisper(fileContent, header.Filename, model, language)
		provider = "openai"
	}

	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		log.Printf("Whisper transcription failed (%s): %v", provider, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

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

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Forward to local whisper server streaming endpoint
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\": \"failed to create form file\"}\n\n")
		flusher.Flush()
		return
	}
	if _, err := part.Write(fileContent); err != nil {
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
	url := whisperServerURL + "/v1/audio/transcriptions/stream"
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\": \"failed to create request\"}\n\n")
		flusher.Flush()
		return
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 300 * time.Second} // Longer timeout for streaming
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\": \"local whisper stream request failed: %s\"}\n\n", err.Error())
		flusher.Flush()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
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
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	logEntry := &RequestLog{
		Timestamp:      startTime,
		RequestedModel: req.Model,
		Precision:      req.Precision,
		HasImages:      hasImages(&req),
		RequestBody:    body, // Store for DB persistence
	}
	if req.Sensitive != nil {
		logEntry.Sensitive = *req.Sensitive
	}

	// Resolve routing
	route, err := resolveRoute(&req)
	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logRequest(logEntry)

		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	logEntry.Provider = route.Provider
	logEntry.Model = route.Model

	// Check cache
	cacheKey := generateCacheKey(&req)
	logEntry.CacheKey = cacheKey

	if !req.NoCache {
		if cached, ok := getCached(cacheKey); ok {
			logEntry.Cached = true
			logEntry.LatencyMs = time.Since(startTime).Milliseconds()
			logEntry.Success = true
			logEntry.ResponseBody = cached // Store cached response for DB persistence
			logRequest(logEntry)

			// Record cache hit metrics
			metrics.recordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, 0, 0, 0, true)

			// Add cached flag to response
			var resp ChatCompletionResponse
			json.Unmarshal(cached, &resp)
			resp.Cached = true

			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-LLM-Proxy-Cached", "true")
			json.NewEncoder(w).Encode(resp)
			return
		}
	}

	// Track pending request
	pendingID := addPendingRequest(&req, route, startTime)
	defer removePendingRequest(pendingID)

	// Make the actual call
	var resp *ChatCompletionResponse
	switch route.Provider {
	case "openai":
		resp, err = callOpenAI(&req, route.Model)
	case "anthropic":
		resp, err = callAnthropic(&req, route.Model)
	case "ollama":
		resp, err = callOllama(&req, route.Model)
	default:
		err = fmt.Errorf("unknown provider: %s", route.Provider)
	}

	logEntry.LatencyMs = time.Since(startTime).Milliseconds()

	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()

		// Cache the request even on error so we can debug it later
		setCache(cacheKey, body, nil)

		logRequest(logEntry)

		// Record error metrics
		metrics.recordRequest(route.Provider, route.Model, "error", logEntry.LatencyMs, 0, 0, 0, false)

		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Update log entry
	if resp.Usage != nil {
		logEntry.InputTokens = resp.Usage.PromptTokens
		logEntry.OutputTokens = resp.Usage.CompletionTokens
	}
	logEntry.CostUSD = calculateCost(route.Model, logEntry.InputTokens, logEntry.OutputTokens)
	logEntry.Success = true

	// Cache the response with original request for replay
	respBytes, _ := json.Marshal(resp)
	setCache(cacheKey, body, respBytes)

	logEntry.ResponseBody = respBytes // Store for DB persistence
	logRequest(logEntry)

	// Record metrics
	metrics.recordRequest(route.Provider, route.Model, "success", logEntry.LatencyMs, logEntry.InputTokens, logEntry.OutputTokens, logEntry.CostUSD, false)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", route.Provider)
	w.Header().Set("X-LLM-Proxy-Model", route.Model)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", logEntry.LatencyMs))
	w.Header().Set("X-LLM-Proxy-Cost-USD", fmt.Sprintf("%.6f", logEntry.CostUSD))
	json.NewEncoder(w).Encode(resp)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	models := []map[string]interface{}{}

	// Add routing models
	models = append(models, map[string]interface{}{
		"id":       "auto",
		"object":   "model",
		"owned_by": "llm-proxy",
	})

	// Add known models
	for model := range modelPricing {
		models = append(models, map[string]interface{}{
			"id":       model,
			"object":   "model",
			"owned_by": "llm-proxy",
		})
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
	ID           int64  `json:"id"`
	Timestamp    string `json:"timestamp"`
	Provider     string `json:"provider"`
	Model        string `json:"model"`
	Sensitive    bool   `json:"sensitive"`
	Precision    string `json:"precision"`
	HasImages    bool   `json:"has_images"`
	LatencyMs    int64  `json:"latency_ms"`
	CostUSD      float64 `json:"cost_usd"`
	Success      bool   `json:"success"`
	CacheKey     string `json:"cache_key"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
}

func handleRequestHistory(w http.ResponseWriter, r *http.Request) {
	dbMutex.Lock()
	defer dbMutex.Unlock()

	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}

	rows, err := db.Query(`
		SELECT id, timestamp, provider, model, sensitive, precision, has_images,
		       latency_ms, cost_usd, success, cache_key, input_tokens, output_tokens
		FROM requests
		ORDER BY id DESC
		LIMIT ?
	`, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var history []RequestHistoryEntry
	for rows.Next() {
		var entry RequestHistoryEntry
		var precision sql.NullString
		var cacheKey sql.NullString
		rows.Scan(&entry.ID, &entry.Timestamp, &entry.Provider, &entry.Model,
			&entry.Sensitive, &precision, &entry.HasImages, &entry.LatencyMs,
			&entry.CostUSD, &entry.Success, &cacheKey, &entry.InputTokens, &entry.OutputTokens)
		if precision.Valid {
			entry.Precision = precision.String
		}
		if cacheKey.Valid {
			entry.CacheKey = cacheKey.String
		}
		history = append(history, entry)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(history)
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
		SELECT id, timestamp, provider, model, requested_model, sensitive, precision,
		       cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images,
		       request_body, response_body
		FROM requests WHERE id = ?
	`, id).Scan(&entry.ID, &entry.Timestamp, &entry.Provider, &entry.Model,
		&entry.RequestedModel, &entry.Sensitive, &entry.Precision, &entry.Cached,
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
				"model":     req.Model,
				"messages":  displayMessages,
				"sensitive": req.Sensitive,
				"precision": req.Precision,
				"no_cache":  req.NoCache,
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

	// Get original request's cache_key to find cached content
	dbMutex.Lock()
	var cacheKey sql.NullString
	err := db.QueryRow(`SELECT cache_key FROM requests WHERE id = ?`, replayReq.RequestID).Scan(&cacheKey)
	dbMutex.Unlock()

	if err != nil {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	if !cacheKey.Valid {
		http.Error(w, "Original request has no cache key - cannot replay", http.StatusBadRequest)
		return
	}

	// Get the original request body from cache
	cachedReqBody, found := getCachedRequest(cacheKey.String)
	if !found {
		http.Error(w, "Original request content not in cache - cannot replay", http.StatusBadRequest)
		return
	}

	// Parse original request
	var origReq ChatCompletionRequest
	if err := json.Unmarshal(cachedReqBody, &origReq); err != nil {
		http.Error(w, "Failed to parse cached request", http.StatusInternalServerError)
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

	// Use response recorder to capture the response
	recorder := &responseRecorder{
		ResponseWriter: w,
		statusCode:     http.StatusOK,
	}

	handleChatCompletions(recorder, internalReq)
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

	cacheMutex.Lock()
	count := len(cache)
	cache = make(map[string]*CacheEntry)
	cacheMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"cleared": count,
		"message": fmt.Sprintf("Cleared %d cached entries", count),
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
			"count":      count,
			"cost_usd":   cost,
			"avg_latency_ms": avgLatency,
		}
	}
	stats["by_provider"] = byProvider

	// By model
	rows, _ = db.Query(`
		SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests GROUP BY model ORDER BY COUNT(*) DESC LIMIT 10
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

	// Recent requests
	rows, _ = db.Query(`
		SELECT id, timestamp, provider, model, cached, latency_ms, cost_usd, success, error,
		       sensitive, precision, has_images, input_tokens, output_tokens
		FROM requests ORDER BY id DESC LIMIT 20
	`)
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
		rows.Scan(&id, &timestamp, &provider, &model, &cached, &latencyMs, &costUsd, &success, &errStr,
			&sensitive, &precision, &hasImages, &inputTokens, &outputTokens)

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
		recent = append(recent, entry)
	}
	stats["recent_requests"] = recent

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

func handleRoutes(w http.ResponseWriter, r *http.Request) {
	routes := []map[string]interface{}{}

	// Text routes
	for sensitive, precisions := range routingTable {
		for precision, config := range precisions {
			entry := map[string]interface{}{
				"type":       "text",
				"sensitive":  sensitive == "true",
				"precision":  precision,
				"available":  config != nil,
			}
			if config != nil {
				entry["provider"] = config.Provider
				entry["model"] = config.Model
			}
			routes = append(routes, entry)
		}
	}

	// Vision routes
	for sensitive, precisions := range visionRoutingTable {
		for precision, config := range precisions {
			entry := map[string]interface{}{
				"type":       "vision",
				"sensitive":  sensitive == "true",
				"precision":  precision,
				"available":  config != nil,
			}
			if config != nil {
				entry["provider"] = config.Provider
				entry["model"] = config.Model
			}
			routes = append(routes, entry)
		}
	}

	// Sort for consistent output
	sort.Slice(routes, func(i, j int) bool {
		si := fmt.Sprintf("%s-%v-%s", routes[i]["type"], routes[i]["sensitive"], routes[i]["precision"])
		sj := fmt.Sprintf("%s-%v-%s", routes[j]["type"], routes[j]["sensitive"], routes[j]["precision"])
		return si < sj
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(routes)
}

// Dashboard HTML
const dashboardHTML = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Proxy Dashboard</title>
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
        .subtitle { color: #888; margin-bottom: 24px; }

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
        .badge.cached { background: #22c55e; color: #000; }
        .badge.error { background: #ef4444; color: #fff; }
        .badge.success { background: #22c55e; color: #000; }
        .badge.sensitive { background: #ef4444; color: #fff; }
        .badge.not-sensitive { background: #374151; color: #9ca3af; }
        .badge.precision { background: #8b5cf6; color: #fff; }
        .badge.images { background: #06b6d4; color: #000; }

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
        .route-flags { font-size: 13px; color: #888; margin-bottom: 8px; }
        .route-target { font-size: 15px; font-weight: 600; }

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
        <p class="subtitle">Unified AI gateway with routing, caching, and cost tracking | <a href="/test" style="color:#6366f1">Test Playground →</a></p>

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
            <h2 class="section-title">Routing Configuration</h2>
            <div class="routes-grid" id="routes-grid"></div>
        </div>

        <div class="section">
            <h2 class="section-title">By Provider</h2>
            <table id="provider-table">
                <thead><tr><th>Provider</th><th>Requests</th><th>Avg Latency</th><th>Cost</th></tr></thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="section" id="pending-section" style="display:none">
            <h2 class="section-title">⏳ Pending Requests <span class="pending-count" id="pending-count"></span></h2>
            <table id="pending-table">
                <thead><tr><th>Started</th><th>Provider</th><th>Model</th><th>Sensitive</th><th>Precision</th><th>Elapsed</th><th>Preview</th></tr></thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Recent Requests <span style="font-size:12px;color:#888;font-weight:normal">(click row for details)</span></h2>
            <table id="recent-table">
                <thead><tr><th>Time</th><th>Provider</th><th>Model</th><th>Sensitive</th><th>Precision</th><th>Tokens</th><th>Latency</th><th>Cost</th><th>Status</th></tr></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <button class="refresh-btn" id="refresh-btn" onclick="refresh()">↻</button>

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
        async function loadStats() {
            const resp = await fetch('/api/stats');
            const stats = await resp.json();

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

            // Recent requests
            const recentBody = document.querySelector('#recent-table tbody');
            recentBody.innerHTML = '';
            for (const req of stats.recent_requests || []) {
                const tr = document.createElement('tr');
                tr.className = 'clickable';
                tr.onclick = () => showRequestDetail(req.id);
                const time = new Date(req.timestamp).toLocaleTimeString();
                const status = req.cached ? '<span class="badge cached">cached</span>' :
                    (req.success ? '<span class="badge success">ok</span>' : '<span class="badge error">error</span>');
                const sensitiveClass = req.sensitive ? 'sensitive' : 'not-sensitive';
                const sensitiveText = req.sensitive ? 'YES' : 'no';
                const precision = req.precision || '-';
                const hasImages = req.has_images ? '<span class="badge images">img</span>' : '';
                const tokens = (req.input_tokens || 0) + (req.output_tokens || 0);
                tr.innerHTML = '<td>' + time + '</td>' +
                    '<td><span class="badge ' + req.provider + '">' + req.provider + '</span></td>' +
                    '<td>' + req.model + ' ' + hasImages + '</td>' +
                    '<td><span class="badge ' + sensitiveClass + '">' + sensitiveText + '</span></td>' +
                    '<td><span class="badge precision">' + precision + '</span></td>' +
                    '<td class="tokens-small">' + tokens + '</td>' +
                    '<td>' + req.latency_ms + 'ms</td>' +
                    '<td>$' + req.cost_usd.toFixed(6) + '</td>' +
                    '<td>' + status + '</td>';
                recentBody.appendChild(tr);
            }
        }

        async function loadRoutes() {
            const resp = await fetch('/api/routes');
            const routes = await resp.json();

            const grid = document.getElementById('routes-grid');
            grid.innerHTML = '';

            for (const route of routes) {
                const card = document.createElement('div');
                card.className = 'route-card' + (route.available ? '' : ' unavailable');
                card.innerHTML = '<div class="route-flags">sensitive: ' + route.sensitive + ', precision: ' + route.precision + '</div>' +
                    '<div class="route-target">' + (route.available ? route.provider + ' / ' + route.model : 'Not Available') + '</div>';
                grid.appendChild(card);
            }
        }

        async function showRequestDetail(id) {
            const overlay = document.getElementById('modal-overlay');
            const body = document.getElementById('modal-body');
            overlay.classList.add('active');
            body.innerHTML = '<div style="text-align:center;color:#888;padding:40px;">Loading...</div>';

            try {
                const resp = await fetch('/api/request?id=' + id);
                const data = await resp.json();
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

            let html = '<div class="modal-section"><h3>Request Metadata</h3><div class="detail-grid">';
            html += '<div class="detail-item"><div class="detail-label">ID</div><div class="detail-value">#' + data.id + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Timestamp</div><div class="detail-value">' + time + '</div></div>';
            html += '<div class="detail-item"><div class="detail-label">Provider</div><div class="detail-value"><span class="badge ' + data.provider + '">' + data.provider + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Model Used</div><div class="detail-value">' + data.model + '</div></div>';
            if (data.requested_model) {
                html += '<div class="detail-item"><div class="detail-label">Requested Model</div><div class="detail-value">' + data.requested_model + '</div></div>';
            }
            html += '<div class="detail-item"><div class="detail-label">Sensitive</div><div class="detail-value"><span class="badge ' + sensitiveClass + '">' + sensitiveText + '</span></div></div>';
            html += '<div class="detail-item"><div class="detail-label">Precision</div><div class="detail-value"><span class="badge precision">' + (data.precision || '-') + '</span></div></div>';
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
                html += '<div class="modal-section"><h3>Response</h3>';
                if (data.response.choices && data.response.choices.length > 0) {
                    const choice = data.response.choices[0];
                    if (choice.message && choice.message.content) {
                        html += '<div class="code-block">' + escapeHtml(choice.message.content) + '</div>';
                    } else {
                        html += '<div class="code-block">' + escapeHtml(JSON.stringify(data.response, null, 2)) + '</div>';
                    }
                } else {
                    html += '<div class="code-block">' + escapeHtml(JSON.stringify(data.response, null, 2)) + '</div>';
                }
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
            await Promise.all([loadStats(), loadRoutes(), loadPending()]);
            btn.classList.remove('spinning');
        }

        refresh();
        setInterval(refresh, 30000);
        // More frequent updates for pending requests
        setInterval(loadPending, 2000);
    </script>
</body>
</html>`

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(dashboardHTML))
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
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { color: #6366f1; margin-bottom: 8px; font-size: 28px; }
        .subtitle { color: #888; margin-bottom: 24px; }
        a { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }

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
        <h1>LLM Proxy Test Playground</h1>
        <p class="subtitle"><a href="/">← Back to Dashboard</a> | Test routing and model responses</p>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('chat')">Chat Completion</button>
            <button class="tab" onclick="switchTab('vision')">Vision Analysis</button>
            <button class="tab" onclick="switchTab('whisper')">Speech to Text</button>
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
                <input type="text" id="chat-model" placeholder="e.g. gpt-4o, claude-sonnet-4-20250514, qwen3:235b">
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
                <input type="text" id="vision-model" placeholder="e.g. gpt-4o, qwen3-vl:30b">
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
    </div>

    <script>
        let selectedImageBase64 = null;

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
                const data = await resp.json();

                if (resp.ok) {
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

                    resultContent.textContent = data.choices[0].message.content;
                } else {
                    resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                    resultContent.textContent = JSON.stringify(data, null, 2);
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
                const data = await resp.json();

                if (resp.ok) {
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

                    resultContent.textContent = data.choices[0].message.content;
                } else {
                    resultHeader.innerHTML = '<div class="result-meta error">Error: ' + resp.status + '</div>';
                    resultContent.textContent = JSON.stringify(data, null, 2);
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

	// SECURITY ASSERTION: Verify sensitive routes only use Ollama (local)
	// This runs on startup to catch configuration errors before serving requests
	validateRoutingTableSecurity()

	// Routes
	http.HandleFunc("/", handleDashboard)
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/metrics", handleMetrics)
	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/v1/estimate", handleEstimate)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/api/stats", handleStats)
	http.HandleFunc("/api/routes", handleRoutes)
	http.HandleFunc("/api/history", handleRequestHistory)
	http.HandleFunc("/api/request", handleRequestDetail)
	http.HandleFunc("/api/replay", handleReplayRequest)
	http.HandleFunc("/api/cache/clear", handleClearCache)
	http.HandleFunc("/api/pending", handlePendingRequests)
	http.HandleFunc("/test", handleTestPlayground)
	http.HandleFunc("/v1/audio/transcriptions", handleWhisperTranscription)
	http.HandleFunc("/v1/audio/transcriptions/stream", handleWhisperStream)

	log.Printf("LLM Proxy starting on port %s", port)
	log.Printf("Whisper server: %s", whisperServerURL)
	log.Printf("OpenAI key: %v", openaiKey != "")
	log.Printf("Anthropic key: %v", anthropicKey != "")
	log.Printf("Ollama host: %s", ollamaHost)

	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
