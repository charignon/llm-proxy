// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/config"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// WebSearchRequest represents a web search request.
type WebSearchRequest struct {
	Query          string   `json:"query"`
	Provider       string   `json:"provider"` // "anthropic" or "openai"
	Model          string   `json:"model"`
	MaxUses        int      `json:"max_uses"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	Usecase        string   `json:"usecase,omitempty"`
}

// WebSearchResponse represents a web search response.
type WebSearchResponse struct {
	Response    string            `json:"response"`
	Model       string            `json:"model"`
	Provider    string            `json:"provider"`
	SearchCount int               `json:"search_count"`
	Sources     []WebSearchSource `json:"sources,omitempty"`
	CostUSD     float64           `json:"cost_usd,omitempty"`
	Error       string            `json:"error,omitempty"`
	// Internal fields for logging (not sent to client)
	InputTokens  int    `json:"-"`
	OutputTokens int    `json:"-"`
	RawRequest   []byte `json:"-"`
	RawResponse  []byte `json:"-"`
}

// WebSearchSource represents a source from web search results.
type WebSearchSource struct {
	URL     string `json:"url"`
	Title   string `json:"title"`
	Snippet string `json:"snippet,omitempty"`
}

// WebSearchHandler handles web search requests.
type WebSearchHandler struct {
	AnthropicKey string
	OpenAIKey    string
	Logger       ports.RequestLogger
}

// NewWebSearchHandler creates a new web search handler.
func NewWebSearchHandler(anthropicKey, openaiKey string, logger ports.RequestLogger) *WebSearchHandler {
	return &WebSearchHandler{
		AnthropicKey: anthropicKey,
		OpenAIKey:    openaiKey,
		Logger:       logger,
	}
}

// HandleWebSearch handles POST /v1/websearch requests.
func (h *WebSearchHandler) HandleWebSearch(w http.ResponseWriter, r *http.Request) {
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

	logEntry := &domain.RequestLog{
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
		resp = h.doOpenAIWebSearch(req)
	} else {
		resp = h.doAnthropicWebSearch(req)
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

	h.Logger.LogRequest(logEntry)

	log.Printf("WebSearch complete (%dms): provider=%s, model=%s, searches=%d, cost=$%.6f",
		logEntry.LatencyMs, provider, logEntry.Model, resp.SearchCount, resp.CostUSD)

	json.NewEncoder(w).Encode(resp)
}

// doAnthropicWebSearch performs a web search using Anthropic's API.
func (h *WebSearchHandler) doAnthropicWebSearch(req WebSearchRequest) WebSearchResponse {
	if h.AnthropicKey == "" {
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
	httpReq.Header.Set("x-api-key", h.AnthropicKey)
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
			Type    string `json:"type"`
			Text    string `json:"text"`
			Name    string `json:"name"`
			Input   struct {
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
			InputTokens   int `json:"input_tokens"`
			OutputTokens  int `json:"output_tokens"`
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

	// Calculate cost
	pricing, ok := config.ModelPricing[model]
	if !ok {
		pricing = config.ModelPricing["claude-sonnet-4-5-20250929"]
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

// doOpenAIWebSearch performs a web search using OpenAI's API.
func (h *WebSearchHandler) doOpenAIWebSearch(req WebSearchRequest) WebSearchResponse {
	if h.OpenAIKey == "" {
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
	httpReq.Header.Set("Authorization", "Bearer "+h.OpenAIKey)

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

	// Calculate cost
	pricing, ok := config.ModelPricing[model]
	if !ok {
		pricing = config.ModelPricing["gpt-4o"]
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
