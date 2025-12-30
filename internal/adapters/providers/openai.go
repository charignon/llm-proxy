// Package providers contains adapters that implement the ChatProvider port.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/domain"
)

// OpenAIProvider implements ChatProvider for OpenAI's API.
type OpenAIProvider struct {
	APIKey           string
	Timeout          int // Timeout in seconds for regular requests
	StreamingTimeout int // Timeout in seconds for streaming requests
}

// NewOpenAIProvider creates a new OpenAI provider adapter.
func NewOpenAIProvider(apiKey string, timeout, streamingTimeout int) *OpenAIProvider {
	return &OpenAIProvider{
		APIKey:           apiKey,
		Timeout:          timeout,
		StreamingTimeout: streamingTimeout,
	}
}

// GetAPIKey returns the API key for streaming support.
func (p *OpenAIProvider) GetAPIKey() string {
	return p.APIKey
}

// Chat implements ChatProvider.Chat for OpenAI.
func (p *OpenAIProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}

	// Check if request has web_search tool - if so, use Responses API
	// LiteLLM may send it as type="web_search" or as type="function" with name containing "web_search"
	hasWebSearch := false
	var functionTools []domain.Tool
	for _, tool := range req.Tools {
		isWebSearch := tool.Type == "web_search" ||
			(tool.Type == "function" && tool.Function != nil &&
				(tool.Function.Name == "web_search" || strings.Contains(tool.Function.Name, "WebSearch")))
		if isWebSearch {
			hasWebSearch = true
			log.Printf("[DEBUG] Detected web_search tool: type=%s, name=%v", tool.Type, tool.Function)
		} else {
			functionTools = append(functionTools, tool)
		}
	}

	if hasWebSearch {
		log.Printf("[DEBUG] Using OpenAI Responses API for web_search")
		return p.chatWithResponses(req, model, functionTools)
	}

	// Standard Chat Completions API path
	return p.chatWithCompletions(req, model)
}

// chatWithCompletions uses the standard Chat Completions API
func (p *OpenAIProvider) chatWithCompletions(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	openaiReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}

	// Optionally strip reasoning summaries for OpenAI (Codex requests these by default,
	// but org verification may be required upstream).
	if req.StripReasoningSummary {
		delete(openaiReq, "reasoning")
		delete(openaiReq, "reasoning_summary")
		delete(openaiReq, "reasoning_summaries")
	}
	// Use MaxCompletionTokens if set, otherwise fall back to MaxTokens
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		// Newer OpenAI models (gpt-4o, o1, gpt-5.x, codex, etc.) require max_completion_tokens
		if strings.HasPrefix(model, "gpt-4o") || strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o1") ||
			strings.HasPrefix(model, "gpt-4.1") || strings.HasPrefix(model, "o3") || strings.HasPrefix(model, "o4") ||
			strings.Contains(model, "codex") {
			openaiReq["max_completion_tokens"] = maxTokens
		} else {
			openaiReq["max_tokens"] = maxTokens
		}
	}
	if req.Temperature > 0 {
		openaiReq["temperature"] = req.Temperature
	}
	if len(req.Tools) > 0 {
		openaiReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		openaiReq["tool_choice"] = req.ToolChoice
	}

	body, _ := json.Marshal(openaiReq)

	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: time.Duration(p.Timeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}
	result.Provider = "openai"
	return &result, nil
}

// chatWithResponses uses the Responses API which supports web_search
func (p *OpenAIProvider) chatWithResponses(req *domain.ChatCompletionRequest, model string, functionTools []domain.Tool) (*domain.ChatCompletionResponse, error) {
	// Build input from messages (Responses API uses a different format)
	var input []map[string]interface{}
	for _, msg := range req.Messages {
		inputMsg := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		input = append(input, inputMsg)
	}

	// Build tools array with web_search
	var tools []map[string]interface{}
	tools = append(tools, map[string]interface{}{"type": "web_search"})

	// Add function tools if any
	for _, tool := range functionTools {
		if tool.Function != nil {
			tools = append(tools, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
			})
		}
	}

	openaiReq := map[string]interface{}{
		"model": model,
		"input": input,
		"tools": tools,
	}

	// Add max tokens if set
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		openaiReq["max_output_tokens"] = maxTokens
	}

	body, _ := json.Marshal(openaiReq)
	log.Printf("[DEBUG] Responses API request: %s", string(body)[:min(500, len(body))])

	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/responses", bytes.NewReader(body))
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 240 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		log.Printf("[ERROR] Responses API error: %s", string(respBody))
		return nil, fmt.Errorf("OpenAI Responses API error %d: %s", resp.StatusCode, string(respBody))
	}

	log.Printf("[DEBUG] Responses API response: %s", string(respBody)[:min(500, len(respBody))])

	// Parse Responses API response format
	var responsesResult struct {
		ID      string `json:"id"`
		Status  string `json:"status"`
		Output  []struct {
			Type    string `json:"type"`
			ID      string `json:"id"`
			Status  string `json:"status"`
			Role    string `json:"role"`
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Annotations []struct {
					Type  string `json:"type"`
					URL   string `json:"url"`
					Title string `json:"title"`
				} `json:"annotations,omitempty"`
			} `json:"content,omitempty"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &responsesResult); err != nil {
		return nil, fmt.Errorf("failed to parse Responses API response: %v", err)
	}

	// Convert to ChatCompletionResponse format
	var textContent string
	for _, output := range responsesResult.Output {
		if output.Type == "message" {
			for _, content := range output.Content {
				if content.Type == "output_text" || content.Type == "text" {
					textContent += content.Text
				}
			}
		}
	}

	result := &domain.ChatCompletionResponse{
		ID:       responsesResult.ID,
		Model:    model,
		Provider: "openai",
		Choices: []domain.Choice{
			{
				Message: domain.Message{
					Role:    "assistant",
					Content: textContent,
				},
				FinishReason: "stop",
			},
		},
		Usage: &domain.Usage{
			PromptTokens:     responsesResult.Usage.InputTokens,
			CompletionTokens: responsesResult.Usage.OutputTokens,
		},
	}

	return result, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
