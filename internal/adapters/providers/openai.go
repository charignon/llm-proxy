// Package providers contains adapters that implement the ChatProvider port.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"llm-proxy/internal/domain"
)

// OpenAIProvider implements ChatProvider for OpenAI's API.
type OpenAIProvider struct {
	APIKey string
}

// NewOpenAIProvider creates a new OpenAI provider adapter.
func NewOpenAIProvider(apiKey string) *OpenAIProvider {
	return &OpenAIProvider{APIKey: apiKey}
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

	// Convert request
	openaiReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}
	if req.MaxTokens > 0 {
		// Newer OpenAI models (gpt-4o, o1, gpt-5.1, etc.) require max_completion_tokens
		if strings.HasPrefix(model, "gpt-4o") || strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o1") {
			openaiReq["max_completion_tokens"] = req.MaxTokens
		} else {
			openaiReq["max_tokens"] = req.MaxTokens
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

	client := &http.Client{Timeout: 240 * time.Second}
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
