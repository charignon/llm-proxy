// Package providers contains adapters that implement the ChatProvider port.
package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"llm-proxy/internal/domain"
)

// TogetherProvider implements ChatProvider for Together.ai API using
// the OpenAI-compatible endpoint.
type TogetherProvider struct {
	APIKey  string
	Timeout int // Timeout in seconds
}

// NewTogetherProvider creates a new Together.ai provider adapter.
func NewTogetherProvider(apiKey string, timeout int) *TogetherProvider {
	return &TogetherProvider{
		APIKey:  apiKey,
		Timeout: timeout,
	}
}

// GetAPIKey returns the API key for streaming support.
func (p *TogetherProvider) GetAPIKey() string {
	return p.APIKey
}

// Chat implements ChatProvider.Chat for Together.ai using the OpenAI-compatible endpoint.
func (p *TogetherProvider) Chat(ctx context.Context, req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("TOGETHER_API_KEY not set")
	}

	// Build OpenAI-compatible request
	togetherReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}

	// Handle max tokens - Together uses max_tokens like standard OpenAI
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		togetherReq["max_tokens"] = maxTokens
	}

	if req.Temperature > 0 {
		togetherReq["temperature"] = req.Temperature
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		togetherReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		togetherReq["tool_choice"] = req.ToolChoice
	}

	body, _ := json.Marshal(togetherReq)

	// Use Together.ai's OpenAI-compatible endpoint
	endpoint := "https://api.together.xyz/v1/chat/completions"

	log.Printf("[DEBUG] Together request to %s: model=%s", endpoint, model)

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
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
		log.Printf("[ERROR] Together error %d: %s", resp.StatusCode, string(respBody))
		return nil, fmt.Errorf("Together error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}
	result.Provider = "together"
	return &result, nil
}
