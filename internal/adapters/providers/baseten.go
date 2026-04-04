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

// BasetenProvider implements ChatProvider for Baseten API using
// the OpenAI-compatible endpoint.
type BasetenProvider struct {
	APIKey  string
	Timeout int // Timeout in seconds
}

// NewBasetenProvider creates a new Baseten provider adapter.
func NewBasetenProvider(apiKey string, timeout int) *BasetenProvider {
	return &BasetenProvider{
		APIKey:  apiKey,
		Timeout: timeout,
	}
}

// GetAPIKey returns the API key for streaming support.
func (p *BasetenProvider) GetAPIKey() string {
	return p.APIKey
}

// Chat implements ChatProvider.Chat for Baseten using the OpenAI-compatible endpoint.
func (p *BasetenProvider) Chat(ctx context.Context, req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("BASETEN_API_KEY not set")
	}

	// Build OpenAI-compatible request
	basetenReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}

	// Handle max tokens - Baseten uses max_tokens like standard OpenAI
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		basetenReq["max_tokens"] = maxTokens
	}

	if req.Temperature > 0 {
		basetenReq["temperature"] = req.Temperature
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		basetenReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		basetenReq["tool_choice"] = req.ToolChoice
	}

	body, _ := json.Marshal(basetenReq)

	// Use Baseten's OpenAI-compatible endpoint
	endpoint := "https://inference.baseten.co/v1/chat/completions"

	log.Printf("[DEBUG] Baseten request to %s: model=%s", endpoint, model)

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
		log.Printf("[ERROR] Baseten error %d: %s", resp.StatusCode, string(respBody))
		return nil, fmt.Errorf("Baseten error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}
	result.Provider = "baseten"
	return &result, nil
}
