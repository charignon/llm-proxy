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

// GeminiProvider implements ChatProvider for Google's Gemini API using
// the OpenAI-compatible endpoint.
type GeminiProvider struct {
	APIKey string
}

// NewGeminiProvider creates a new Gemini provider adapter.
func NewGeminiProvider(apiKey string) *GeminiProvider {
	return &GeminiProvider{APIKey: apiKey}
}

// GetAPIKey returns the API key for streaming support.
func (p *GeminiProvider) GetAPIKey() string {
	return p.APIKey
}

// Chat implements ChatProvider.Chat for Gemini using the OpenAI-compatible endpoint.
func (p *GeminiProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}

	// Build OpenAI-compatible request
	geminiReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}

	// Handle max tokens - Gemini uses max_completion_tokens like newer OpenAI models
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		geminiReq["max_completion_tokens"] = maxTokens
	}

	if req.Temperature > 0 {
		geminiReq["temperature"] = req.Temperature
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		geminiReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		geminiReq["tool_choice"] = req.ToolChoice
	}

	body, _ := json.Marshal(geminiReq)

	// Use Google's OpenAI-compatible endpoint
	endpoint := "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

	log.Printf("[DEBUG] Gemini request to %s: model=%s", endpoint, model)

	httpReq, _ := http.NewRequest("POST", endpoint, bytes.NewReader(body))
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
		log.Printf("[ERROR] Gemini error %d: %s", resp.StatusCode, string(respBody))
		return nil, fmt.Errorf("Gemini error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, err
	}
	result.Provider = "gemini"
	return &result, nil
}

// ChatNative uses Gemini's native API format (non-OpenAI compatible)
// This can be used for features not available in the OpenAI-compatible endpoint.
func (p *GeminiProvider) ChatNative(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}

	// Convert messages to Gemini native format
	var contents []map[string]interface{}
	var systemInstruction string

	for _, msg := range req.Messages {
		// Handle system messages - Gemini uses system_instruction
		if msg.Role == "system" {
			if content, ok := msg.Content.(string); ok {
				if systemInstruction != "" {
					systemInstruction += "\n\n"
				}
				systemInstruction += content
			}
			continue
		}

		// Map OpenAI roles to Gemini roles
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}

		// Handle content
		var parts []map[string]interface{}
		switch c := msg.Content.(type) {
		case string:
			parts = append(parts, map[string]interface{}{"text": c})
		case []interface{}:
			// Handle multimodal content
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "text" {
						parts = append(parts, map[string]interface{}{"text": m["text"]})
					} else if m["type"] == "image_url" {
						if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
							url := imgURL["url"].(string)
							if strings.HasPrefix(url, "data:") {
								// Parse data URL: data:image/png;base64,<data>
								urlParts := strings.SplitN(url, ",", 2)
								if len(urlParts) == 2 {
									mimeType := "image/png"
									if strings.Contains(urlParts[0], "image/jpeg") {
										mimeType = "image/jpeg"
									} else if strings.Contains(urlParts[0], "image/gif") {
										mimeType = "image/gif"
									} else if strings.Contains(urlParts[0], "image/webp") {
										mimeType = "image/webp"
									}
									parts = append(parts, map[string]interface{}{
										"inline_data": map[string]interface{}{
											"mime_type": mimeType,
											"data":      urlParts[1],
										},
									})
								}
							}
						}
					}
				}
			}
		}

		contents = append(contents, map[string]interface{}{
			"role":  role,
			"parts": parts,
		})
	}

	geminiReq := map[string]interface{}{
		"contents": contents,
	}

	if systemInstruction != "" {
		geminiReq["system_instruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{{"text": systemInstruction}},
		}
	}

	// Generation config
	genConfig := map[string]interface{}{}
	maxTokens := req.MaxCompletionTokens
	if maxTokens == 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens > 0 {
		genConfig["maxOutputTokens"] = maxTokens
	}
	if req.Temperature > 0 {
		genConfig["temperature"] = req.Temperature
	}
	if len(genConfig) > 0 {
		geminiReq["generationConfig"] = genConfig
	}

	body, _ := json.Marshal(geminiReq)

	// Native Gemini endpoint
	endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", model, p.APIKey)

	log.Printf("[DEBUG] Gemini native request: model=%s", model)

	httpReq, _ := http.NewRequest("POST", endpoint, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 240 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		log.Printf("[ERROR] Gemini native error %d: %s", resp.StatusCode, string(respBody))
		return nil, fmt.Errorf("Gemini error %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse native Gemini response
	var geminiResp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
				Role string `json:"role"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}

	if err := json.Unmarshal(respBody, &geminiResp); err != nil {
		return nil, err
	}

	// Convert to OpenAI format
	content := ""
	for _, candidate := range geminiResp.Candidates {
		for _, part := range candidate.Content.Parts {
			content += part.Text
		}
	}

	// Map Gemini finish reasons to OpenAI format
	finishReason := "stop"
	if len(geminiResp.Candidates) > 0 {
		switch geminiResp.Candidates[0].FinishReason {
		case "STOP":
			finishReason = "stop"
		case "MAX_TOKENS":
			finishReason = "length"
		case "SAFETY":
			finishReason = "content_filter"
		}
	}

	return &domain.ChatCompletionResponse{
		ID:       fmt.Sprintf("gemini-%d", time.Now().UnixNano()),
		Object:   "chat.completion",
		Created:  time.Now().Unix(),
		Model:    model,
		Provider: "gemini",
		Choices: []domain.Choice{{
			Index: 0,
			Message: domain.Message{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: finishReason,
		}},
		Usage: &domain.Usage{
			PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}
