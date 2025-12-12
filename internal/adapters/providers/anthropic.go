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

// AnthropicProvider implements ChatProvider for Anthropic's Claude API.
type AnthropicProvider struct {
	APIKey string
}

// NewAnthropicProvider creates a new Anthropic provider adapter.
func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return &AnthropicProvider{APIKey: apiKey}
}

// Anthropic-specific types for API communication
type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Tools     []anthropicTool    `json:"tools,omitempty"`
}

type anthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type anthropicResponse struct {
	ID         string                  `json:"id"`
	Type       string                  `json:"type"`
	Role       string                  `json:"role"`
	Content    []anthropicContentBlock `json:"content"`
	Model      string                  `json:"model"`
	StopReason string                  `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicContentBlock struct {
	Type  string                 `json:"type"`
	Text  string                 `json:"text,omitempty"`
	ID    string                 `json:"id,omitempty"`
	Name  string                 `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`
}

// Chat implements ChatProvider.Chat for Anthropic.
func (p *AnthropicProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	if p.APIKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	// Convert messages to Anthropic format
	var systemParts []string
	var messages []anthropicMessage
	for _, msg := range req.Messages {
		// Handle system messages separately - Anthropic requires top-level system param
		if msg.Role == "system" {
			if content, ok := msg.Content.(string); ok {
				systemParts = append(systemParts, content)
			}
			continue
		}

		// Handle tool response messages (OpenAI role: "tool" -> Anthropic role: "user" with tool_result)
		if msg.Role == "tool" {
			content, _ := msg.Content.(string)
			toolResult := []map[string]interface{}{{
				"type":        "tool_result",
				"tool_use_id": msg.ToolCallID,
				"content":     content,
			}}
			messages = append(messages, anthropicMessage{
				Role:    "user",
				Content: toolResult,
			})
			continue
		}

		// Handle assistant messages with tool_calls
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var contentParts []map[string]interface{}
			// Add text content if present
			if content, ok := msg.Content.(string); ok && content != "" {
				contentParts = append(contentParts, map[string]interface{}{
					"type": "text",
					"text": content,
				})
			}
			// Add tool_use blocks
			for _, tc := range msg.ToolCalls {
				var inputArgs map[string]interface{}
				json.Unmarshal([]byte(tc.Function.Arguments), &inputArgs)
				contentParts = append(contentParts, map[string]interface{}{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": inputArgs,
				})
			}
			messages = append(messages, anthropicMessage{
				Role:    "assistant",
				Content: contentParts,
			})
			continue
		}

		// Convert content from OpenAI format to Anthropic format
		var anthropicContent interface{}
		switch c := msg.Content.(type) {
		case string:
			anthropicContent = c
		case []interface{}:
			// Handle multimodal content - convert image_url to Anthropic image format
			var contentParts []map[string]interface{}
			for _, part := range c {
				if m, ok := part.(map[string]interface{}); ok {
					if m["type"] == "text" {
						contentParts = append(contentParts, map[string]interface{}{
							"type": "text",
							"text": m["text"],
						})
					} else if m["type"] == "image_url" {
						if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
							url := imgURL["url"].(string)
							// Extract base64 and media type from data URL
							if strings.HasPrefix(url, "data:") {
								// Format: data:image/png;base64,<data>
								parts := strings.SplitN(url, ",", 2)
								if len(parts) == 2 {
									// Extract media type from first part (data:image/png;base64)
									mediaType := "image/png" // default
									if strings.Contains(parts[0], "image/jpeg") {
										mediaType = "image/jpeg"
									} else if strings.Contains(parts[0], "image/gif") {
										mediaType = "image/gif"
									} else if strings.Contains(parts[0], "image/webp") {
										mediaType = "image/webp"
									}
									contentParts = append(contentParts, map[string]interface{}{
										"type": "image",
										"source": map[string]interface{}{
											"type":       "base64",
											"media_type": mediaType,
											"data":       parts[1],
										},
									})
								}
							}
						}
					}
				}
			}
			anthropicContent = contentParts
		default:
			anthropicContent = c
		}
		messages = append(messages, anthropicMessage{
			Role:    msg.Role,
			Content: anthropicContent,
		})
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	anthropicReq := anthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		System:    strings.Join(systemParts, "\n\n"),
		Messages:  messages,
	}

	// Convert OpenAI tools to Anthropic format
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			if tool.Type == "function" && tool.Function != nil {
				antTool := anthropicTool{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					InputSchema: tool.Function.Parameters,
				}
				// Ensure input_schema has type: object if parameters exist
				if antTool.InputSchema == nil {
					antTool.InputSchema = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
				}
				anthropicReq.Tools = append(anthropicReq.Tools, antTool)
			}
			// TODO: Handle server tools like web_search when routing to Anthropic
		}
	}

	body, _ := json.Marshal(anthropicReq)

	httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	httpReq.Header.Set("x-api-key", p.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 240 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Anthropic error %d: %s", resp.StatusCode, string(respBody))
	}

	var antResp anthropicResponse
	if err := json.Unmarshal(respBody, &antResp); err != nil {
		return nil, err
	}

	// Convert to OpenAI format
	content := ""
	var toolCalls []domain.ToolCall
	for _, c := range antResp.Content {
		if c.Type == "text" {
			content += c.Text
		} else if c.Type == "tool_use" {
			// Convert Anthropic tool_use to OpenAI tool_calls format
			argsJSON, _ := json.Marshal(c.Input)
			toolCalls = append(toolCalls, domain.ToolCall{
				ID:   c.ID,
				Type: "function",
				Function: domain.ToolCallFunction{
					Name:      c.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Map Anthropic stop reasons to OpenAI format
	finishReason := antResp.StopReason
	if finishReason == "end_turn" {
		finishReason = "stop"
	} else if finishReason == "tool_use" {
		finishReason = "tool_calls"
	}

	msg := domain.Message{Role: "assistant", Content: content}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		// OpenAI sets content to null when there are tool calls
		if content == "" {
			msg.Content = nil
		}
	}

	return &domain.ChatCompletionResponse{
		ID:       antResp.ID,
		Object:   "chat.completion",
		Created:  time.Now().Unix(),
		Model:    antResp.Model,
		Provider: "anthropic",
		Choices: []domain.Choice{{
			Index:        0,
			Message:      msg,
			FinishReason: finishReason,
		}},
		Usage: &domain.Usage{
			PromptTokens:     antResp.Usage.InputTokens,
			CompletionTokens: antResp.Usage.OutputTokens,
			TotalTokens:      antResp.Usage.InputTokens + antResp.Usage.OutputTokens,
		},
	}, nil
}
