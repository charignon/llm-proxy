package domain

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// AnthropicMessagesRequest represents an Anthropic-compatible /v1/messages request.
// It includes llm-proxy routing extensions so Anthropic clients can still drive routing.
type AnthropicMessagesRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens,omitempty"`
	System        interface{}        `json:"system,omitempty"`
	Messages      []AnthropicMessage `json:"messages"`
	Tools         []AnthropicTool    `json:"tools,omitempty"`
	ToolChoice    interface{}        `json:"tool_choice,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`

	// llm-proxy routing extensions
	Sensitive *bool  `json:"sensitive,omitempty"`
	Precision string `json:"precision,omitempty"`
	Usecase   string `json:"usecase,omitempty"`
	NoCache   bool   `json:"no_cache,omitempty"`
}

// AnthropicMessage represents a single message in Anthropic's Messages API.
type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// AnthropicTool represents a tool definition in Anthropic's Messages API.
type AnthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// AnthropicContentBlock represents a content block in Anthropic's Messages API.
type AnthropicContentBlock struct {
	Type      string                 `json:"type"`
	Text      string                 `json:"text,omitempty"`
	Source    *AnthropicImageSource  `json:"source,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Input     map[string]interface{} `json:"input,omitempty"`
	ToolUseID string                 `json:"tool_use_id,omitempty"`
	Content   interface{}            `json:"content,omitempty"`
	IsError   bool                   `json:"is_error,omitempty"`
}

// AnthropicImageSource represents an Anthropic image source block.
type AnthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// AnthropicMessagesResponse represents an Anthropic-compatible /v1/messages response.
type AnthropicMessagesResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage          `json:"usage"`
}

// AnthropicUsage represents Anthropic token usage.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ToChatCompletionRequest converts an Anthropic Messages request into the proxy's
// canonical Chat Completions request.
func (r *AnthropicMessagesRequest) ToChatCompletionRequest() *ChatCompletionRequest {
	req := &ChatCompletionRequest{
		Model:     r.Model,
		MaxTokens: r.MaxTokens,
		Stream:    r.Stream,
		Sensitive: r.Sensitive,
		Precision: r.Precision,
		Usecase:   r.Usecase,
		NoCache:   r.NoCache,
	}

	if r.Temperature != nil {
		req.Temperature = *r.Temperature
	}

	if systemText := anthropicSystemToString(r.System); systemText != "" {
		req.Messages = append(req.Messages, Message{
			Role:    "system",
			Content: systemText,
		})
	}

	for _, tool := range r.Tools {
		req.Tools = append(req.Tools, Tool{
			Type: "function",
			Function: &ToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	if toolChoice := anthropicToolChoiceToOpenAI(r.ToolChoice); toolChoice != nil {
		req.ToolChoice = toolChoice
	}

	for _, msg := range r.Messages {
		req.Messages = append(req.Messages, anthropicMessageToChatMessages(msg)...)
	}

	return req
}

// ChatCompletionToAnthropicMessagesResponse converts a chat completion response
// into Anthropic's /v1/messages response shape.
func ChatCompletionToAnthropicMessagesResponse(resp *ChatCompletionResponse) *AnthropicMessagesResponse {
	result := &AnthropicMessagesResponse{
		ID:      resp.ID,
		Type:    "message",
		Role:    "assistant",
		Model:   resp.Model,
		Content: []AnthropicContentBlock{},
	}

	if result.ID == "" {
		result.ID = fmt.Sprintf("msg_%d", time.Now().UnixNano())
	}

	if resp.Usage != nil {
		result.Usage = AnthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
	}

	if len(resp.Choices) == 0 {
		result.StopReason = "end_turn"
		return result
	}

	choice := resp.Choices[0]
	result.StopReason = openAIFinishReasonToAnthropic(choice.FinishReason)

	switch content := choice.Message.Content.(type) {
	case string:
		if content != "" {
			result.Content = append(result.Content, AnthropicContentBlock{
				Type: "text",
				Text: content,
			})
		}
	case []interface{}:
		for _, part := range content {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if partMap["type"] != "text" {
				continue
			}
			text, _ := partMap["text"].(string)
			if text == "" {
				continue
			}
			result.Content = append(result.Content, AnthropicContentBlock{
				Type: "text",
				Text: text,
			})
		}
	}

	for _, toolCall := range choice.Message.ToolCalls {
		input := map[string]interface{}{}
		if strings.TrimSpace(toolCall.Function.Arguments) != "" {
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
				input = map[string]interface{}{
					"raw_arguments": toolCall.Function.Arguments,
				}
			}
		}
		result.Content = append(result.Content, AnthropicContentBlock{
			Type:  "tool_use",
			ID:    toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: input,
		})
	}

	return result
}

func anthropicSystemToString(system interface{}) string {
	switch s := system.(type) {
	case string:
		return s
	case []interface{}:
		var parts []string
		for _, block := range s {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] != "text" {
				continue
			}
			text, _ := blockMap["text"].(string)
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n\n")
	default:
		return ""
	}
}

func anthropicToolChoiceToOpenAI(toolChoice interface{}) interface{} {
	choice, ok := toolChoice.(map[string]interface{})
	if !ok {
		return nil
	}

	choiceType, _ := choice["type"].(string)
	switch choiceType {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "tool":
		name, _ := choice["name"].(string)
		if name == "" {
			return nil
		}
		return map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": name,
			},
		}
	case "none":
		return "none"
	default:
		return nil
	}
}

func anthropicMessageToChatMessages(msg AnthropicMessage) []Message {
	switch content := msg.Content.(type) {
	case string:
		return []Message{{
			Role:    msg.Role,
			Content: content,
		}}
	case []interface{}:
		switch msg.Role {
		case "assistant":
			return anthropicAssistantContentToChatMessages(content)
		case "user":
			return anthropicUserContentToChatMessages(content)
		default:
			chatContent := anthropicTextAndImageContent(content)
			if chatContent == nil {
				return nil
			}
			return []Message{{
				Role:    msg.Role,
				Content: chatContent,
			}}
		}
	default:
		return []Message{{
			Role:    msg.Role,
			Content: content,
		}}
	}
}

func anthropicAssistantContentToChatMessages(content []interface{}) []Message {
	var textParts []string
	var toolCalls []ToolCall

	for _, block := range content {
		blockMap, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		switch blockMap["type"] {
		case "text":
			text, _ := blockMap["text"].(string)
			if text != "" {
				textParts = append(textParts, text)
			}
		case "tool_use":
			toolCallID, _ := blockMap["id"].(string)
			name, _ := blockMap["name"].(string)
			input, _ := blockMap["input"].(map[string]interface{})
			argsJSON, _ := json.Marshal(input)
			toolCalls = append(toolCalls, ToolCall{
				ID:   toolCallID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	if len(textParts) == 0 && len(toolCalls) == 0 {
		return nil
	}

	message := Message{Role: "assistant"}
	if len(textParts) > 0 {
		message.Content = strings.Join(textParts, "\n")
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
		if len(textParts) == 0 {
			message.Content = nil
		}
	}

	return []Message{message}
}

func anthropicUserContentToChatMessages(content []interface{}) []Message {
	var result []Message
	var userParts []interface{}
	var textOnly []string

	flushUserMessage := func() {
		if len(userParts) == 0 && len(textOnly) == 0 {
			return
		}

		if len(userParts) == 0 {
			result = append(result, Message{
				Role:    "user",
				Content: strings.Join(textOnly, "\n"),
			})
			textOnly = nil
			return
		}

		if len(textOnly) > 0 {
			textParts := make([]interface{}, 0, len(textOnly))
			for _, text := range textOnly {
				textParts = append(textParts, map[string]interface{}{
					"type": "text",
					"text": text,
				})
			}
			userParts = append(textParts, userParts...)
			textOnly = nil
		}

		result = append(result, Message{
			Role:    "user",
			Content: append([]interface{}(nil), userParts...),
		})
		userParts = nil
	}

	for _, block := range content {
		blockMap, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		switch blockMap["type"] {
		case "text":
			text, _ := blockMap["text"].(string)
			if text != "" {
				textOnly = append(textOnly, text)
			}
		case "image":
			if len(textOnly) > 0 {
				for _, text := range textOnly {
					userParts = append(userParts, map[string]interface{}{
						"type": "text",
						"text": text,
					})
				}
				textOnly = nil
			}
			if part := anthropicImageBlockToOpenAI(blockMap); part != nil {
				userParts = append(userParts, part)
			}
		case "tool_result":
			flushUserMessage()
			result = append(result, Message{
				Role:       "tool",
				ToolCallID: stringValue(blockMap["tool_use_id"]),
				Content:    anthropicToolResultToString(blockMap["content"]),
			})
		}
	}

	flushUserMessage()
	return result
}

func anthropicTextAndImageContent(content []interface{}) interface{} {
	var textParts []string
	var multimodalParts []interface{}

	for _, block := range content {
		blockMap, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		switch blockMap["type"] {
		case "text":
			text, _ := blockMap["text"].(string)
			if text != "" {
				textParts = append(textParts, text)
				multimodalParts = append(multimodalParts, map[string]interface{}{
					"type": "text",
					"text": text,
				})
			}
		case "image":
			if part := anthropicImageBlockToOpenAI(blockMap); part != nil {
				multimodalParts = append(multimodalParts, part)
			}
		}
	}

	if len(multimodalParts) == 0 {
		if len(textParts) == 0 {
			return nil
		}
		return strings.Join(textParts, "\n")
	}

	hasNonText := false
	for _, part := range multimodalParts {
		partMap, ok := part.(map[string]interface{})
		if ok && partMap["type"] != "text" {
			hasNonText = true
			break
		}
	}
	if !hasNonText {
		return strings.Join(textParts, "\n")
	}

	return multimodalParts
}

func anthropicImageBlockToOpenAI(blockMap map[string]interface{}) map[string]interface{} {
	source, _ := blockMap["source"].(map[string]interface{})
	if source == nil {
		return nil
	}

	data, _ := source["data"].(string)
	if data == "" {
		return nil
	}

	mediaType, _ := source["media_type"].(string)
	if mediaType == "" {
		mediaType = "image/png"
	}

	return map[string]interface{}{
		"type": "image_url",
		"image_url": map[string]interface{}{
			"url": fmt.Sprintf("data:%s;base64,%s", mediaType, data),
		},
	}
}

func anthropicToolResultToString(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var textParts []string
		for _, block := range c {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if blockMap["type"] != "text" {
				continue
			}
			text, _ := blockMap["text"].(string)
			if text != "" {
				textParts = append(textParts, text)
			}
		}
		if len(textParts) > 0 {
			return strings.Join(textParts, "\n")
		}
		payload, _ := json.Marshal(c)
		return string(payload)
	default:
		payload, _ := json.Marshal(c)
		return string(payload)
	}
}

func openAIFinishReasonToAnthropic(finishReason string) string {
	switch finishReason {
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "stop", "":
		return "end_turn"
	default:
		return "end_turn"
	}
}

func stringValue(value interface{}) string {
	str, _ := value.(string)
	return str
}
