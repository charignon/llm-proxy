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

// OllamaProvider implements ChatProvider for local Ollama instances.
type OllamaProvider struct {
	Host    string // e.g., "localhost:11434"
	Timeout int    // Chat timeout in seconds
}

// NewOllamaProvider creates a new Ollama provider adapter.
func NewOllamaProvider(host string, timeout int) *OllamaProvider {
	return &OllamaProvider{
		Host:    host,
		Timeout: timeout,
	}
}

// Ollama-specific types for API communication
type ollamaRequest struct {
	Model      string          `json:"model"`
	Messages   []ollamaMessage `json:"messages"`
	Stream     bool            `json:"stream"`
	Tools      []domain.Tool   `json:"tools,omitempty"`
	ToolChoice interface{}     `json:"tool_choice,omitempty"`
	KeepAlive  string          `json:"keep_alive,omitempty"`
	Options    *ollamaOptions  `json:"options,omitempty"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

type ollamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ollamaResponse struct {
	Model   string `json:"model"`
	Message struct {
		Role      string           `json:"role"`
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking,omitempty"`
		ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done bool `json:"done"`
}

type ollamaToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

// Chat implements ChatProvider.Chat for Ollama.
func (p *OllamaProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	// Convert messages to Ollama format
	var messages []ollamaMessage
	for _, msg := range req.Messages {
		ollamaMsg := ollamaMessage{Role: msg.Role}

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

	ollamaReq := ollamaRequest{
		Model:      model,
		Messages:   messages,
		Stream:     false,
		Tools:      req.Tools,
		ToolChoice: req.ToolChoice,
		KeepAlive:  "30m", // Keep model loaded for 30 min of inactivity
		Options: &ollamaOptions{
			Temperature: 0.3,  // Low temperature for consistent detection
			NumCtx:      8192, // Larger context for vision tasks
		},
	}

	body, _ := json.Marshal(ollamaReq)
	log.Printf("Ollama request - tools: %d, tool_choice: %v", len(req.Tools), req.ToolChoice)

	httpReq, _ := http.NewRequest("POST", "http://"+p.Host+"/api/chat", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: time.Duration(p.Timeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Ollama error %d: %s", resp.StatusCode, string(respBody))
	}

	var ollamaResp ollamaResponse
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

	// Build response message
	respMsg := domain.Message{Role: "assistant", Content: responseContent}
	finishReason := "stop"

	// Convert Ollama tool calls to OpenAI format
	if len(ollamaResp.Message.ToolCalls) > 0 {
		finishReason = "tool_calls"
		for _, tc := range ollamaResp.Message.ToolCalls {
			// Convert arguments map to JSON string (OpenAI format)
			argsJSON, _ := json.Marshal(tc.Function.Arguments)
			toolCallID := tc.ID
			if toolCallID == "" {
				toolCallID = fmt.Sprintf("call_%d", time.Now().UnixNano())
			}
			respMsg.ToolCalls = append(respMsg.ToolCalls, domain.ToolCall{
				ID:   toolCallID,
				Type: "function",
				Function: domain.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: string(argsJSON),
				},
			})
		}
		// Tool calls typically have empty content
		if responseContent == "" {
			respMsg.Content = nil
		}
	}

	return &domain.ChatCompletionResponse{
		ID:       fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Object:   "chat.completion",
		Created:  time.Now().Unix(),
		Model:    ollamaResp.Model,
		Provider: "ollama",
		Choices: []domain.Choice{{
			Index:        0,
			Message:      respMsg,
			FinishReason: finishReason,
		}},
		Usage: &domain.Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}, nil
}

// OllamaTagsResponse is the response from Ollama /api/tags.
type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// OllamaModelDetails is the detailed response from Ollama /api/show.
type OllamaModelDetails struct {
	Details struct {
		Family   string   `json:"family"`
		Families []string `json:"families"`
	} `json:"details"`
}

// GetModels fetches the list of available models from Ollama.
func (p *OllamaProvider) GetModels() []string {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://" + p.Host + "/api/tags")
	if err != nil {
		log.Printf("Failed to fetch Ollama models: %v", err)
		return []string{"mistral:7b", "qwen3-vl:30b"} // fallback
	}
	defer resp.Body.Close()

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		log.Printf("Failed to decode Ollama models: %v", err)
		return []string{"mistral:7b", "qwen3-vl:30b"} // fallback
	}

	models := make([]string, 0, len(tagsResp.Models))
	for _, m := range tagsResp.Models {
		name := strings.ToLower(m.Name)
		if strings.Contains(name, "gemma") {
			continue
		}
		models = append(models, m.Name)
	}
	return models
}

// GetVisionModels returns only vision-capable Ollama models.
func (p *OllamaProvider) GetVisionModels() []string {
	client := &http.Client{Timeout: 10 * time.Second}

	// First get all models
	resp, err := client.Get("http://" + p.Host + "/api/tags")
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
		detailResp, err := client.Post("http://"+p.Host+"/api/show",
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
			if strings.Contains(strings.ToLower(m.Name), "gemma") {
				continue
			}
			visionModels = append(visionModels, m.Name)
		}
	}
	return visionModels
}
