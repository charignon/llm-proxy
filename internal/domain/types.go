// Package domain contains core business types and logic for the LLM proxy.
// These types are provider-agnostic and represent the domain model.
package domain

import "time"

// RouteConfig represents a routing decision to a specific provider/model.
type RouteConfig struct {
	Provider string
	Model    string
}

// ChatCompletionRequest represents a chat completion request in the domain.
// Uses OpenAI-compatible format as the canonical representation.
type ChatCompletionRequest struct {
	Model                 string      `json:"model"`
	Messages              []Message   `json:"messages"`
	MaxTokens             int         `json:"max_tokens,omitempty"`
	MaxCompletionTokens   int         `json:"max_completion_tokens,omitempty"` // Newer OpenAI API field
	Temperature           float64     `json:"temperature,omitempty"`
	Stream      bool        `json:"stream,omitempty"`
	Tools       []Tool      `json:"tools,omitempty"`
	ToolChoice  interface{} `json:"tool_choice,omitempty"`
	// Custom routing fields
	Sensitive *bool  `json:"sensitive,omitempty"`
	Precision string `json:"precision,omitempty"`
	Usecase   string `json:"usecase,omitempty"`
	NoCache   bool   `json:"no_cache,omitempty"`
	// Internal fields
	IsReplay bool `json:"-"`
}

// Message represents a chat message.
type Message struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"` // string or []ContentPart
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// ContentPart represents a part of multimodal content.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL in content.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// Tool represents a tool definition (function tools or server tools like web_search).
type Tool struct {
	Type     string        `json:"type"`
	Function *ToolFunction `json:"function,omitempty"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a tool invocation by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction contains the function name and arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionResponse represents a chat completion response.
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

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage contains token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CacheEntry represents a cached request/response pair.
type CacheEntry struct {
	Request   []byte
	Response  []byte
	CreatedAt time.Time
}

// PendingRequest tracks an in-flight request.
type PendingRequest struct {
	ID        string    `json:"id"`
	StartTime time.Time `json:"start_time"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	HasImages bool      `json:"has_images"`
	Sensitive bool      `json:"sensitive"`
	Precision string    `json:"precision"`
	Usecase   string    `json:"usecase"`
	Preview   string    `json:"preview"`
}

// RequestLog represents a logged request for history/analytics.
type RequestLog struct {
	ID              int64     `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	RequestType     string    `json:"request_type"`
	Provider        string    `json:"provider"`
	Model           string    `json:"model"`
	RequestedModel  string    `json:"requested_model"`
	Sensitive       bool      `json:"sensitive"`
	Precision       string    `json:"precision"`
	Usecase         string    `json:"usecase"`
	Cached          bool      `json:"cached"`
	InputTokens     int       `json:"input_tokens"`
	OutputTokens    int       `json:"output_tokens"`
	LatencyMs       int64     `json:"latency_ms"`
	CostUSD         float64   `json:"cost_usd"`
	Success         bool      `json:"success"`
	Error           string    `json:"error,omitempty"`
	CacheKey        string    `json:"cache_key"`
	HasImages       bool      `json:"has_images"`
	RequestBody     []byte    `json:"-"`
	ResponseBody    []byte    `json:"-"`
	Voice           string    `json:"voice,omitempty"`
	AudioDurationMs int64     `json:"audio_duration_ms,omitempty"`
	InputChars      int       `json:"input_chars,omitempty"`
	IsReplay        bool      `json:"is_replay,omitempty"`
	ClientIP        string    `json:"client_ip,omitempty"`
}

// TTSRequest represents a text-to-speech request.
type TTSRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format"`
	Speed          float64 `json:"speed"`
}

// WhisperTranscriptionResponse represents a transcription result.
type WhisperTranscriptionResponse struct {
	Text string `json:"text"`
}

// HasImages checks if the request contains image content.
func (req *ChatCompletionRequest) HasImages() bool {
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
