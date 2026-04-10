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
	Think     *bool  `json:"think,omitempty"` // Enable thinking mode for Ollama models (default: false)

	// Proxy behavior overrides
	StripReasoningSummary bool `json:"-"`

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
	Text     string                `json:"text"`
	Segments []WhisperSegment      `json:"segments,omitempty"`
	Language string                `json:"language,omitempty"`
}

// WhisperSegment represents a segment in verbose_json output.
type WhisperSegment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// ============================================================================
// OpenAI Responses API Types (newer API for reasoning models, web search, etc.)
// ============================================================================

// ResponsesAPIMode controls how /v1/responses requests are handled
type ResponsesAPIMode string

const (
	ResponsesAPIModeAuto      ResponsesAPIMode = "auto"       // Smart routing based on features
	ResponsesAPIModeOpenAI    ResponsesAPIMode = "openai"     // Always forward to OpenAI
	ResponsesAPIModeTranslate ResponsesAPIMode = "translate"  // Always translate to chat completions
)

// ResponsesRequest represents an OpenAI Responses API request
type ResponsesRequest struct {
	Model        string        `json:"model"`
	Input        interface{}   `json:"input"` // string or []InputItem
	Instructions string        `json:"instructions,omitempty"`
	Tools        []ResponsesTool `json:"tools,omitempty"`
	ToolChoice   interface{}   `json:"tool_choice,omitempty"`
	Reasoning    *ReasoningConfig `json:"reasoning,omitempty"`
	MaxOutputTokens int        `json:"max_output_tokens,omitempty"`
	Temperature  *float64      `json:"temperature,omitempty"`
	Stream       bool          `json:"stream,omitempty"`

	// Previous response for multi-turn
	PreviousResponseID string `json:"previous_response_id,omitempty"`

	// Custom routing fields (llm-proxy extensions)
	Sensitive *bool  `json:"sensitive,omitempty"`
	Precision string `json:"precision,omitempty"`
	Usecase   string `json:"usecase,omitempty"`
}

// ResponsesTool represents a tool in the Responses API
type ResponsesTool struct {
	Type     string        `json:"type"` // "function", "web_search", "code_interpreter", "file_search"
	Function *ToolFunction `json:"function,omitempty"`
}

// ReasoningConfig controls reasoning behavior for o1/o3 models
type ReasoningConfig struct {
	Effort  string `json:"effort,omitempty"`  // "low", "medium", "high"
	Summary string `json:"summary,omitempty"` // "auto", "concise", "detailed"
}

// InputItem represents an item in the input array (for multi-turn)
type InputItem struct {
	Type    string      `json:"type"` // "message", "item_reference"
	Role    string      `json:"role,omitempty"`
	Content interface{} `json:"content,omitempty"`
	ID      string      `json:"id,omitempty"` // for item_reference
}

// ResponsesResponse represents an OpenAI Responses API response
type ResponsesResponse struct {
	ID        string       `json:"id"`
	Object    string       `json:"object"` // "response"
	CreatedAt int64        `json:"created_at"`
	Status    string       `json:"status"` // "completed", "in_progress", "failed"
	Model     string       `json:"model"`
	Output    []OutputItem `json:"output"`
	Usage     *ResponsesUsage `json:"usage,omitempty"`
	Error     *ResponsesError `json:"error,omitempty"`
}

// OutputItem represents an item in the response output
type OutputItem struct {
	Type    string      `json:"type"` // "message", "web_search_call", "function_call", "reasoning"
	ID      string      `json:"id,omitempty"`
	Status  string      `json:"status,omitempty"`
	Role    string      `json:"role,omitempty"`
	Content []ContentItem `json:"content,omitempty"`
	// For function calls
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	// For reasoning
	Summary []ContentItem `json:"summary,omitempty"`
}

// ContentItem represents content within an output item
type ContentItem struct {
	Type string `json:"type"` // "text", "refusal"
	Text string `json:"text,omitempty"`
}

// ResponsesUsage contains token usage for Responses API
type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
	// Reasoning tokens (for o1/o3)
	InputTokensDetails  *TokenDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *TokenDetails `json:"output_tokens_details,omitempty"`
}

// TokenDetails provides breakdown of token usage
type TokenDetails struct {
	CachedTokens    int `json:"cached_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ResponsesError represents an error in the Responses API
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// RequiresOpenAI checks if this request needs OpenAI-specific features
func (r *ResponsesRequest) RequiresOpenAI() (bool, string) {
	// Check for built-in tools that only OpenAI supports
	for _, tool := range r.Tools {
		switch tool.Type {
		case "web_search":
			return true, "web_search tool requires OpenAI Responses API"
		case "code_interpreter":
			return true, "code_interpreter tool requires OpenAI Responses API"
		case "file_search":
			return true, "file_search tool requires OpenAI Responses API"
		}
	}

	// Check for reasoning config (o1/o3 specific)
	if r.Reasoning != nil {
		return true, "reasoning config requires OpenAI Responses API"
	}

	return false, ""
}

// ToChatCompletionRequest converts a Responses API request to Chat Completions format
func (r *ResponsesRequest) ToChatCompletionRequest() *ChatCompletionRequest {
	req := &ChatCompletionRequest{
		Model:     r.Model,
		Stream:    r.Stream,
		Sensitive: r.Sensitive,
		Precision: r.Precision,
		Usecase:   r.Usecase,
	}

	// Convert max tokens
	if r.MaxOutputTokens > 0 {
		req.MaxTokens = r.MaxOutputTokens
	}

	// Convert temperature
	if r.Temperature != nil {
		req.Temperature = *r.Temperature
	}

	// Build messages from instructions and input
	if r.Instructions != "" {
		req.Messages = append(req.Messages, Message{
			Role:    "system",
			Content: r.Instructions,
		})
	}

	// Convert input to user message
	switch input := r.Input.(type) {
	case string:
		req.Messages = append(req.Messages, Message{
			Role:    "user",
			Content: input,
		})
	case []interface{}:
		// Array of input items - convert each
		for _, item := range input {
			if m, ok := item.(map[string]interface{}); ok {
				msg := Message{}
				if role, ok := m["role"].(string); ok {
					msg.Role = role
				}
				if content, ok := m["content"]; ok {
					msg.Content = content
				}
				if msg.Role != "" {
					req.Messages = append(req.Messages, msg)
				}
			}
		}
	}

	// Convert function tools (skip built-in tools like web_search)
	for _, tool := range r.Tools {
		if tool.Type == "function" && tool.Function != nil {
			req.Tools = append(req.Tools, Tool{
				Type:     "function",
				Function: tool.Function,
			})
		}
	}

	if r.ToolChoice != nil {
		req.ToolChoice = r.ToolChoice
	}

	return req
}

// ChatCompletionToResponsesResponse converts a Chat Completions response to Responses API format
func ChatCompletionToResponsesResponse(resp *ChatCompletionResponse) *ResponsesResponse {
	result := &ResponsesResponse{
		ID:        "resp_" + resp.ID,
		Object:    "response",
		CreatedAt: resp.Created,
		Status:    "completed",
		Model:     resp.Model,
		Output:    []OutputItem{},
	}

	// Convert choices to output items
	for _, choice := range resp.Choices {
		item := OutputItem{
			Type: "message",
			Role: choice.Message.Role,
		}

		// Convert content
		switch content := choice.Message.Content.(type) {
		case string:
			item.Content = []ContentItem{{Type: "text", Text: content}}
		}

		// Convert tool calls
		for _, tc := range choice.Message.ToolCalls {
			result.Output = append(result.Output, OutputItem{
				Type:      "function_call",
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				CallID:    tc.ID,
				Status:    "completed",
			})
		}

		result.Output = append(result.Output, item)
	}

	// Convert usage
	if resp.Usage != nil {
		result.Usage = &ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
	}

	return result
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

// ProviderBudget represents a budget limit for a specific provider.
type ProviderBudget struct {
	Provider      string    `json:"provider"`
	BudgetUSD     float64   `json:"budget_usd"`
	MonthStartDay int       `json:"month_start_day"` // Day of month when period starts (1-31)
	Enabled       bool      `json:"enabled"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// GlobalBudget represents a global budget limit across all providers.
type GlobalBudget struct {
	ID           int64     `json:"id"`
	BudgetUSD    float64   `json:"budget_usd"`
	MonthStartDay int      `json:"month_start_day"` // Day of month when period starts (1-31)
	Enabled      bool      `json:"enabled"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}
