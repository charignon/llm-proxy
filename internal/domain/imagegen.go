// Package domain contains core business types and logic for the LLM proxy.
package domain

// ImageGenerationRequest represents an image generation request (OpenAI-compatible format).
type ImageGenerationRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`               // Number of images (DALL-E 3 only supports 1)
	Size           string `json:"size,omitempty"`            // 1024x1024, 1792x1024, 1024x1792
	Quality        string `json:"quality,omitempty"`         // standard, hd
	ResponseFormat string `json:"response_format,omitempty"` // url, b64_json
	Style          string `json:"style,omitempty"`           // vivid, natural
	User           string `json:"user,omitempty"`
	// Custom routing fields (llm-proxy extensions)
	Sensitive *bool  `json:"sensitive,omitempty"`
	Precision string `json:"precision,omitempty"`
	Usecase   string `json:"usecase,omitempty"`
}

// ImageGenerationResponse represents an image generation response (OpenAI-compatible format).
type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
}

// ImageData represents a generated image in the response.
type ImageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}
