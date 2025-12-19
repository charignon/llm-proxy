package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"llm-proxy/internal/domain"
)

// LlamaCppProvider implements ChatProvider for llama.cpp server's OpenAI-compatible API.
type LlamaCppProvider struct {
	Host string // e.g., "studio.lan:8081"
}

// NewLlamaCppProvider creates a new llama.cpp server provider adapter.
func NewLlamaCppProvider(host string) *LlamaCppProvider {
	return &LlamaCppProvider{Host: host}
}

// Chat implements ChatProvider.Chat for llama.cpp server.
// llama-server provides an OpenAI-compatible /v1/chat/completions endpoint.
func (p *LlamaCppProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	llamaReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
		"stream":   false,
	}

	// Add optional parameters
	if req.MaxTokens > 0 {
		llamaReq["max_tokens"] = req.MaxTokens
	}
	if req.MaxCompletionTokens > 0 {
		llamaReq["max_tokens"] = req.MaxCompletionTokens
	}
	if req.Temperature > 0 {
		llamaReq["temperature"] = req.Temperature
	}

	body, _ := json.Marshal(llamaReq)
	log.Printf("[LlamaCpp] Request to %s, model=%s, messages=%d", p.Host, model, len(req.Messages))

	url := "http://" + p.Host + "/v1/chat/completions"
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second} // 5 min timeout for vision
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llama.cpp request failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("llama.cpp error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	result.Provider = "llamacpp"
	return &result, nil
}

// IsHealthy checks if the llama.cpp server is responding.
func (p *LlamaCppProvider) IsHealthy() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://" + p.Host + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// GetSlotInfo returns information about available slots from the llama.cpp server.
func (p *LlamaCppProvider) GetSlotInfo() (int, int, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://" + p.Host + "/slots")
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return 0, 0, fmt.Errorf("slots endpoint returned %d", resp.StatusCode)
	}

	var slots []struct {
		ID    int    `json:"id"`
		State int    `json:"state"` // 0 = idle, 1 = processing
		Model string `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return 0, 0, err
	}

	total := len(slots)
	idle := 0
	for _, slot := range slots {
		if slot.State == 0 {
			idle++
		}
	}

	return total, idle, nil
}
