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

// MLXProvider implements ChatProvider for MLX LM server's OpenAI-compatible API.
// MLX is Apple's machine learning framework, optimized for Apple Silicon.
// The MLX LM server provides 2-3x faster inference than Ollama for dense models.
type MLXProvider struct {
	Host    string // e.g., "localhost:8086"
	Timeout int    // Timeout in seconds
}

// NewMLXProvider creates a new MLX LM server provider adapter.
func NewMLXProvider(host string, timeout int) *MLXProvider {
	return &MLXProvider{
		Host:    host,
		Timeout: timeout,
	}
}

// Chat implements ChatProvider.Chat for MLX LM server.
// MLX server provides an OpenAI-compatible /v1/chat/completions endpoint.
func (p *MLXProvider) Chat(ctx context.Context, req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	mlxReq := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
		"stream":   false,
	}

	// Add optional parameters
	if req.MaxTokens > 0 {
		mlxReq["max_tokens"] = req.MaxTokens
	}
	if req.MaxCompletionTokens > 0 {
		mlxReq["max_tokens"] = req.MaxCompletionTokens
	}
	if req.Temperature > 0 {
		mlxReq["temperature"] = req.Temperature
	}

	body, _ := json.Marshal(mlxReq)
	log.Printf("[MLX] Request to %s, model=%s, messages=%d", p.Host, model, len(req.Messages))

	url := "http://" + p.Host + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: time.Duration(p.Timeout) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("MLX request failed: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("MLX error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.ChatCompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	result.Provider = "mlx"
	return &result, nil
}

// IsHealthy checks if the MLX server is responding.
func (p *MLXProvider) IsHealthy() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://" + p.Host + "/v1/models")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// GetModels returns the list of available models from the MLX server.
func (p *MLXProvider) GetModels() ([]string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://" + p.Host + "/v1/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("models endpoint returned %d", resp.StatusCode)
	}

	var response struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	models := make([]string, len(response.Data))
	for i, m := range response.Data {
		models[i] = m.ID
	}

	return models, nil
}
