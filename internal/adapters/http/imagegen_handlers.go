// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"llm-proxy/internal/app"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// ImageGenPricing maps model -> size-quality -> cost per image.
var ImageGenPricing = map[string]map[string]float64{
	"dall-e-3": {
		"1024x1024-standard": 0.040,
		"1792x1024-standard": 0.080,
		"1024x1792-standard": 0.080,
		"1024x1024-hd":       0.080,
		"1792x1024-hd":       0.120,
		"1024x1792-hd":       0.120,
	},
	"dall-e-2": {
		"256x256":   0.016,
		"512x512":   0.018,
		"1024x1024": 0.020,
	},
}

// ImageGenHandler handles image generation requests.
type ImageGenHandler struct {
	OpenAIKey  string
	Logger     ports.RequestLogger
	Router     *app.Router
	CheckBudget func(provider string) error
}

// NewImageGenHandler creates a new image generation handler.
func NewImageGenHandler(openAIKey string, logger ports.RequestLogger, router *app.Router) *ImageGenHandler {
	return &ImageGenHandler{
		OpenAIKey: openAIKey,
		Logger:    logger,
		Router:    router,
	}
}

// HandleImageGeneration handles POST /v1/images/generations requests.
func (h *ImageGenHandler) HandleImageGeneration(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var req domain.ImageGenerationRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Prompt == "" {
		http.Error(w, "Missing required field: prompt", http.StatusBadRequest)
		return
	}

	// Resolve route (checks sensitive flag)
	route, err := h.Router.ResolveImageGenRoute(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.Model == "" || req.Model == "auto" || req.Model == "route" {
		req.Model = route.Model
	}
	if req.Size == "" {
		req.Size = "1024x1024"
	}
	if req.Quality == "" {
		req.Quality = "standard"
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "b64_json" // User requested b64_json as default
	}
	if req.N == 0 {
		req.N = 1 // DALL-E 3 only supports 1
	}

	// Check budget before processing request
	if h.CheckBudget != nil {
		if err := h.CheckBudget(route.Provider); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusPaymentRequired)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "budget_exceeded",
					"code":    "budget_exceeded",
				},
			})
			return
		}
	}

	// Prepare log entry (metadata only, not base64 response)
	sensitive := false
	if req.Sensitive != nil {
		sensitive = *req.Sensitive
	}
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "image_gen",
		Provider:    route.Provider,
		Model:       req.Model,
		Sensitive:   sensitive,
		Precision:   req.Precision,
		Usecase:     req.Usecase,
		InputChars:  len(req.Prompt),
		RequestBody: []byte(req.Prompt), // Store just the prompt, not full request
		ClientIP:    getClientIP(r),
	}

	// Build OpenAI request
	openaiReq := map[string]interface{}{
		"model":           req.Model,
		"prompt":          req.Prompt,
		"n":               req.N,
		"size":            req.Size,
		"quality":         req.Quality,
		"response_format": req.ResponseFormat,
	}
	if req.Style != "" {
		openaiReq["style"] = req.Style
	}
	if req.User != "" {
		openaiReq["user"] = req.User
	}

	reqBody, _ := json.Marshal(openaiReq)

	// Call OpenAI
	httpReq, _ := http.NewRequest("POST", "https://api.openai.com/v1/images/generations", bytes.NewReader(reqBody))
	httpReq.Header.Set("Authorization", "Bearer "+h.OpenAIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)

		log.Printf("Image generation failed: %v", err)
		http.Error(w, "OpenAI unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		h.Logger.LogRequest(logEntry)

		log.Printf("Image generation error %d: %s", resp.StatusCode, string(respBody))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true

	// Calculate cost
	logEntry.CostUSD = calculateImageGenCost(req.Model, req.Size, req.Quality, req.N)

	// Don't store base64 response in logs (too large)
	h.Logger.LogRequest(logEntry)

	log.Printf("Image generated (%dms): model=%s, size=%s, quality=%s, cost=$%.4f",
		latencyMs, req.Model, req.Size, req.Quality, logEntry.CostUSD)

	// Return response
	w.Header().Set("Content-Type", "application/json")
	w.Write(respBody)
}

// calculateImageGenCost calculates the cost for an image generation request.
func calculateImageGenCost(model, size, quality string, n int) float64 {
	pricing, ok := ImageGenPricing[model]
	if !ok {
		return 0
	}

	// Build key: size-quality for DALL-E 3, just size for DALL-E 2
	key := size
	if model == "dall-e-3" {
		key = size + "-" + quality
	}

	cost, ok := pricing[key]
	if !ok {
		return 0
	}

	return cost * float64(n)
}
