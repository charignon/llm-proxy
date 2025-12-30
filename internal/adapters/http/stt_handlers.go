// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"llm-proxy/internal/adapters/loadmanager"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// STTHandler handles speech-to-text (Whisper) transcription requests.
type STTHandler struct {
	WhisperServerURL  string
	OpenAIKey         string
	Logger            ports.RequestLogger
	ConcurrencyMgr    *loadmanager.ConcurrencyManager
	Timeout           int // Speech timeout in seconds
	StreamingTimeout  int // Speech streaming timeout in seconds
}

// NewSTTHandler creates a new STT handler.
func NewSTTHandler(whisperServerURL, openaiKey string, logger ports.RequestLogger, timeout, streamingTimeout int) *STTHandler {
	return &STTHandler{
		WhisperServerURL: whisperServerURL,
		OpenAIKey:        openaiKey,
		Logger:           logger,
		Timeout:          timeout,
		StreamingTimeout: streamingTimeout,
	}
}

// HandleTranscription handles POST /v1/audio/transcriptions requests.
func (h *STTHandler) HandleTranscription(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Acquire concurrency slot with configured timeout
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(h.Timeout)*time.Second)
	defer cancel()

	if err := h.ConcurrencyMgr.AcquireSlot(ctx); err != nil {
		if err == context.DeadlineExceeded {
			http.Error(w, "Request timeout waiting for available slot", http.StatusServiceUnavailable)
		} else {
			// Queue full (429)
			w.Header().Set("Retry-After", "30")
			http.Error(w, err.Error(), http.StatusTooManyRequests)
		}
		return
	}
	defer h.ConcurrencyMgr.ReleaseSlot()

	startTime := time.Now()

	// Parse multipart form (max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Get the audio file
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing or invalid file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read file content
	fileContent, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Get optional parameters
	model := r.FormValue("model")
	language := r.FormValue("language")
	prompt := r.FormValue("prompt") // Context hints for transcription (names, jargon, etc.)

	// Route based on sensitive flag
	sensitive := isSensitiveRequest(r)

	// Prepare log entry
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "stt",
		Sensitive:   sensitive,
		InputChars:  len(fileContent), // Store file size in InputChars field
		ClientIP:    getClientIP(r),
	}

	var resp *domain.WhisperTranscriptionResponse
	var provider string

	if sensitive {
		// Use local whisper server
		resp, err = h.callLocalWhisper(fileContent, header.Filename, model, language, prompt)
		provider = "local"
		logEntry.Provider = "local"
		logEntry.Model = "whisper-large-v3" // Local server uses whisper-large-v3
	} else {
		// Use OpenAI Whisper API
		resp, err = h.callOpenAIWhisper(fileContent, header.Filename, model, language, prompt)
		provider = "openai"
		logEntry.Provider = "openai"
		logEntry.Model = "whisper-1"
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs

	if err != nil {
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)

		log.Printf("Whisper transcription failed (%s): %v", provider, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	logEntry.Success = true
	logEntry.OutputTokens = len(resp.Text) // Store output text length
	// Store transcribed text in ResponseBody for history display
	logEntry.ResponseBody = []byte(resp.Text)
	h.Logger.LogRequest(logEntry)

	log.Printf("Whisper transcription complete (%s, %dms): %d chars", provider, latencyMs, len(resp.Text))

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-LLM-Proxy-Provider", provider)
	w.Header().Set("X-LLM-Proxy-Latency-Ms", fmt.Sprintf("%d", latencyMs))
	json.NewEncoder(w).Encode(resp)
}

// HandleStream handles POST /v1/audio/transcriptions/stream requests.
func (h *STTHandler) HandleStream(w http.ResponseWriter, r *http.Request) {
	// Streaming transcription - ONLY local, refuse cloud requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Acquire concurrency slot with configured streaming timeout
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(h.StreamingTimeout)*time.Second)
	defer cancel()

	if err := h.ConcurrencyMgr.AcquireSlot(ctx); err != nil {
		if err == context.DeadlineExceeded {
			http.Error(w, "Request timeout waiting for available slot", http.StatusServiceUnavailable)
		} else {
			// Queue full (429)
			w.Header().Set("Retry-After", "30")
			http.Error(w, err.Error(), http.StatusTooManyRequests)
		}
		return
	}
	defer h.ConcurrencyMgr.ReleaseSlot()

	startTime := time.Now()

	// Check sensitive flag - streaming MUST be local only
	if !isSensitiveRequest(r) {
		http.Error(w, "Streaming transcription only available locally (OpenAI API does not support streaming). Set sensitive=true or remove X-Sensitive header.", http.StatusBadRequest)
		return
	}

	// Parse multipart form (max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Get the audio file
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing or invalid file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read file content
	fileContent, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Get optional parameters
	model := r.FormValue("model")
	language := r.FormValue("language")

	// Prepare log entry
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "stt",
		Provider:    "local",
		Model:       "whisper-local-stream",
		Sensitive:   true, // Streaming is always local/sensitive
		InputChars:  len(fileContent),
		ClientIP:    getClientIP(r),
	}

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "Streaming not supported"
		h.Logger.LogRequest(logEntry)
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Forward to local whisper server streaming endpoint
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to create form file"
		h.Logger.LogRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to create form file\"}\n\n")
		flusher.Flush()
		return
	}
	if _, err := part.Write(fileContent); err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to write file content"
		h.Logger.LogRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to write file content\"}\n\n")
		flusher.Flush()
		return
	}

	if model != "" {
		writer.WriteField("model", model)
	}
	if language != "" {
		writer.WriteField("language", language)
	}
	writer.Close()

	// Make streaming request to local whisper server
	whisperURL := h.WhisperServerURL + "/v1/audio/transcriptions/stream"
	req, err := http.NewRequest("POST", whisperURL, &buf)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = "failed to create request"
		h.Logger.LogRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"failed to create request\"}\n\n")
		flusher.Flush()
		return
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: time.Duration(h.StreamingTimeout) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("local whisper stream request failed: %s", err.Error())
		h.Logger.LogRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"local whisper stream request failed: %s\"}\n\n", err.Error())
		flusher.Flush()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("local whisper error %d: %s", resp.StatusCode, string(respBody))
		h.Logger.LogRequest(logEntry)
		fmt.Fprintf(w, "data: {\"error\": \"local whisper error %d: %s\"}\n\n", resp.StatusCode, string(respBody))
		flusher.Flush()
		return
	}

	// Stream the response back to client
	buf2 := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf2)
		if n > 0 {
			w.Write(buf2[:n])
			flusher.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(w, "data: {\"error\": \"stream read error: %s\"}\n\n", err.Error())
			flusher.Flush()
			break
		}
	}

	// Log successful stream completion
	logEntry.LatencyMs = time.Since(startTime).Milliseconds()
	logEntry.Success = true
	h.Logger.LogRequest(logEntry)
}

// callLocalWhisper sends a transcription request to the local whisper server.
func (h *STTHandler) callLocalWhisper(fileContent []byte, filename, model, language, prompt string) (*domain.WhisperTranscriptionResponse, error) {
	// Create multipart form for local whisper server
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(fileContent); err != nil {
		return nil, fmt.Errorf("failed to write file content: %w", err)
	}

	// Add optional fields
	if model != "" {
		writer.WriteField("model", model)
	}
	if language != "" {
		writer.WriteField("language", language)
	}
	if prompt != "" {
		writer.WriteField("prompt", prompt)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Make request to local whisper server
	url := h.WhisperServerURL + "/v1/audio/transcriptions"
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: time.Duration(h.Timeout) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("local whisper request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("local whisper error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.WhisperTranscriptionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

// callOpenAIWhisper sends a transcription request to OpenAI's Whisper API.
func (h *STTHandler) callOpenAIWhisper(fileContent []byte, filename, model, language, prompt string) (*domain.WhisperTranscriptionResponse, error) {
	if h.OpenAIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}

	// Create multipart form for OpenAI
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(fileContent); err != nil {
		return nil, fmt.Errorf("failed to write file content: %w", err)
	}

	// Add model (default to whisper-1)
	if model == "" {
		model = "whisper-1"
	}
	writer.WriteField("model", model)

	// Add optional language
	if language != "" {
		writer.WriteField("language", language)
	}

	// Add optional prompt for context hints
	if prompt != "" {
		writer.WriteField("prompt", prompt)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Make request to OpenAI
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/transcriptions", &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+h.OpenAIKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: time.Duration(h.Timeout) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenAI whisper request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI whisper error %d: %s", resp.StatusCode, string(respBody))
	}

	var result domain.WhisperTranscriptionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

// isSensitiveRequest checks if the request should be routed locally (sensitive data).
func isSensitiveRequest(r *http.Request) bool {
	// Check header first
	if s := r.Header.Get("X-Sensitive"); s != "" {
		return s != "false"
	}
	// Check form value
	if s := r.FormValue("sensitive"); s != "" {
		return s != "false"
	}
	// Default to sensitive (local) for safety
	return true
}
