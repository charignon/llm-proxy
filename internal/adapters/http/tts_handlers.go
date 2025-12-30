// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"llm-proxy/internal/adapters/audiocache"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// TTSVoiceMap maps OpenAI voice names to Kokoro voices.
var TTSVoiceMap = map[string]string{
	"alloy":   "af_nicole", // American female
	"echo":    "am_adam",   // American male
	"fable":   "bf_emma",   // British female
	"onyx":    "bm_george", // British male
	"nova":    "af_sky",    // American female
	"shimmer": "af_bella",  // American female
}

// TTSHandler handles text-to-speech requests.
type TTSHandler struct {
	TTSServerURL string
	OpenAIAPIKey string
	Logger       ports.RequestLogger
	CheckBudget  func(provider string) error
	AudioCache   ports.AudioCache
	CacheTTL     time.Duration
	Timeout        int // Timeout in seconds for OpenAI TTS
	KokoroTimeout  int // Timeout in seconds for Kokoro TTS
}

// NewTTSHandler creates a new TTS handler.
func NewTTSHandler(ttsServerURL string, openaiAPIKey string, logger ports.RequestLogger, timeout, kokoroTimeout int) *TTSHandler {
	return &TTSHandler{
		TTSServerURL: ttsServerURL,
		OpenAIAPIKey: openaiAPIKey,
		Logger:       logger,
		CacheTTL:     7 * 24 * time.Hour, // Default 7 day TTL
		Timeout:       timeout,
		KokoroTimeout: kokoroTimeout,
	}
}

// HandleTTS handles POST /v1/audio/speech requests (OpenAI format).
func (h *TTSHandler) HandleTTS(w http.ResponseWriter, r *http.Request) {
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

	var req domain.TTSRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Input == "" {
		http.Error(w, "Missing required field: input", http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.Voice == "" {
		req.Voice = "alloy"
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "mp3"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}

	// For Kokoro (non-OpenAI), map the voice first for cache key
	effectiveVoice := req.Voice
	if !strings.HasPrefix(req.Model, "tts-1") {
		if mappedVoice, ok := TTSVoiceMap[req.Voice]; ok {
			effectiveVoice = mappedVoice
		}
	}

	// Generate cache key and check cache
	cacheKey := audiocache.GenerateCacheKey(req.Input, effectiveVoice, req.Speed, req.ResponseFormat)

	if h.AudioCache != nil {
		if audio, contentType, found := h.AudioCache.Get(cacheKey); found {
			// Log cache hit
			logEntry := &domain.RequestLog{
				Timestamp:   startTime,
				RequestType: "tts",
				Provider:    "cache",
				Model:       "tts-cache",
				Voice:       effectiveVoice,
				InputChars:  len(req.Input),
				Cached:      true,
				Success:     true,
				LatencyMs:   time.Since(startTime).Milliseconds(),
				ClientIP:    getClientIP(r),
			}
			h.Logger.LogRequest(logEntry)

			log.Printf("TTS cache HIT (%dms): voice=%s, len=%d chars, key=%s",
				time.Since(startTime).Milliseconds(), effectiveVoice, len(req.Input), cacheKey[:16])

			w.Header().Set("Content-Type", contentType)
			w.Header().Set("X-Cache", "HIT")
			w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=\"speech.%s\"", req.ResponseFormat))
			w.Write(audio)
			return
		}
	}

	// Check if this should go to OpenAI TTS
	if strings.HasPrefix(req.Model, "tts-1") {
		h.handleOpenAITTS(w, r, req, body, startTime, cacheKey)
		return
	}

	// Use Kokoro for everything else
	h.handleKokoroTTS(w, r, req, body, startTime, cacheKey)
}

// handleOpenAITTS forwards the request to OpenAI's TTS API.
func (h *TTSHandler) handleOpenAITTS(w http.ResponseWriter, r *http.Request, req domain.TTSRequest, body []byte, startTime time.Time, cacheKey string) {
	// Check budget before processing request
	if h.CheckBudget != nil {
		if err := h.CheckBudget("openai"); err != nil {
			logEntry := &domain.RequestLog{
				Timestamp:   startTime,
				RequestType: "tts",
				Provider:    "openai",
				Model:       req.Model,
				Success:     false,
				Error:       err.Error(),
				LatencyMs:   time.Since(startTime).Milliseconds(),
				ClientIP:    getClientIP(r),
			}
			h.Logger.LogRequest(logEntry)

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

	// Calculate cost: tts-1 = $0.015/1K chars, tts-1-hd = $0.030/1K chars
	costPer1K := 0.015
	if req.Model == "tts-1-hd" {
		costPer1K = 0.030
	}
	costUSD := float64(len(req.Input)) / 1000.0 * costPer1K

	// Prepare log entry
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "tts",
		Provider:    "openai",
		Model:       req.Model,
		Voice:       req.Voice,
		InputChars:  len(req.Input),
		CostUSD:     costUSD,
		RequestBody: body,
		ClientIP:    getClientIP(r),
	}

	// Forward to OpenAI
	client := &http.Client{Timeout: time.Duration(h.Timeout) * time.Second}
	openaiReq, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/speech", bytes.NewReader(body))
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)
		http.Error(w, "Failed to create request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	openaiReq.Header.Set("Content-Type", "application/json")
	openaiReq.Header.Set("Authorization", "Bearer "+h.OpenAIAPIKey)

	resp, err := client.Do(openaiReq)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)

		log.Printf("OpenAI TTS request failed: %v", err)
		http.Error(w, "OpenAI TTS unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		h.Logger.LogRequest(logEntry)

		log.Printf("OpenAI TTS error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, fmt.Sprintf("OpenAI TTS error: %s", string(respBody)), resp.StatusCode)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true
	h.Logger.LogRequest(logEntry)

	log.Printf("OpenAI TTS complete (%dms): model=%s voice=%s, len=%d chars", latencyMs, req.Model, req.Voice, len(req.Input))

	// Read audio response into buffer for caching
	audioBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read OpenAI TTS response: %v", err)
		http.Error(w, "Failed to read audio response", http.StatusInternalServerError)
		return
	}

	// Determine content type
	contentType := "audio/mpeg"
	if req.ResponseFormat == "wav" {
		contentType = "audio/wav"
	} else if req.ResponseFormat == "opus" {
		contentType = "audio/opus"
	} else if req.ResponseFormat == "aac" {
		contentType = "audio/aac"
	} else if req.ResponseFormat == "flac" {
		contentType = "audio/flac"
	}

	// Store in cache
	if h.AudioCache != nil && cacheKey != "" {
		if err := h.AudioCache.Set(cacheKey, audioBytes, contentType, h.CacheTTL); err != nil {
			log.Printf("Failed to cache TTS audio: %v", err)
		} else {
			log.Printf("TTS audio cached: key=%s, size=%d bytes", cacheKey[:16], len(audioBytes))
		}
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Cache", "MISS")
	w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=\"speech.%s\"", req.ResponseFormat))
	w.Write(audioBytes)
}

// handleKokoroTTS forwards the request to local Kokoro TTS.
func (h *TTSHandler) handleKokoroTTS(w http.ResponseWriter, r *http.Request, req domain.TTSRequest, body []byte, startTime time.Time, cacheKey string) {
	// Map OpenAI voice to Kokoro voice
	kokoroVoice, ok := TTSVoiceMap[req.Voice]
	if !ok {
		// If not in map, use directly (allows passing Kokoro voice names)
		kokoroVoice = req.Voice
	}

	// Build Kokoro TTS server URL with query params
	ttsURL := fmt.Sprintf("%s/tts?text=%s&voice=%s&format=%s&speed=%.2f",
		h.TTSServerURL,
		url.QueryEscape(req.Input),
		kokoroVoice,
		req.ResponseFormat,
		req.Speed,
	)

	// Prepare log entry
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "tts",
		Provider:    "kokoro",
		Model:       "kokoro-tts",
		Voice:       kokoroVoice,
		InputChars:  len(req.Input),
		RequestBody: body,
		ClientIP:    getClientIP(r),
	}

	// Forward request to Kokoro TTS server
	client := &http.Client{Timeout: time.Duration(h.KokoroTimeout) * time.Second}
	resp, err := client.Get(ttsURL)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)

		log.Printf("TTS request failed: %v", err)
		http.Error(w, "TTS server unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		h.Logger.LogRequest(logEntry)

		log.Printf("TTS error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, fmt.Sprintf("TTS error: %s", string(respBody)), resp.StatusCode)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true
	h.Logger.LogRequest(logEntry)

	log.Printf("Kokoro TTS complete (%dms): voice=%s, len=%d chars", latencyMs, kokoroVoice, len(req.Input))

	// Read audio response into buffer for caching
	audioBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read Kokoro TTS response: %v", err)
		http.Error(w, "Failed to read audio response", http.StatusInternalServerError)
		return
	}

	// Determine content type
	contentType := "audio/mpeg"
	if req.ResponseFormat == "wav" {
		contentType = "audio/wav"
	}

	// Store in cache
	if h.AudioCache != nil && cacheKey != "" {
		if err := h.AudioCache.Set(cacheKey, audioBytes, contentType, h.CacheTTL); err != nil {
			log.Printf("Failed to cache TTS audio: %v", err)
		} else {
			log.Printf("TTS audio cached: key=%s, size=%d bytes", cacheKey[:16], len(audioBytes))
		}
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Cache", "MISS")
	w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=\"speech.%s\"", req.ResponseFormat))
	w.Write(audioBytes)
}

// TTSCompatRequest is the voice_cloning format (different from OpenAI's TTSRequest).
type TTSCompatRequest struct {
	Text  string  `json:"text"`
	Voice string  `json:"voice"`
	Speed float64 `json:"speed"`
}

// HandleTTSCompat handles the voice_cloning API format (/tts endpoint).
// This allows tts.lan to route through llm-proxy for logging while maintaining
// compatibility with existing clients that use the voice_cloning format.
func (h *TTSHandler) HandleTTSCompat(w http.ResponseWriter, r *http.Request) {
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

	var req TTSCompatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Text == "" {
		http.Error(w, "Missing required field: text", http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.Voice == "" {
		req.Voice = "af_nicole"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}

	// Generate cache key and check cache (mp3 is the default format for compat)
	cacheKey := audiocache.GenerateCacheKey(req.Text, req.Voice, req.Speed, "mp3")

	if h.AudioCache != nil {
		if audio, contentType, found := h.AudioCache.Get(cacheKey); found {
			// Log cache hit
			logEntry := &domain.RequestLog{
				Timestamp:   startTime,
				RequestType: "tts",
				Provider:    "cache",
				Model:       "tts-cache",
				Voice:       req.Voice,
				InputChars:  len(req.Text),
				Cached:      true,
				Success:     true,
				LatencyMs:   time.Since(startTime).Milliseconds(),
				ClientIP:    getClientIP(r),
			}
			h.Logger.LogRequest(logEntry)

			log.Printf("TTS compat cache HIT (%dms): voice=%s, len=%d chars, key=%s",
				time.Since(startTime).Milliseconds(), req.Voice, len(req.Text), cacheKey[:16])

			w.Header().Set("Content-Type", contentType)
			w.Header().Set("X-Cache", "HIT")
			w.Write(audio)
			return
		}
	}

	// Build Kokoro TTS server URL with query params (mp3 is default format)
	ttsURL := fmt.Sprintf("%s/tts?text=%s&voice=%s&format=mp3&speed=%.2f",
		h.TTSServerURL,
		url.QueryEscape(req.Text),
		req.Voice,
		req.Speed,
	)

	// Prepare log entry
	logEntry := &domain.RequestLog{
		Timestamp:   startTime,
		RequestType: "tts",
		Provider:    "kokoro",
		Model:       "kokoro-tts",
		Voice:       req.Voice,
		InputChars:  len(req.Text),
		RequestBody: body,
		ClientIP:    getClientIP(r),
	}

	// Forward request to Kokoro TTS server
	client := &http.Client{Timeout: time.Duration(h.KokoroTimeout) * time.Second}
	resp, err := client.Get(ttsURL)
	if err != nil {
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = err.Error()
		h.Logger.LogRequest(logEntry)

		log.Printf("TTS compat request failed: %v", err)
		http.Error(w, "TTS server unavailable: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		logEntry.LatencyMs = time.Since(startTime).Milliseconds()
		logEntry.Success = false
		logEntry.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody))
		h.Logger.LogRequest(logEntry)

		log.Printf("TTS compat error %d: %s", resp.StatusCode, string(respBody))
		http.Error(w, fmt.Sprintf("TTS error: %s", string(respBody)), resp.StatusCode)
		return
	}

	latencyMs := time.Since(startTime).Milliseconds()
	logEntry.LatencyMs = latencyMs
	logEntry.Success = true
	h.Logger.LogRequest(logEntry)

	log.Printf("TTS compat complete (%dms): voice=%s, len=%d chars", latencyMs, req.Voice, len(req.Text))

	// Read audio response into buffer for caching
	audioBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read TTS compat response: %v", err)
		http.Error(w, "Failed to read audio response", http.StatusInternalServerError)
		return
	}

	// Store in cache
	if h.AudioCache != nil && cacheKey != "" {
		if err := h.AudioCache.Set(cacheKey, audioBytes, "audio/mpeg", h.CacheTTL); err != nil {
			log.Printf("Failed to cache TTS compat audio: %v", err)
		} else {
			log.Printf("TTS compat audio cached: key=%s, size=%d bytes", cacheKey[:16], len(audioBytes))
		}
	}

	w.Header().Set("Content-Type", "audio/mpeg")
	w.Header().Set("X-Cache", "MISS")
	w.Write(audioBytes)
}
