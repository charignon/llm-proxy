// Package repository provides database adapter implementations.
package repository

import (
	"database/sql"
	"log"
	"sync"
	"time"

	"llm-proxy/internal/domain"
)

// SQLiteLogger implements RequestLogger port with SQLite storage.
type SQLiteLogger struct {
	db    *sql.DB
	mutex sync.Mutex
}

// NewSQLiteLogger creates a new SQLite request logger.
func NewSQLiteLogger(db *sql.DB) *SQLiteLogger {
	return &SQLiteLogger{db: db}
}

// LogRequest stores a request log entry and returns its ID.
func (r *SQLiteLogger) LogRequest(entry *domain.RequestLog) int64 {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	// Default request type to "llm" if not set
	if entry.RequestType == "" {
		entry.RequestType = "llm"
	}

	result, err := r.db.Exec(`
		INSERT INTO requests (timestamp, request_type, provider, model, requested_model, sensitive, precision, usecase, cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images, request_body, response_body, voice, audio_duration_ms, input_chars, is_replay)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, entry.Timestamp.Format(time.RFC3339), entry.RequestType, entry.Provider, entry.Model, entry.RequestedModel,
		entry.Sensitive, entry.Precision, entry.Usecase, entry.Cached, entry.InputTokens, entry.OutputTokens,
		entry.LatencyMs, entry.CostUSD, entry.Success, entry.Error, entry.CacheKey, entry.HasImages,
		string(entry.RequestBody), string(entry.ResponseBody), entry.Voice, entry.AudioDurationMs, entry.InputChars, entry.IsReplay)

	if err != nil {
		log.Printf("Failed to log request: %v", err)
		return 0
	}

	id, err := result.LastInsertId()
	if err != nil {
		log.Printf("Failed to get last insert ID: %v", err)
		return 0
	}
	return id
}
