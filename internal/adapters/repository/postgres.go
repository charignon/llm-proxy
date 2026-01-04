// Package repository provides database adapter implementations.
package repository

import (
	"database/sql"
	"log"
	"sync"

	_ "github.com/lib/pq" // PostgreSQL driver

	"llm-proxy/internal/domain"
)

// PostgresLogger implements RequestLogger port with PostgreSQL storage.
// It writes request logs to PostgreSQL for centralized analytics while
// SQLite remains the primary local store.
type PostgresLogger struct {
	db    *sql.DB
	mutex sync.Mutex
}

// NewPostgresLogger creates a new PostgreSQL request logger.
// It initializes the schema if needed.
func NewPostgresLogger(connStr string) (*PostgresLogger, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, err
	}

	logger := &PostgresLogger{db: db}

	// Initialize schema
	if err := logger.initSchema(); err != nil {
		db.Close()
		return nil, err
	}

	return logger, nil
}

// initSchema creates the requests table if it doesn't exist.
func (r *PostgresLogger) initSchema() error {
	// Create requests table with PostgreSQL types
	_, err := r.db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id BIGSERIAL PRIMARY KEY,
			timestamp TIMESTAMPTZ NOT NULL,
			request_type TEXT DEFAULT 'llm',
			provider TEXT NOT NULL,
			model TEXT NOT NULL,
			requested_model TEXT,
			sensitive BOOLEAN DEFAULT FALSE,
			precision TEXT,
			usecase TEXT,
			cached BOOLEAN DEFAULT FALSE,
			input_tokens INTEGER DEFAULT 0,
			output_tokens INTEGER DEFAULT 0,
			latency_ms INTEGER DEFAULT 0,
			cost_usd DOUBLE PRECISION DEFAULT 0,
			success BOOLEAN DEFAULT TRUE,
			error TEXT,
			cache_key TEXT,
			has_images BOOLEAN DEFAULT FALSE,
			request_body TEXT,
			response_body TEXT,
			voice TEXT,
			audio_duration_ms INTEGER DEFAULT 0,
			input_chars INTEGER DEFAULT 0,
			is_replay BOOLEAN DEFAULT FALSE,
			client_ip TEXT
		)
	`)
	if err != nil {
		return err
	}

	// Create indexes for common queries
	indexes := []string{
		`CREATE INDEX IF NOT EXISTS idx_pg_timestamp ON requests(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_provider ON requests(provider)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_model ON requests(model)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_request_type ON requests(request_type)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_client_ip ON requests(client_ip)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_cached ON requests(cached)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_usecase ON requests(usecase)`,
		`CREATE INDEX IF NOT EXISTS idx_pg_usecase_sensitive_precision ON requests(usecase, sensitive, precision)`,
	}

	for _, idx := range indexes {
		if _, err := r.db.Exec(idx); err != nil {
			log.Printf("Warning: failed to create index: %v", err)
		}
	}

	return nil
}

// LogRequest stores a request log entry and returns its ID.
func (r *PostgresLogger) LogRequest(entry *domain.RequestLog) int64 {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	// Default request type to "llm" if not set
	if entry.RequestType == "" {
		entry.RequestType = "llm"
	}

	var id int64
	err := r.db.QueryRow(`
		INSERT INTO requests (
			timestamp, request_type, provider, model, requested_model,
			sensitive, precision, usecase, cached, input_tokens, output_tokens,
			latency_ms, cost_usd, success, error, cache_key, has_images,
			request_body, response_body, voice, audio_duration_ms, input_chars,
			is_replay, client_ip
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
		RETURNING id
	`, entry.Timestamp, entry.RequestType, entry.Provider, entry.Model, entry.RequestedModel,
		entry.Sensitive, entry.Precision, entry.Usecase, entry.Cached, entry.InputTokens, entry.OutputTokens,
		entry.LatencyMs, entry.CostUSD, entry.Success, entry.Error, entry.CacheKey, entry.HasImages,
		string(entry.RequestBody), string(entry.ResponseBody), entry.Voice, entry.AudioDurationMs, entry.InputChars,
		entry.IsReplay, entry.ClientIP).Scan(&id)

	if err != nil {
		log.Printf("PostgreSQL: Failed to log request: %v", err)
		return 0
	}

	return id
}

// Close closes the database connection.
func (r *PostgresLogger) Close() error {
	return r.db.Close()
}
