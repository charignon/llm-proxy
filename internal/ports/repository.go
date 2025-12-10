package ports

import "llm-proxy/internal/domain"

// RequestLogger is a secondary port for persisting request logs.
// This is a minimal interface for the core logging functionality.
type RequestLogger interface {
	// LogRequest stores a request log entry and returns its ID.
	LogRequest(entry *domain.RequestLog) int64
}
