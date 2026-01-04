// Package repository provides database adapter implementations.
package repository

import (
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// MultiLogger implements RequestLogger by writing to multiple loggers.
// The primary logger's ID is returned; secondary loggers run in the background.
type MultiLogger struct {
	primary    ports.RequestLogger
	secondaries []ports.RequestLogger
}

// NewMultiLogger creates a logger that writes to multiple destinations.
// The first logger is primary (its ID is returned), the rest are secondary.
func NewMultiLogger(primary ports.RequestLogger, secondaries ...ports.RequestLogger) *MultiLogger {
	return &MultiLogger{
		primary:    primary,
		secondaries: secondaries,
	}
}

// LogRequest logs to all configured loggers.
// Returns the ID from the primary logger.
func (m *MultiLogger) LogRequest(entry *domain.RequestLog) int64 {
	// Log to primary synchronously
	id := m.primary.LogRequest(entry)

	// Log to secondaries asynchronously to avoid latency
	for _, logger := range m.secondaries {
		go logger.LogRequest(entry)
	}

	return id
}
