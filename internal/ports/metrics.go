// Package ports defines interfaces (ports) for the hexagonal architecture.
package ports

// MetricsRecorder records request metrics for monitoring and observability.
type MetricsRecorder interface {
	// RecordRequest records metrics for a completed request.
	RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool)
}
