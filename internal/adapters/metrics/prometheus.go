// Package metrics provides metrics collection and recording adapters.
package metrics

import (
	"fmt"
	"sync"
)

// PrometheusMetrics collects metrics for Prometheus-style monitoring.
// It implements the ports.MetricsRecorder interface.
type PrometheusMetrics struct {
	RequestsTotal map[string]int64 // provider|model|status -> count
	TokensTotal   map[string]int64 // provider|model|direction -> count
	DurationSumMs map[string]int64 // provider|model -> sum of ms
	DurationCount map[string]int64 // provider|model -> count
	CostTotal     float64
	CacheHits     int64
	CacheMisses   int64
	mutex         sync.RWMutex
}

// NewPrometheusMetrics creates a new PrometheusMetrics instance.
func NewPrometheusMetrics() *PrometheusMetrics {
	return &PrometheusMetrics{
		RequestsTotal: make(map[string]int64),
		TokensTotal:   make(map[string]int64),
		DurationSumMs: make(map[string]int64),
		DurationCount: make(map[string]int64),
	}
}

// RecordRequest records metrics for a completed request.
// It implements the ports.MetricsRecorder interface.
func (m *PrometheusMetrics) RecordRequest(provider, model, status string, durationMs int64, inputTokens, outputTokens int, cost float64, cached bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Use | as separator since model names can contain :
	key := fmt.Sprintf("%s|%s|%s", provider, model, status)
	m.RequestsTotal[key]++

	if status == "success" {
		durKey := fmt.Sprintf("%s|%s", provider, model)
		m.DurationSumMs[durKey] += durationMs
		m.DurationCount[durKey]++

		inputKey := fmt.Sprintf("%s|%s|input", provider, model)
		outputKey := fmt.Sprintf("%s|%s|output", provider, model)
		m.TokensTotal[inputKey] += int64(inputTokens)
		m.TokensTotal[outputKey] += int64(outputTokens)

		m.CostTotal += cost
	}

	if cached {
		m.CacheHits++
	} else {
		m.CacheMisses++
	}
}

// GetRequestsTotal returns a copy of the requests total map.
func (m *PrometheusMetrics) GetRequestsTotal() map[string]int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	result := make(map[string]int64)
	for k, v := range m.RequestsTotal {
		result[k] = v
	}
	return result
}

// GetTokensTotal returns a copy of the tokens total map.
func (m *PrometheusMetrics) GetTokensTotal() map[string]int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	result := make(map[string]int64)
	for k, v := range m.TokensTotal {
		result[k] = v
	}
	return result
}

// GetDurationSumMs returns a copy of the duration sum map.
func (m *PrometheusMetrics) GetDurationSumMs() map[string]int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	result := make(map[string]int64)
	for k, v := range m.DurationSumMs {
		result[k] = v
	}
	return result
}

// GetDurationCount returns a copy of the duration count map.
func (m *PrometheusMetrics) GetDurationCount() map[string]int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	result := make(map[string]int64)
	for k, v := range m.DurationCount {
		result[k] = v
	}
	return result
}

// GetCostTotal returns the total cost.
func (m *PrometheusMetrics) GetCostTotal() float64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.CostTotal
}

// GetCacheHits returns the cache hit count.
func (m *PrometheusMetrics) GetCacheHits() int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.CacheHits
}

// GetCacheMisses returns the cache miss count.
func (m *PrometheusMetrics) GetCacheMisses() int64 {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.CacheMisses
}
