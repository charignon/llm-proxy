package http

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// QueueEntry represents a request waiting in the queue
type QueueEntry struct {
	ID        string    `json:"id"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	Usecase   string    `json:"usecase"`
	HasImages bool      `json:"has_images"`
	EnqueuedAt time.Time `json:"enqueued_at"`
	Status    string    `json:"status"` // "queued" or "running"
}

// ProviderQueueStats holds stats for a single provider
type ProviderQueueStats struct {
	Provider     string       `json:"provider"`
	MaxConcurrent int         `json:"max_concurrent"`
	Running      int          `json:"running"`
	Queued       int          `json:"queued"`
	TotalServed  int64        `json:"total_served"`
	TotalQueued  int64        `json:"total_queued"`
	Entries      []QueueEntry `json:"entries"`
}

// QueueStats holds the full queue state
type QueueStats struct {
	Providers []ProviderQueueStats `json:"providers"`
}

// providerQueue manages concurrency for a single provider
type providerQueue struct {
	name          string
	maxConcurrent int
	sem           chan struct{}
	mu            sync.Mutex
	entries       map[string]*QueueEntry
	totalServed   atomic.Int64
	totalQueued   atomic.Int64
	counter       atomic.Int64
}

func newProviderQueue(name string, maxConcurrent int) *providerQueue {
	pq := &providerQueue{
		name:          name,
		maxConcurrent: maxConcurrent,
		sem:           make(chan struct{}, maxConcurrent),
		entries:       make(map[string]*QueueEntry),
	}
	// Fill semaphore with available slots
	for i := 0; i < maxConcurrent; i++ {
		pq.sem <- struct{}{}
	}
	return pq
}

func (pq *providerQueue) acquire(ctx context.Context, model, usecase string, hasImages bool) (string, error) {
	id := fmt.Sprintf("%s-%d", pq.name, pq.counter.Add(1))

	entry := &QueueEntry{
		ID:         id,
		Provider:   pq.name,
		Model:      model,
		Usecase:    usecase,
		HasImages:  hasImages,
		EnqueuedAt: time.Now(),
		Status:     "queued",
	}

	pq.mu.Lock()
	pq.entries[id] = entry
	pq.mu.Unlock()
	pq.totalQueued.Add(1)

	// Try to acquire semaphore slot
	select {
	case <-pq.sem:
		pq.mu.Lock()
		entry.Status = "running"
		pq.mu.Unlock()
		pq.totalServed.Add(1)
		return id, nil
	case <-ctx.Done():
		pq.mu.Lock()
		delete(pq.entries, id)
		pq.mu.Unlock()
		return "", ctx.Err()
	}
}

func (pq *providerQueue) release(id string) {
	pq.mu.Lock()
	delete(pq.entries, id)
	pq.mu.Unlock()

	// Return slot to semaphore
	select {
	case pq.sem <- struct{}{}:
	default:
	}
}

func (pq *providerQueue) stats() ProviderQueueStats {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	stats := ProviderQueueStats{
		Provider:      pq.name,
		MaxConcurrent: pq.maxConcurrent,
		TotalServed:   pq.totalServed.Load(),
		TotalQueued:   pq.totalQueued.Load(),
	}

	for _, e := range pq.entries {
		stats.Entries = append(stats.Entries, *e)
		if e.Status == "running" {
			stats.Running++
		} else {
			stats.Queued++
		}
	}
	return stats
}

// RequestQueue manages per-provider concurrency queues
type RequestQueue struct {
	mu     sync.RWMutex
	queues map[string]*providerQueue

	// Default concurrency limits per provider type
	defaults map[string]int
}

// NewRequestQueue creates a new request queue with per-provider limits
func NewRequestQueue(providerLimits map[string]int) *RequestQueue {
	rq := &RequestQueue{
		queues:   make(map[string]*providerQueue),
		defaults: providerLimits,
	}
	for name, limit := range providerLimits {
		rq.queues[name] = newProviderQueue(name, limit)
	}
	return rq
}

// Acquire waits for a slot on the provider's queue.
func (rq *RequestQueue) Acquire(ctx context.Context, provider, model, usecase string, hasImages bool) (string, error) {
	pq := rq.getOrCreate(provider)
	return pq.acquire(ctx, model, usecase, hasImages)
}

// Release returns a slot to the provider's queue
func (rq *RequestQueue) Release(provider, model, id string) {
	rq.mu.RLock()
	pq, ok := rq.queues[provider]
	rq.mu.RUnlock()
	if ok {
		pq.release(id)
	}
}

// Stats returns the current queue state for all providers
func (rq *RequestQueue) Stats() QueueStats {
	rq.mu.RLock()
	defer rq.mu.RUnlock()

	var stats QueueStats
	for _, pq := range rq.queues {
		s := pq.stats()
		stats.Providers = append(stats.Providers, s)
	}
	return stats
}


func (rq *RequestQueue) getOrCreate(provider string) *providerQueue {
	rq.mu.RLock()
	pq, ok := rq.queues[provider]
	rq.mu.RUnlock()
	if ok {
		return pq
	}

	rq.mu.Lock()
	defer rq.mu.Unlock()

	// Double-check after acquiring write lock
	if pq, ok := rq.queues[provider]; ok {
		return pq
	}

	limit := 2
	if l, ok := rq.defaults[provider]; ok {
		limit = l
	}
	pq = newProviderQueue(provider, limit)
	rq.queues[provider] = pq
	return pq
}
