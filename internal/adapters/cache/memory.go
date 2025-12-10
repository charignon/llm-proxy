// Package cache provides cache adapter implementations.
package cache

import (
	"sync"
	"time"

	"llm-proxy/internal/domain"
)

// MemoryCache implements Cache port with an in-memory store.
type MemoryCache struct {
	entries  map[string]*domain.CacheEntry
	mutex    sync.RWMutex
	ttlHours int
}

// NewMemoryCache creates a new in-memory cache with the specified TTL in hours.
func NewMemoryCache(ttlHours int) *MemoryCache {
	return &MemoryCache{
		entries:  make(map[string]*domain.CacheEntry),
		ttlHours: ttlHours,
	}
}

// Get retrieves a cached response by key.
func (c *MemoryCache) Get(key string) ([]byte, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, ok := c.entries[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(c.ttlHours)*time.Hour {
		return nil, false
	}

	// Don't return entries with nil/empty responses (e.g., from errors)
	if entry.Response == nil || len(entry.Response) == 0 {
		return nil, false
	}

	return entry.Response, true
}

// Set stores a request/response pair in the cache.
func (c *MemoryCache) Set(key string, request, response []byte) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.entries[key] = &domain.CacheEntry{
		Request:   request,
		Response:  response,
		CreatedAt: time.Now(),
	}
}

// GetRequest retrieves the original request body for a cache key.
func (c *MemoryCache) GetRequest(key string) ([]byte, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, ok := c.entries[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(c.ttlHours)*time.Hour {
		return nil, false
	}

	return entry.Request, true
}

// GetResponse retrieves just the response for a cache key.
func (c *MemoryCache) GetResponse(key string) ([]byte, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, ok := c.entries[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.CreatedAt) > time.Duration(c.ttlHours)*time.Hour {
		return nil, false
	}

	return entry.Response, true
}

// Clear removes all entries from the cache.
func (c *MemoryCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.entries = make(map[string]*domain.CacheEntry)
}
