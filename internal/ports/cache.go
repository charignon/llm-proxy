package ports

import "time"

// Cache is a secondary port for caching request/response pairs.
type Cache interface {
	// Get retrieves a cached response by key, returning (response, found).
	Get(key string) ([]byte, bool)

	// Set stores a request/response pair in the cache.
	Set(key string, request, response []byte)

	// GetRequest retrieves the original request body for a cache key.
	GetRequest(key string) ([]byte, bool)

	// GetResponse retrieves just the response for a cache key.
	GetResponse(key string) ([]byte, bool)

	// Clear removes all entries from the cache.
	Clear()
}

// AudioCache is a secondary port for caching TTS audio responses.
type AudioCache interface {
	// Get retrieves cached audio by key, returning (audio, contentType, found).
	Get(key string) ([]byte, string, bool)

	// Set stores audio in the cache with the specified TTL.
	Set(key string, audio []byte, contentType string, ttl time.Duration) error

	// Delete removes a specific entry from the cache.
	Delete(key string) error

	// Clear removes all entries from the cache.
	Clear() error

	// Stats returns cache statistics (hits, misses, sizeBytes).
	Stats() (hits int64, misses int64, sizeBytes int64)
}
