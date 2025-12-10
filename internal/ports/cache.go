package ports

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
