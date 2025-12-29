// Package audiocache provides a file-based audio cache implementation.
package audiocache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FileAudioCache implements a file-based cache for TTS audio responses.
type FileAudioCache struct {
	baseDir   string
	maxSize   int64         // Maximum cache size in bytes
	ttl       time.Duration // Default TTL for cache entries
	mu        sync.RWMutex
	hits      int64
	misses    int64
	stopClean chan struct{}
}

// cacheMetadata stores metadata about a cached audio file.
type cacheMetadata struct {
	ContentType string    `json:"content_type"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	Size        int64     `json:"size"`
}

// NewFileAudioCache creates a new file-based audio cache.
func NewFileAudioCache(baseDir string, maxSize int64, ttl time.Duration) (*FileAudioCache, error) {
	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	cache := &FileAudioCache{
		baseDir:   baseDir,
		maxSize:   maxSize,
		ttl:       ttl,
		stopClean: make(chan struct{}),
	}

	// Start background cleanup goroutine
	go cache.cleanupLoop()

	log.Printf("TTS audio cache initialized: dir=%s, maxSize=%d MB, ttl=%s",
		baseDir, maxSize/(1024*1024), ttl)

	return cache, nil
}

// GenerateCacheKey creates a deterministic cache key from TTS parameters.
func GenerateCacheKey(text, voice string, speed float64, format string) string {
	// Normalize text: trim whitespace, lowercase for case-insensitive matching
	normalizedText := strings.TrimSpace(strings.ToLower(text))

	// Create composite key from all parameters
	keyData := fmt.Sprintf("%s:%s:%.2f:%s", normalizedText, voice, speed, format)

	// Hash for filename safety
	hash := sha256.Sum256([]byte(keyData))
	return hex.EncodeToString(hash[:])
}

// keyToPath converts a cache key to file paths (audio and metadata).
func (c *FileAudioCache) keyToPath(key string) (audioPath, metaPath string) {
	// Use first 2 chars as subdirectory to avoid too many files in one dir
	subdir := key[:2]
	dir := filepath.Join(c.baseDir, subdir)
	audioPath = filepath.Join(dir, key+".audio")
	metaPath = filepath.Join(dir, key+".meta")
	return
}

// Get retrieves cached audio by key.
func (c *FileAudioCache) Get(key string) ([]byte, string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	audioPath, metaPath := c.keyToPath(key)

	// Read metadata first
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		atomic.AddInt64(&c.misses, 1)
		return nil, "", false
	}

	var meta cacheMetadata
	if err := json.Unmarshal(metaData, &meta); err != nil {
		atomic.AddInt64(&c.misses, 1)
		return nil, "", false
	}

	// Check if expired
	if time.Now().After(meta.ExpiresAt) {
		atomic.AddInt64(&c.misses, 1)
		// Don't delete here - let cleanup goroutine handle it
		return nil, "", false
	}

	// Read audio file
	audio, err := os.ReadFile(audioPath)
	if err != nil {
		atomic.AddInt64(&c.misses, 1)
		return nil, "", false
	}

	atomic.AddInt64(&c.hits, 1)
	return audio, meta.ContentType, true
}

// Set stores audio in the cache.
func (c *FileAudioCache) Set(key string, audio []byte, contentType string, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	audioPath, metaPath := c.keyToPath(key)

	// Create subdirectory if needed
	dir := filepath.Dir(audioPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create cache subdirectory: %w", err)
	}

	// Use provided TTL or default
	if ttl == 0 {
		ttl = c.ttl
	}

	// Write audio file
	if err := os.WriteFile(audioPath, audio, 0644); err != nil {
		return fmt.Errorf("failed to write audio file: %w", err)
	}

	// Write metadata
	meta := cacheMetadata{
		ContentType: contentType,
		CreatedAt:   time.Now(),
		ExpiresAt:   time.Now().Add(ttl),
		Size:        int64(len(audio)),
	}

	metaData, err := json.Marshal(meta)
	if err != nil {
		os.Remove(audioPath) // Clean up audio file
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	if err := os.WriteFile(metaPath, metaData, 0644); err != nil {
		os.Remove(audioPath) // Clean up audio file
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// Delete removes a specific entry from the cache.
func (c *FileAudioCache) Delete(key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	audioPath, metaPath := c.keyToPath(key)
	os.Remove(audioPath)
	os.Remove(metaPath)
	return nil
}

// Clear removes all entries from the cache.
func (c *FileAudioCache) Clear() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove all subdirectories
	entries, err := os.ReadDir(c.baseDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			os.RemoveAll(filepath.Join(c.baseDir, entry.Name()))
		}
	}

	atomic.StoreInt64(&c.hits, 0)
	atomic.StoreInt64(&c.misses, 0)

	return nil
}

// Stats returns cache statistics.
func (c *FileAudioCache) Stats() (hits int64, misses int64, sizeBytes int64) {
	hits = atomic.LoadInt64(&c.hits)
	misses = atomic.LoadInt64(&c.misses)
	sizeBytes = c.calculateSize()
	return
}

// calculateSize computes the total size of cached files.
func (c *FileAudioCache) calculateSize() int64 {
	var totalSize int64

	filepath.Walk(c.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(path, ".audio") {
			totalSize += info.Size()
		}
		return nil
	})

	return totalSize
}

// cacheEntry represents a cached file for sorting/eviction.
type cacheEntry struct {
	key       string
	expiresAt time.Time
	size      int64
}

// cleanupLoop periodically cleans up expired entries and enforces size limits.
func (c *FileAudioCache) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanup()
		case <-c.stopClean:
			return
		}
	}
}

// cleanup removes expired entries and enforces size limits.
func (c *FileAudioCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	var entries []cacheEntry
	var expiredCount, evictedCount int
	now := time.Now()

	// Collect all entries
	filepath.Walk(c.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".meta") {
			return nil
		}

		metaData, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		var meta cacheMetadata
		if err := json.Unmarshal(metaData, &meta); err != nil {
			return nil
		}

		key := strings.TrimSuffix(filepath.Base(path), ".meta")
		subdir := filepath.Base(filepath.Dir(path))
		fullKey := subdir + "/" + key

		// Remove expired entries immediately
		if now.After(meta.ExpiresAt) {
			audioPath := strings.TrimSuffix(path, ".meta") + ".audio"
			os.Remove(path)
			os.Remove(audioPath)
			expiredCount++
			return nil
		}

		entries = append(entries, cacheEntry{
			key:       fullKey,
			expiresAt: meta.ExpiresAt,
			size:      meta.Size,
		})

		return nil
	})

	// Check if we need to evict for size
	var totalSize int64
	for _, e := range entries {
		totalSize += e.size
	}

	if totalSize > c.maxSize {
		// Sort by expiration time (oldest first)
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].expiresAt.Before(entries[j].expiresAt)
		})

		// Evict until under limit
		for _, e := range entries {
			if totalSize <= c.maxSize {
				break
			}

			parts := strings.Split(e.key, "/")
			if len(parts) == 2 {
				dir := filepath.Join(c.baseDir, parts[0])
				audioPath := filepath.Join(dir, parts[1]+".audio")
				metaPath := filepath.Join(dir, parts[1]+".meta")
				os.Remove(audioPath)
				os.Remove(metaPath)
				totalSize -= e.size
				evictedCount++
			}
		}
	}

	if expiredCount > 0 || evictedCount > 0 {
		log.Printf("TTS cache cleanup: expired=%d, evicted=%d, remaining_size=%d MB",
			expiredCount, evictedCount, totalSize/(1024*1024))
	}
}

// Stop stops the cleanup goroutine.
func (c *FileAudioCache) Stop() {
	close(c.stopClean)
}
