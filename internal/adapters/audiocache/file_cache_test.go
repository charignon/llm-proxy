package audiocache

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGenerateCacheKey(t *testing.T) {
	// Same input should produce same key
	key1 := GenerateCacheKey("Hello world", "af_nicole", 1.0, "mp3")
	key2 := GenerateCacheKey("Hello world", "af_nicole", 1.0, "mp3")
	if key1 != key2 {
		t.Errorf("Same input should produce same key: %s != %s", key1, key2)
	}

	// Different text should produce different key
	key3 := GenerateCacheKey("Goodbye world", "af_nicole", 1.0, "mp3")
	if key1 == key3 {
		t.Errorf("Different text should produce different key")
	}

	// Different voice should produce different key
	key4 := GenerateCacheKey("Hello world", "am_adam", 1.0, "mp3")
	if key1 == key4 {
		t.Errorf("Different voice should produce different key")
	}

	// Different speed should produce different key
	key5 := GenerateCacheKey("Hello world", "af_nicole", 1.5, "mp3")
	if key1 == key5 {
		t.Errorf("Different speed should produce different key")
	}

	// Case insensitive text
	key6 := GenerateCacheKey("HELLO WORLD", "af_nicole", 1.0, "mp3")
	if key1 != key6 {
		t.Errorf("Text should be case insensitive: %s != %s", key1, key6)
	}

	// Trimmed whitespace
	key7 := GenerateCacheKey("  Hello world  ", "af_nicole", 1.0, "mp3")
	if key1 != key7 {
		t.Errorf("Whitespace should be trimmed: %s != %s", key1, key7)
	}
}

func TestFileAudioCache(t *testing.T) {
	// Create temp directory for cache
	tmpDir, err := os.MkdirTemp("", "tts-cache-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create cache
	cache, err := NewFileAudioCache(tmpDir, 10*1024*1024, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Stop()

	// Test cache miss
	key := GenerateCacheKey("test audio", "af_nicole", 1.0, "mp3")
	audio, contentType, found := cache.Get(key)
	if found {
		t.Error("Expected cache miss for empty cache")
	}

	// Test set and get
	testAudio := []byte("fake audio data for testing")
	testContentType := "audio/mpeg"

	err = cache.Set(key, testAudio, testContentType, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}

	audio, contentType, found = cache.Get(key)
	if !found {
		t.Error("Expected cache hit after set")
	}
	if string(audio) != string(testAudio) {
		t.Errorf("Audio mismatch: got %s, want %s", string(audio), string(testAudio))
	}
	if contentType != testContentType {
		t.Errorf("Content type mismatch: got %s, want %s", contentType, testContentType)
	}

	// Check stats
	hits, misses, _ := cache.Stats()
	if hits != 1 {
		t.Errorf("Expected 1 hit, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("Expected 1 miss, got %d", misses)
	}

	// Test delete
	err = cache.Delete(key)
	if err != nil {
		t.Fatalf("Failed to delete cache entry: %v", err)
	}

	_, _, found = cache.Get(key)
	if found {
		t.Error("Expected cache miss after delete")
	}

	// Test clear
	cache.Set(key, testAudio, testContentType, 1*time.Hour)
	err = cache.Clear()
	if err != nil {
		t.Fatalf("Failed to clear cache: %v", err)
	}

	_, _, found = cache.Get(key)
	if found {
		t.Error("Expected cache miss after clear")
	}
}

func TestFileAudioCacheExpiration(t *testing.T) {
	// Create temp directory for cache
	tmpDir, err := os.MkdirTemp("", "tts-cache-expiry-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create cache with very short TTL
	cache, err := NewFileAudioCache(tmpDir, 10*1024*1024, 1*time.Millisecond)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Stop()

	key := GenerateCacheKey("expiry test", "af_nicole", 1.0, "mp3")
	testAudio := []byte("will expire")

	err = cache.Set(key, testAudio, "audio/mpeg", 1*time.Millisecond)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}

	// Wait for expiration
	time.Sleep(10 * time.Millisecond)

	_, _, found := cache.Get(key)
	if found {
		t.Error("Expected cache miss for expired entry")
	}
}

func TestFileAudioCacheSubdirectories(t *testing.T) {
	// Create temp directory for cache
	tmpDir, err := os.MkdirTemp("", "tts-cache-subdir-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := NewFileAudioCache(tmpDir, 10*1024*1024, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Stop()

	// Set an entry
	key := GenerateCacheKey("subdir test", "af_nicole", 1.0, "mp3")
	err = cache.Set(key, []byte("data"), "audio/mpeg", 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}

	// Check that subdirectory was created (first 2 chars of key)
	subdir := filepath.Join(tmpDir, key[:2])
	if _, err := os.Stat(subdir); os.IsNotExist(err) {
		t.Errorf("Expected subdirectory %s to exist", subdir)
	}

	// Check that audio and meta files exist
	audioFile := filepath.Join(subdir, key+".audio")
	metaFile := filepath.Join(subdir, key+".meta")

	if _, err := os.Stat(audioFile); os.IsNotExist(err) {
		t.Errorf("Expected audio file %s to exist", audioFile)
	}
	if _, err := os.Stat(metaFile); os.IsNotExist(err) {
		t.Errorf("Expected meta file %s to exist", metaFile)
	}
}
