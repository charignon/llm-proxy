// Package loadmanager provides dynamic concurrency management based on system load.
package loadmanager

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
)

// LoadMetrics represents current system metrics
type LoadMetrics struct {
	CPUPercent      float64
	MemPercent      float64
	MemFreeGB       float64
	AllowedConcurrent int
}

// ConcurrencyManager manages dynamic concurrency limits based on system load
type ConcurrencyManager struct {
	maxConcurrent      int           // Absolute max concurrent requests
	maxQueueSize       int           // Max pending requests before rejecting
	semaphore          chan struct{} // Semaphore for limiting concurrent requests
	queue              chan struct{} // Queue for pending requests
	metrics            LoadMetrics
	metricsLock        sync.RWMutex
	stopChan           chan struct{}
	wg                 sync.WaitGroup
	currentConcurrent  int
	concurrentLock     sync.Mutex
	totalRequests      int64
	totalRejections    int64
	rejectionLock      sync.Mutex
}

// NewConcurrencyManager creates a new concurrency manager
func NewConcurrencyManager(maxConcurrent, maxQueueSize int) *ConcurrencyManager {
	cm := &ConcurrencyManager{
		maxConcurrent:     maxConcurrent,
		maxQueueSize:      maxQueueSize,
		semaphore:         make(chan struct{}, maxConcurrent),
		queue:             make(chan struct{}, maxQueueSize),
		stopChan:          make(chan struct{}),
		currentConcurrent: 0,
		totalRequests:     0,
		totalRejections:   0,
	}

	// Start load monitor
	cm.wg.Add(1)
	go cm.monitorLoad()

	return cm
}

// monitorLoad periodically checks system load and adjusts concurrency
func (cm *ConcurrencyManager) monitorLoad() {
	defer cm.wg.Done()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-cm.stopChan:
			return
		case <-ticker.C:
			cm.updateLoad()
		}
	}
}

// updateLoad reads system metrics and adjusts concurrency limits
func (cm *ConcurrencyManager) updateLoad() {
	// Get CPU percent (1 second sample)
	cpuPercent, err := cpu.Percent(time.Second, false)
	cpuUsage := 0.0
	if err == nil && len(cpuPercent) > 0 {
		cpuUsage = cpuPercent[0]
	}

	// Get memory stats
	memStats, err := mem.VirtualMemory()
	memUsage := 0.0
	memFreeGB := 0.0
	if err == nil {
		memUsage = memStats.UsedPercent
		memFreeGB = float64(memStats.Available) / (1024 * 1024 * 1024)
	}

	// Calculate allowed concurrent requests based on load
	allowed := cm.calculateAllowedConcurrent(cpuUsage, memUsage, memFreeGB)

	// Update metrics
	cm.metricsLock.Lock()
	cm.metrics.CPUPercent = cpuUsage
	cm.metrics.MemPercent = memUsage
	cm.metrics.MemFreeGB = memFreeGB
	cm.metrics.AllowedConcurrent = allowed
	cm.metricsLock.Unlock()

	// Adjust semaphore size if needed
	cm.adjustSemaphore(allowed)
}

// calculateAllowedConcurrent determines safe concurrency based on system load
func (cm *ConcurrencyManager) calculateAllowedConcurrent(cpuPercent, memPercent, memFreeGB float64) int {
	// Start with a baseline based on CPU usage
	var allowed int

	// CPU-based tiers (M2 Ultra with 12 cores)
	switch {
	case cpuPercent < 30:
		allowed = 5 // Light load: be aggressive
	case cpuPercent < 50:
		allowed = 4 // Moderate load
	case cpuPercent < 70:
		allowed = 3 // Heavy load
	case cpuPercent < 85:
		allowed = 2 // Very heavy load
	default:
		allowed = 1 // Critical load: minimal concurrency
	}

	// Memory adjustments (192GB available on M2 Ultra)
	switch {
	case memFreeGB < 10:
		// Less than 10GB free: reduce significantly
		allowed = 1
	case memFreeGB < 20:
		// Less than 20GB free: reduce by 2
		allowed = allowed - 2
		if allowed < 1 {
			allowed = 1
		}
	case memFreeGB < 40:
		// Less than 40GB free: reduce by 1
		allowed = allowed - 1
		if allowed < 1 {
			allowed = 1
		}
	case memPercent > 85:
		// More than 85% used: be conservative
		allowed = allowed - 1
		if allowed < 1 {
			allowed = 1
		}
	}

	// Never exceed absolute maximum
	if allowed > cm.maxConcurrent {
		allowed = cm.maxConcurrent
	}

	return allowed
}

// adjustSemaphore resizes the semaphore to the allowed limit
func (cm *ConcurrencyManager) adjustSemaphore(newSize int) {
	cm.concurrentLock.Lock()
	currentSize := len(cm.semaphore)
	cm.concurrentLock.Unlock()

	if newSize == currentSize {
		return // No change needed
	}

	if newSize > currentSize {
		// Add slots
		for i := 0; i < newSize-currentSize; i++ {
			cm.semaphore <- struct{}{}
		}
	} else if newSize < currentSize {
		// Remove slots (only if not in use)
		for i := 0; i < currentSize-newSize; i++ {
			select {
			case <-cm.semaphore:
				// Successfully removed a slot
			default:
				// Semaphore busy, can't reduce now
				break
			}
		}
	}
}

// AcquireSlot attempts to acquire a concurrency slot, with optional queuing
func (cm *ConcurrencyManager) AcquireSlot(ctx context.Context) error {
	cm.recordRequest()

	// Try to acquire immediately
	select {
	case <-cm.semaphore:
		cm.concurrentLock.Lock()
		cm.currentConcurrent++
		cm.concurrentLock.Unlock()
		return nil
	default:
		// No immediate slot available, try queuing
	}

	// Try to queue the request
	select {
	case <-cm.queue:
		// Got a queue slot, now wait for a semaphore slot
		select {
		case <-cm.semaphore:
			cm.concurrentLock.Lock()
			cm.currentConcurrent++
			cm.concurrentLock.Unlock()
			return nil
		case <-ctx.Done():
			// Context cancelled, release queue slot
			cm.queue <- struct{}{}
			return ctx.Err()
		}
	default:
		// Queue is full, reject
		cm.recordRejection()
		return ErrQueueFull
	}
}

// ReleaseSlot releases a concurrency slot
func (cm *ConcurrencyManager) ReleaseSlot() {
	cm.concurrentLock.Lock()
	cm.currentConcurrent--
	if cm.currentConcurrent < 0 {
		cm.currentConcurrent = 0
	}
	cm.concurrentLock.Unlock()

	// Release semaphore slot
	select {
	case cm.semaphore <- struct{}{}:
	default:
	}

	// Release queue slot if any are queued
	select {
	case cm.queue <- struct{}{}:
	default:
	}
}

// recordRequest increments request counter
func (cm *ConcurrencyManager) recordRequest() {
	cm.rejectionLock.Lock()
	cm.totalRequests++
	cm.rejectionLock.Unlock()
}

// recordRejection increments rejection counter
func (cm *ConcurrencyManager) recordRejection() {
	cm.rejectionLock.Lock()
	cm.totalRejections++
	cm.rejectionLock.Unlock()
}

// GetMetrics returns current load metrics
func (cm *ConcurrencyManager) GetMetrics() LoadMetrics {
	cm.metricsLock.RLock()
	defer cm.metricsLock.RUnlock()
	return cm.metrics
}

// GetStats returns concurrency statistics
func (cm *ConcurrencyManager) GetStats() map[string]interface{} {
	cm.rejectionLock.Lock()
	defer cm.rejectionLock.Unlock()

	cm.concurrentLock.Lock()
	current := cm.currentConcurrent
	cm.concurrentLock.Unlock()

	return map[string]interface{}{
		"current_concurrent": current,
		"total_requests":     cm.totalRequests,
		"total_rejections":   cm.totalRejections,
		"queue_pending":      len(cm.queue),
		"rejection_rate":     float64(cm.totalRejections) / float64(cm.totalRequests+1),
	}
}

// Stop gracefully stops the load manager
func (cm *ConcurrencyManager) Stop() {
	close(cm.stopChan)
	cm.wg.Wait()
	log.Printf("Concurrency manager stopped")
}
