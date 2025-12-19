package providers

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"

	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// LoadBalancedProvider distributes requests across multiple backend providers.
type LoadBalancedProvider struct {
	name      string
	providers []ports.ChatProvider
	counter   uint64 // For round-robin
	mu        sync.RWMutex
}

// NewLoadBalancedProvider creates a load balancer wrapping multiple providers.
func NewLoadBalancedProvider(name string, providers ...ports.ChatProvider) *LoadBalancedProvider {
	return &LoadBalancedProvider{
		name:      name,
		providers: providers,
	}
}

// Chat implements ChatProvider.Chat using round-robin load balancing.
// Falls back to next provider if one fails.
func (lb *LoadBalancedProvider) Chat(req *domain.ChatCompletionRequest, model string) (*domain.ChatCompletionResponse, error) {
	lb.mu.RLock()
	count := len(lb.providers)
	lb.mu.RUnlock()

	if count == 0 {
		return nil, fmt.Errorf("no providers available in load balancer %s", lb.name)
	}

	// Get current index and increment for next call (round-robin)
	idx := int(atomic.AddUint64(&lb.counter, 1) - 1)

	// Try each provider in round-robin order, starting from idx
	var lastErr error
	for i := 0; i < count; i++ {
		providerIdx := (idx + i) % count

		lb.mu.RLock()
		provider := lb.providers[providerIdx]
		lb.mu.RUnlock()

		// Try this provider
		resp, err := provider.Chat(req, model)
		if err == nil {
			log.Printf("[LoadBalancer:%s] Request served by backend %d", lb.name, providerIdx)
			return resp, nil
		}

		lastErr = err
		log.Printf("[LoadBalancer:%s] Backend %d failed: %v, trying next", lb.name, providerIdx, err)
	}

	return nil, fmt.Errorf("all %d providers failed in %s, last error: %v", count, lb.name, lastErr)
}

// AddProvider adds a new provider to the load balancer.
func (lb *LoadBalancedProvider) AddProvider(p ports.ChatProvider) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	lb.providers = append(lb.providers, p)
	log.Printf("[LoadBalancer:%s] Added provider, total: %d", lb.name, len(lb.providers))
}

// RemoveProvider removes a provider from the load balancer by index.
func (lb *LoadBalancedProvider) RemoveProvider(idx int) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	if idx >= 0 && idx < len(lb.providers) {
		lb.providers = append(lb.providers[:idx], lb.providers[idx+1:]...)
		log.Printf("[LoadBalancer:%s] Removed provider %d, remaining: %d", lb.name, idx, len(lb.providers))
	}
}

// ProviderCount returns the number of providers in the balancer.
func (lb *LoadBalancedProvider) ProviderCount() int {
	lb.mu.RLock()
	defer lb.mu.RUnlock()
	return len(lb.providers)
}
