// Package app contains application services that orchestrate domain logic.
package app

import (
	"fmt"
	"strings"
	"sync"

	"llm-proxy/internal/domain"
)

// Router handles routing decisions for chat completion requests.
type Router struct {
	textRoutes     map[string]map[string]*domain.RouteConfig // sensitive -> precision -> config
	visionRoutes   map[string]map[string]*domain.RouteConfig // sensitive -> precision -> config
	imageGenRoutes map[string]map[string]*domain.RouteConfig // sensitive -> precision -> config

	// Usecase overrides: usecase -> type -> sensitive -> precision -> config
	usecaseRoutes map[string]map[string]map[string]map[string]*domain.RouteConfig
	usecaseMutex  sync.RWMutex
}

// NewRouter creates a new router with the provided routing tables.
func NewRouter(
	textRoutes map[string]map[string]*domain.RouteConfig,
	visionRoutes map[string]map[string]*domain.RouteConfig,
) *Router {
	return &Router{
		textRoutes:    textRoutes,
		visionRoutes:  visionRoutes,
		usecaseRoutes: make(map[string]map[string]map[string]map[string]*domain.RouteConfig),
	}
}

// ResolveRoute determines the provider and model for a chat completion request.
func (r *Router) ResolveRoute(req *domain.ChatCompletionRequest) (*domain.RouteConfig, error) {
	// If model is explicitly specified (not a routing keyword), use it directly
	if req.Model != "" && req.Model != "auto" && req.Model != "route" {
		return r.resolveExplicitModel(req.Model), nil
	}

	// Use routing table - choose text or vision based on content
	// Default to sensitive=true (local Ollama) for privacy
	sensitive := "true"
	if req.Sensitive != nil && !*req.Sensitive {
		sensitive = "false"
	}

	precision := req.Precision
	if precision == "" {
		precision = "medium"
	}

	// Determine route type
	routeType := "text"
	if req.HasImages() {
		routeType = "vision"
	}

	// Check for usecase-specific override first
	if req.Usecase != "" {
		if override := r.GetUsecaseRoute(req.Usecase, routeType, sensitive, precision); override != nil {
			return override, nil
		}
	}

	// Select appropriate base routing table
	var selectedTable map[string]map[string]*domain.RouteConfig
	if routeType == "vision" {
		selectedTable = r.visionRoutes
	} else {
		selectedTable = r.textRoutes
	}

	routes, ok := selectedTable[sensitive]
	if !ok {
		return nil, fmt.Errorf("invalid sensitive value")
	}

	route, ok := routes[precision]
	if !ok {
		return nil, fmt.Errorf("invalid precision value: %s", precision)
	}

	if route == nil {
		return nil, fmt.Errorf("capability not available: sensitive=%s, precision=%s", sensitive, precision)
	}

	return route, nil
}

// resolveExplicitModel determines the provider for an explicitly specified model.
func (r *Router) resolveExplicitModel(model string) *domain.RouteConfig {
	provider := "openai"
	if strings.HasPrefix(model, "ollama/") {
		provider = "ollama"
		model = strings.TrimPrefix(model, "ollama/")
	} else if strings.HasPrefix(model, "claude") {
		provider = "anthropic"
	} else if strings.Contains(model, ":") || model == "llama3" || model == "llava" || strings.HasPrefix(model, "qwen") {
		provider = "ollama"
	}
	// Codex models are OpenAI (already default, but explicit for clarity)
	// gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini, etc.
	return &domain.RouteConfig{Provider: provider, Model: model}
}

// GetUsecaseRoute retrieves a usecase-specific route override.
func (r *Router) GetUsecaseRoute(usecase, routeType, sensitive, precision string) *domain.RouteConfig {
	r.usecaseMutex.RLock()
	defer r.usecaseMutex.RUnlock()

	if r.usecaseRoutes[usecase] != nil &&
		r.usecaseRoutes[usecase][routeType] != nil &&
		r.usecaseRoutes[usecase][routeType][sensitive] != nil {
		return r.usecaseRoutes[usecase][routeType][sensitive][precision]
	}
	return nil
}

// SetUsecaseRoute sets a usecase-specific route override.
func (r *Router) SetUsecaseRoute(usecase, routeType string, sensitive bool, precision, provider, model string) {
	r.usecaseMutex.Lock()
	defer r.usecaseMutex.Unlock()

	sensitiveStr := "false"
	if sensitive {
		sensitiveStr = "true"
	}

	if r.usecaseRoutes[usecase] == nil {
		r.usecaseRoutes[usecase] = make(map[string]map[string]map[string]*domain.RouteConfig)
	}
	if r.usecaseRoutes[usecase][routeType] == nil {
		r.usecaseRoutes[usecase][routeType] = make(map[string]map[string]*domain.RouteConfig)
	}
	if r.usecaseRoutes[usecase][routeType][sensitiveStr] == nil {
		r.usecaseRoutes[usecase][routeType][sensitiveStr] = make(map[string]*domain.RouteConfig)
	}

	r.usecaseRoutes[usecase][routeType][sensitiveStr][precision] = &domain.RouteConfig{
		Provider: provider,
		Model:    model,
	}
}

// DeleteUsecaseRoute removes a usecase-specific route override.
func (r *Router) DeleteUsecaseRoute(usecase, routeType string, sensitive bool, precision string) {
	r.usecaseMutex.Lock()
	defer r.usecaseMutex.Unlock()

	sensitiveStr := "false"
	if sensitive {
		sensitiveStr = "true"
	}

	if r.usecaseRoutes[usecase] != nil &&
		r.usecaseRoutes[usecase][routeType] != nil &&
		r.usecaseRoutes[usecase][routeType][sensitiveStr] != nil {
		delete(r.usecaseRoutes[usecase][routeType][sensitiveStr], precision)
	}
}

// LoadUsecaseRoutes loads usecase routes from a map (typically from database).
func (r *Router) LoadUsecaseRoutes(routes map[string]map[string]map[string]map[string]*domain.RouteConfig) {
	r.usecaseMutex.Lock()
	defer r.usecaseMutex.Unlock()
	r.usecaseRoutes = routes
}

// GetAllUsecaseRoutes returns a copy of all usecase routes.
func (r *Router) GetAllUsecaseRoutes() map[string]map[string]map[string]map[string]*domain.RouteConfig {
	r.usecaseMutex.RLock()
	defer r.usecaseMutex.RUnlock()
	// Return shallow copy
	return r.usecaseRoutes
}

// GetTextRoutes returns the text routing table.
func (r *Router) GetTextRoutes() map[string]map[string]*domain.RouteConfig {
	return r.textRoutes
}

// GetVisionRoutes returns the vision routing table.
func (r *Router) GetVisionRoutes() map[string]map[string]*domain.RouteConfig {
	return r.visionRoutes
}

// SetImageGenRoutes sets the image generation routing table.
func (r *Router) SetImageGenRoutes(routes map[string]map[string]*domain.RouteConfig) {
	r.imageGenRoutes = routes
}

// GetImageGenRoutes returns the image generation routing table.
func (r *Router) GetImageGenRoutes() map[string]map[string]*domain.RouteConfig {
	return r.imageGenRoutes
}

// ResolveImageGenRoute determines the provider and model for an image generation request.
func (r *Router) ResolveImageGenRoute(req *domain.ImageGenerationRequest) (*domain.RouteConfig, error) {
	// If model is explicitly specified (not a routing keyword), use it directly
	if req.Model != "" && req.Model != "auto" && req.Model != "route" {
		// For image generation, only OpenAI DALL-E models are supported
		return &domain.RouteConfig{Provider: "openai", Model: req.Model}, nil
	}

	// Default to sensitive=true for privacy
	sensitive := "true"
	if req.Sensitive != nil && !*req.Sensitive {
		sensitive = "false"
	}

	// Sensitive requests not supported for image generation (no local provider)
	if sensitive == "true" {
		return nil, fmt.Errorf("image generation not available for sensitive requests")
	}

	precision := req.Precision
	if precision == "" {
		precision = "medium"
	}

	// Check for usecase-specific override first
	if req.Usecase != "" {
		if override := r.GetUsecaseRoute(req.Usecase, "image_gen", sensitive, precision); override != nil {
			return override, nil
		}
	}

	// Use image generation routing table if available
	if r.imageGenRoutes != nil {
		routes, ok := r.imageGenRoutes[sensitive]
		if ok {
			route, ok := routes[precision]
			if ok && route != nil {
				return route, nil
			}
		}
	}

	// Default to DALL-E 3
	return &domain.RouteConfig{Provider: "openai", Model: "dall-e-3"}, nil
}
