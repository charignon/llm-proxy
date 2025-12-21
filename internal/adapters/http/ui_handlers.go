// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"net/http"
	"strings"

	"llm-proxy/web"
)

// UIHandler handles serving HTML templates for the web UI.
type UIHandler struct{}

// NewUIHandler creates a new UI handler.
func NewUIHandler() *UIHandler {
	return &UIHandler{}
}

// HandleAnalyticsPage serves the analytics page.
func (h *UIHandler) HandleAnalyticsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.AnalyticsTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

// HandleStatsPage serves the stats page.
func (h *UIHandler) HandleStatsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.StatsTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

// HandleDashboard serves the main dashboard page.
func (h *UIHandler) HandleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.DashboardTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

// HandleRequestPage handles /request/{id} URLs and redirects to /?request={id}.
func (h *UIHandler) HandleRequestPage(w http.ResponseWriter, r *http.Request) {
	// Extract ID from path like /request/123
	path := strings.TrimPrefix(r.URL.Path, "/request/")
	if path == "" || path == r.URL.Path {
		http.Error(w, "Request ID required", http.StatusBadRequest)
		return
	}
	// Redirect to dashboard with request param
	http.Redirect(w, r, "/?request="+path, http.StatusFound)
}

// HandleBudgetsPage serves the budgets management page.
func (h *UIHandler) HandleBudgetsPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.BudgetsTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}

// HandleTestPlayground serves the test playground page.
func (h *UIHandler) HandleTestPlayground(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	data, err := web.Templates.ReadFile(web.PlaygroundTemplate)
	if err != nil {
		http.Error(w, "Failed to load template", http.StatusInternalServerError)
		return
	}
	w.Write(data)
}
