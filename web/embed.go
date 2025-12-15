// Package web provides embedded web templates for the LLM Proxy dashboard.
package web

import (
	"embed"
)

// Templates contains all embedded HTML templates.
//
//go:embed templates/*.html
var Templates embed.FS

// Template names for accessing embedded files.
const (
	AnalyticsTemplate  = "templates/analytics.html"
	StatsTemplate      = "templates/stats.html"
	DashboardTemplate  = "templates/dashboard.html"
	PlaygroundTemplate = "templates/playground.html"
)
