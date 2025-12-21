// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"llm-proxy/internal/adapters/budget"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// HistoryHandler handles request history and stats API endpoints.
type HistoryHandler struct {
	DB             *sql.DB
	DBMutex        *sync.Mutex
	Cache          ports.Cache
	GetPending     func() []*domain.PendingRequest
	ReplayHelper   func(w http.ResponseWriter, r *http.Request, reqBody []byte, modelOverride string)
	BudgetRepo     ports.BudgetRepository
}

// RequestHistoryEntry represents a request in the history list.
type RequestHistoryEntry struct {
	ID              int64    `json:"id"`
	Timestamp       string   `json:"timestamp"`
	RequestType     string   `json:"request_type"`
	Provider        string   `json:"provider"`
	Model           string   `json:"model"`
	Sensitive       bool     `json:"sensitive"`
	Precision       string   `json:"precision,omitempty"`
	Usecase         string   `json:"usecase,omitempty"`
	HasImages       bool     `json:"has_images"`
	LatencyMs       int64    `json:"latency_ms"`
	CostUSD         float64  `json:"cost_usd"`
	Success         bool     `json:"success"`
	CacheKey        string   `json:"cache_key,omitempty"`
	InputTokens     int      `json:"input_tokens"`
	OutputTokens    int      `json:"output_tokens"`
	IsReplay        bool     `json:"is_replay,omitempty"`
	Voice           string   `json:"voice,omitempty"`
	AudioDurationMs int64    `json:"audio_duration_ms,omitempty"`
	InputChars      int      `json:"input_chars,omitempty"`
	ClientIP        string   `json:"client_ip,omitempty"`
	Tools           []string `json:"tools,omitempty"`
	HasWebSearch    bool     `json:"has_web_search,omitempty"`
}

// RequestHistoryResponse is the response for request history.
type RequestHistoryResponse struct {
	Requests []RequestHistoryEntry `json:"requests"`
	Total    int                   `json:"total"`
	Page     int                   `json:"page"`
	PageSize int                   `json:"page_size"`
}

// NewHistoryHandler creates a new history handler.
func NewHistoryHandler(db *sql.DB, dbMutex *sync.Mutex, cache ports.Cache) *HistoryHandler {
	return &HistoryHandler{
		DB:      db,
		DBMutex: dbMutex,
		Cache:   cache,
	}
}

// HandleRequestHistory handles GET /api/history requests.
func (h *HistoryHandler) HandleRequestHistory(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	pageSize := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &pageSize)
	}

	page := 1
	if p := r.URL.Query().Get("page"); p != "" {
		fmt.Sscanf(p, "%d", &page)
		if page < 1 {
			page = 1
		}
	}
	offset := (page - 1) * pageSize

	// Build WHERE conditions (shared between count and data queries)
	var conditions []string
	var filterArgs []interface{}

	// Filter by request_type if specified (can be comma-separated like "llm,tts")
	if typeFilter := r.URL.Query().Get("type"); typeFilter != "" {
		types := strings.Split(typeFilter, ",")
		placeholders := make([]string, len(types))
		for i, t := range types {
			placeholders[i] = "?"
			filterArgs = append(filterArgs, strings.TrimSpace(t))
		}
		conditions = append(conditions, "request_type IN ("+strings.Join(placeholders, ",")+")")
	}

	// Filter by usecase if specified
	if usecaseFilter := r.URL.Query().Get("usecase"); usecaseFilter != "" {
		conditions = append(conditions, "usecase = ?")
		filterArgs = append(filterArgs, usecaseFilter)
	}

	// Exclude usecases if specified (comma-separated list)
	if excludeUsecases := r.URL.Query().Get("exclude_usecases"); excludeUsecases != "" {
		usecases := strings.Split(excludeUsecases, ",")
		placeholders := make([]string, len(usecases))
		for i, u := range usecases {
			placeholders[i] = "?"
			filterArgs = append(filterArgs, strings.TrimSpace(u))
		}
		conditions = append(conditions, "usecase NOT IN ("+strings.Join(placeholders, ",")+")")
	}

	// Filter by model if specified
	if modelFilter := r.URL.Query().Get("model"); modelFilter != "" {
		conditions = append(conditions, "model = ?")
		filterArgs = append(filterArgs, modelFilter)
	}

	// Filter by sensitive if specified
	if sensitiveFilter := r.URL.Query().Get("sensitive"); sensitiveFilter != "" {
		conditions = append(conditions, "sensitive = ?")
		if sensitiveFilter == "1" || sensitiveFilter == "true" {
			filterArgs = append(filterArgs, 1)
		} else {
			filterArgs = append(filterArgs, 0)
		}
	}

	// Filter by precision if specified
	if precisionFilter := r.URL.Query().Get("precision"); precisionFilter != "" {
		conditions = append(conditions, "precision = ?")
		filterArgs = append(filterArgs, precisionFilter)
	}

	// Filter by client_ip if specified
	if clientIPFilter := r.URL.Query().Get("client_ip"); clientIPFilter != "" {
		conditions = append(conditions, "client_ip = ?")
		filterArgs = append(filterArgs, clientIPFilter)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = " WHERE " + strings.Join(conditions, " AND ")
	}

	// Get total count for pagination
	countQuery := "SELECT COUNT(*) FROM requests" + whereClause
	var total int
	if err := h.DB.QueryRow(countQuery, filterArgs...).Scan(&total); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Build data query
	query := `
		SELECT id, timestamp, request_type, provider, model, sensitive, precision, usecase, has_images,
		       latency_ms, cost_usd, success, cache_key, input_tokens, output_tokens, is_replay,
		       voice, audio_duration_ms, input_chars, client_ip, request_body
		FROM requests
	` + whereClause + " ORDER BY id DESC LIMIT ? OFFSET ?"

	args := append(filterArgs, pageSize, offset)

	rows, err := h.DB.Query(query, args...)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var history []RequestHistoryEntry
	for rows.Next() {
		var entry RequestHistoryEntry
		var requestType sql.NullString
		var precision sql.NullString
		var usecase sql.NullString
		var cacheKey sql.NullString
		var voice sql.NullString
		var isReplay sql.NullBool
		var clientIP sql.NullString
		var requestBody sql.NullString
		rows.Scan(&entry.ID, &entry.Timestamp, &requestType, &entry.Provider, &entry.Model,
			&entry.Sensitive, &precision, &usecase, &entry.HasImages, &entry.LatencyMs,
			&entry.CostUSD, &entry.Success, &cacheKey, &entry.InputTokens, &entry.OutputTokens,
			&isReplay, &voice, &entry.AudioDurationMs, &entry.InputChars, &clientIP, &requestBody)
		if requestType.Valid {
			entry.RequestType = requestType.String
		} else {
			entry.RequestType = "llm" // Default for old entries
		}
		if precision.Valid {
			entry.Precision = precision.String
		}
		if usecase.Valid {
			entry.Usecase = usecase.String
		}
		if cacheKey.Valid {
			entry.CacheKey = cacheKey.String
		}
		if voice.Valid {
			entry.Voice = voice.String
		}
		if isReplay.Valid {
			entry.IsReplay = isReplay.Bool
		}
		if clientIP.Valid {
			entry.ClientIP = clientIP.String
		}
		// Extract tool names from request body
		if requestBody.Valid && requestBody.String != "" {
			entry.Tools, entry.HasWebSearch = extractToolsFromRequestBody(requestBody.String)
		}
		history = append(history, entry)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(RequestHistoryResponse{
		Requests: history,
		Total:    total,
		Page:     page,
		PageSize: pageSize,
	})
}

// extractToolsFromRequestBody extracts tool names from a request body JSON string.
func extractToolsFromRequestBody(requestBody string) ([]string, bool) {
	var tools []string
	hasWebSearch := false

	// Try parsing as structured format first
	var reqData struct {
		Tools []struct {
			Type     string `json:"type"`
			Name     string `json:"name"`
			Function *struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
	}
	if json.Unmarshal([]byte(requestBody), &reqData) == nil && len(reqData.Tools) > 0 {
		for _, tool := range reqData.Tools {
			toolName := tool.Name
			if toolName == "" && tool.Function != nil && tool.Function.Name != "" {
				toolName = tool.Function.Name
			}
			if toolName == "" && tool.Type != "" {
				toolName = tool.Type
			}
			if toolName != "" {
				tools = append(tools, toolName)
				if strings.Contains(toolName, "web_search") {
					hasWebSearch = true
				}
			}
		}
		return tools, hasWebSearch
	}

	// Try parsing as generic map for unexpected formats
	var genericReq map[string]interface{}
	if json.Unmarshal([]byte(requestBody), &genericReq) == nil {
		if toolsArr, ok := genericReq["tools"].([]interface{}); ok {
			for _, t := range toolsArr {
				if tool, ok := t.(map[string]interface{}); ok {
					var toolName string
					if name, ok := tool["name"].(string); ok && name != "" {
						toolName = name
					}
					if toolName == "" {
						if fn, ok := tool["function"].(map[string]interface{}); ok {
							if name, ok := fn["name"].(string); ok && name != "" {
								toolName = name
							}
						}
					}
					if toolName == "" {
						if t, ok := tool["type"].(string); ok && t != "" {
							toolName = t
						}
					}
					if toolName != "" {
						tools = append(tools, toolName)
						if strings.Contains(toolName, "web_search") {
							hasWebSearch = true
						}
					}
				}
			}
		}
	}
	return tools, hasWebSearch
}

// HandlePendingRequests handles GET /api/pending requests.
func (h *HistoryHandler) HandlePendingRequests(w http.ResponseWriter, r *http.Request) {
	if h.GetPending == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]interface{}{})
		return
	}

	pending := h.GetPending()

	type PendingWithElapsed struct {
		*domain.PendingRequest
		ElapsedMs int64 `json:"elapsed_ms"`
	}

	result := make([]PendingWithElapsed, len(pending))
	now := time.Now()
	for i, p := range pending {
		result[i] = PendingWithElapsed{
			PendingRequest: p,
			ElapsedMs:      now.Sub(p.StartTime).Milliseconds(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleUsecasesHistory handles GET /api/usecases/history requests.
func (h *HistoryHandler) HandleUsecasesHistory(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	rows, err := h.DB.Query(`
		SELECT DISTINCT usecase FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		ORDER BY usecase
	`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var usecases []string
	for rows.Next() {
		var usecase string
		rows.Scan(&usecase)
		usecases = append(usecases, usecase)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(usecases)
}

// HandleUsecaseDistribution handles GET /api/usecase-distribution requests.
func (h *HistoryHandler) HandleUsecaseDistribution(w http.ResponseWriter, r *http.Request) {
	usecase := r.URL.Query().Get("usecase")
	timeRange := r.URL.Query().Get("range")

	var interval string
	switch timeRange {
	case "24h":
		interval = "-1 day"
	case "7d":
		interval = "-7 days"
	default:
		interval = "-30 days"
		timeRange = "30d"
	}

	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	type Distribution struct {
		Sensitive string `json:"sensitive"`
		Precision string `json:"precision"`
		RouteType string `json:"route_type"`
		Model     string `json:"model"`
		Count     int    `json:"count"`
	}

	var rows *sql.Rows
	var err error

	if usecase == "" {
		rows, err = h.DB.Query(`
			SELECT
				CASE WHEN sensitive = 1 THEN 'true' ELSE 'false' END as sensitive,
				COALESCE(precision, 'medium') as precision,
				CASE WHEN has_images = 1 THEN 'vision' ELSE 'text' END as route_type,
				model,
				COUNT(*) as count
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '`+interval+`')
			GROUP BY sensitive, precision, route_type, model
			ORDER BY count DESC
		`)
	} else {
		rows, err = h.DB.Query(`
			SELECT
				CASE WHEN sensitive = 1 THEN 'true' ELSE 'false' END as sensitive,
				COALESCE(precision, 'medium') as precision,
				CASE WHEN has_images = 1 THEN 'vision' ELSE 'text' END as route_type,
				model,
				COUNT(*) as count
			FROM requests
			WHERE usecase = ?
				AND timestamp >= datetime('now', 'localtime', '`+interval+`')
			GROUP BY sensitive, precision, route_type, model
			ORDER BY count DESC
		`, usecase)
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var distribution []Distribution
	totalCount := 0
	for rows.Next() {
		var d Distribution
		rows.Scan(&d.Sensitive, &d.Precision, &d.RouteType, &d.Model, &d.Count)
		distribution = append(distribution, d)
		totalCount += d.Count
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"usecase":      usecase,
		"distribution": distribution,
		"total":        totalCount,
		"range":        timeRange,
	})
}

// HandleModelStats handles GET /api/model-stats requests.
func (h *HistoryHandler) HandleModelStats(w http.ResponseWriter, r *http.Request) {
	usecase := r.URL.Query().Get("usecase")
	timeRange := r.URL.Query().Get("range")

	var interval string
	switch timeRange {
	case "24h":
		interval = "-1 day"
	case "7d":
		interval = "-7 days"
	default:
		interval = "-30 days"
	}

	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	var rows *sql.Rows
	var err error

	if usecase == "" {
		rows, err = h.DB.Query(`
			SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '` + interval + `')
			GROUP BY model ORDER BY COUNT(*) DESC LIMIT 20
		`)
	} else {
		rows, err = h.DB.Query(`
			SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE usecase = ?
				AND timestamp >= datetime('now', 'localtime', '` + interval + `')
			GROUP BY model ORDER BY COUNT(*) DESC LIMIT 20
		`, usecase)
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var models []map[string]interface{}
	for rows.Next() {
		var model string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&model, &count, &cost, &avgLatency)
		models = append(models, map[string]interface{}{
			"model":          model,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

// HandleClearCache handles POST /api/cache/clear requests.
func (h *HistoryHandler) HandleClearCache(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" && r.Method != "DELETE" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	h.Cache.Clear()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Cache cleared",
	})
}

// HandleStats handles GET /api/stats requests.
func (h *HistoryHandler) HandleStats(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	stats := map[string]interface{}{}

	// Total requests
	var total int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests").Scan(&total)
	stats["total_requests"] = total

	// Cached requests
	var cached int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE cached = 1").Scan(&cached)
	stats["cached_requests"] = cached
	stats["cache_hit_rate"] = 0.0
	if total > 0 {
		stats["cache_hit_rate"] = float64(cached) / float64(total)
	}

	// Total cost
	var totalCost float64
	h.DB.QueryRow("SELECT COALESCE(SUM(cost_usd), 0) FROM requests").Scan(&totalCost)
	stats["total_cost_usd"] = totalCost

	// By provider
	rows, _ := h.DB.Query(`
		SELECT provider, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests GROUP BY provider
	`)
	defer rows.Close()

	byProvider := map[string]interface{}{}
	now := time.Now()
	for rows.Next() {
		var provider string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&provider, &count, &cost, &avgLatency)
		
		providerData := map[string]interface{}{
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		}
		
		// Add budget information if available
		if h.BudgetRepo != nil {
			providerBudget, err := h.BudgetRepo.GetProviderBudget(provider)
			if err == nil && providerBudget != nil && providerBudget.Enabled {
				// Calculate current period spending
				periodStart, periodEnd := budget.CalculatePeriod(now, providerBudget.MonthStartDay)
				periodSpending, err := h.BudgetRepo.GetProviderSpending(provider, periodStart, periodEnd)
				if err == nil {
					providerData["budget_usd"] = providerBudget.BudgetUSD
					providerData["period_spending"] = periodSpending
					providerData["budget_percentage"] = (periodSpending / providerBudget.BudgetUSD) * 100
				}
			}
		}
		
		byProvider[provider] = providerData
	}
	stats["by_provider"] = byProvider

	// By model
	rows, _ = h.DB.Query(`
		SELECT model, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests GROUP BY model ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	byModel := []map[string]interface{}{}
	for rows.Next() {
		var model string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&model, &count, &cost, &avgLatency)
		byModel = append(byModel, map[string]interface{}{
			"model":          model,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_model"] = byModel

	// By usecase
	rows, _ = h.DB.Query(`
		SELECT COALESCE(usecase, ''), COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
		FROM requests WHERE usecase IS NOT NULL AND usecase != '' GROUP BY usecase ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	byUsecase := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var count int
		var cost float64
		var avgLatency float64
		rows.Scan(&usecase, &count, &cost, &avgLatency)
		byUsecase = append(byUsecase, map[string]interface{}{
			"usecase":        usecase,
			"count":          count,
			"cost_usd":       cost,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_usecase"] = byUsecase

	// By usecase + sensitivity + precision
	rows, _ = h.DB.Query(`
		SELECT COALESCE(usecase, ''), sensitive, COALESCE(precision, ''), COUNT(*), AVG(latency_ms)
		FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		GROUP BY usecase, sensitive, precision
		ORDER BY COUNT(*) DESC
	`)
	defer rows.Close()

	byUsecaseDetail := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var sensitive bool
		var precision string
		var count int
		var avgLatency float64
		rows.Scan(&usecase, &sensitive, &precision, &count, &avgLatency)
		byUsecaseDetail = append(byUsecaseDetail, map[string]interface{}{
			"usecase":        usecase,
			"sensitive":      sensitive,
			"precision":      precision,
			"count":          count,
			"avg_latency_ms": avgLatency,
		})
	}
	stats["by_usecase_detail"] = byUsecaseDetail

	// Recent requests with optional filtering
	modelFilter := r.URL.Query().Get("model")
	sensitiveFilter := r.URL.Query().Get("sensitive")
	precisionFilter := r.URL.Query().Get("precision")

	recentQuery := `
		SELECT id, timestamp, provider, model, cached, latency_ms, cost_usd, success, error,
		       sensitive, precision, has_images, input_tokens, output_tokens, usecase
		FROM requests WHERE 1=1`
	var args []interface{}

	if modelFilter != "" {
		recentQuery += " AND model = ?"
		args = append(args, modelFilter)
	}
	if sensitiveFilter != "" {
		recentQuery += " AND sensitive = ?"
		if sensitiveFilter == "1" || sensitiveFilter == "true" {
			args = append(args, 1)
		} else {
			args = append(args, 0)
		}
	}
	if precisionFilter != "" {
		recentQuery += " AND precision = ?"
		args = append(args, precisionFilter)
	}
	recentQuery += " ORDER BY id DESC LIMIT 50"

	rows, _ = h.DB.Query(recentQuery, args...)
	defer rows.Close()

	recent := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var cached, success bool
		var latencyMs int64
		var costUsd float64
		var errStr sql.NullString
		var sensitive bool
		var precision sql.NullString
		var hasImages bool
		var inputTokens, outputTokens int
		var usecase sql.NullString
		rows.Scan(&id, &timestamp, &provider, &model, &cached, &latencyMs, &costUsd, &success, &errStr,
			&sensitive, &precision, &hasImages, &inputTokens, &outputTokens, &usecase)

		entry := map[string]interface{}{
			"id":            id,
			"timestamp":     timestamp,
			"provider":      provider,
			"model":         model,
			"cached":        cached,
			"latency_ms":    latencyMs,
			"cost_usd":      costUsd,
			"success":       success,
			"sensitive":     sensitive,
			"has_images":    hasImages,
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if precision.Valid {
			entry["precision"] = precision.String
		}
		if usecase.Valid {
			entry["usecase"] = usecase.String
		}
		recent = append(recent, entry)
	}
	stats["recent_requests"] = recent

	// Include active filters in response
	if modelFilter != "" || sensitiveFilter != "" || precisionFilter != "" {
		stats["filters"] = map[string]string{
			"model":     modelFilter,
			"sensitive": sensitiveFilter,
			"precision": precisionFilter,
		}
	}

	// Today's stats
	today := time.Now().Format("2006-01-02")
	var todayCount int
	var todayCost float64
	h.DB.QueryRow("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM requests WHERE timestamp LIKE ?", today+"%").Scan(&todayCount, &todayCost)
	stats["today"] = map[string]interface{}{
		"requests": todayCount,
		"cost_usd": todayCost,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// HandleTiming handles GET /api/timing requests.
func (h *HistoryHandler) HandleTiming(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")

	timing := map[string]interface{}{}

	// Vision requests by precision
	rows, _ := h.DB.Query(`
		SELECT precision, AVG(latency_ms), COUNT(*), MIN(latency_ms), MAX(latency_ms)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0 AND precision IS NOT NULL
		GROUP BY precision
	`)
	defer rows.Close()

	visionByPrecision := map[string]interface{}{}
	for rows.Next() {
		var precision string
		var avgLatency, minLatency, maxLatency float64
		var count int
		rows.Scan(&precision, &avgLatency, &count, &minLatency, &maxLatency)

		var p75Latency float64
		h.DB.QueryRow(`
			SELECT latency_ms FROM (
				SELECT latency_ms, ROW_NUMBER() OVER (ORDER BY latency_ms) as rn,
				       COUNT(*) OVER () as total
				FROM requests
				WHERE has_images = 1 AND success = 1 AND cached = 0 AND precision = ?
			) WHERE rn = CAST(total * 0.75 AS INTEGER) + 1
		`, precision).Scan(&p75Latency)

		if p75Latency == 0 {
			p75Latency = avgLatency
		}

		visionByPrecision[precision] = map[string]interface{}{
			"avg_ms":   int64(avgLatency),
			"p75_ms":   int64(p75Latency),
			"min_ms":   int64(minLatency),
			"max_ms":   int64(maxLatency),
			"count":    count,
			"avg_secs": avgLatency / 1000.0,
			"p75_secs": p75Latency / 1000.0,
		}
	}
	timing["vision_by_precision"] = visionByPrecision

	// Vision requests by sensitive flag
	rows, _ = h.DB.Query(`
		SELECT sensitive, AVG(latency_ms), COUNT(*)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0
		GROUP BY sensitive
	`)
	defer rows.Close()

	visionBySensitive := map[string]interface{}{}
	for rows.Next() {
		var sensitive bool
		var avgLatency float64
		var count int
		rows.Scan(&sensitive, &avgLatency, &count)
		key := "cloud"
		if sensitive {
			key = "local"
		}
		visionBySensitive[key] = map[string]interface{}{
			"avg_ms":   int64(avgLatency),
			"count":    count,
			"avg_secs": avgLatency / 1000.0,
		}
	}
	timing["vision_by_sensitive"] = visionBySensitive

	// Overall vision average
	var overallAvg float64
	var overallCount int
	h.DB.QueryRow(`
		SELECT AVG(latency_ms), COUNT(*)
		FROM requests
		WHERE has_images = 1 AND success = 1 AND cached = 0
	`).Scan(&overallAvg, &overallCount)
	timing["vision_overall"] = map[string]interface{}{
		"avg_ms":   int64(overallAvg),
		"count":    overallCount,
		"avg_secs": overallAvg / 1000.0,
	}

	// Default estimates if no data
	timing["defaults"] = map[string]interface{}{
		"low":       map[string]interface{}{"avg_secs": 15.0},
		"medium":    map[string]interface{}{"avg_secs": 20.0},
		"high":      map[string]interface{}{"avg_secs": 30.0},
		"very_high": map[string]interface{}{"avg_secs": 45.0},
	}

	json.NewEncoder(w).Encode(timing)
}

// HandleTTSHistory handles GET /api/tts-history requests.
func (h *HistoryHandler) HandleTTSHistory(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	rows, _ := h.DB.Query(`
		SELECT id, timestamp, provider, model, latency_ms, success, error, voice, input_chars
		FROM requests WHERE request_type = 'tts' ORDER BY id DESC LIMIT 50
	`)
	defer rows.Close()

	history := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var latencyMs int64
		var success bool
		var errStr, voice sql.NullString
		var inputChars int
		rows.Scan(&id, &timestamp, &provider, &model, &latencyMs, &success, &errStr, &voice, &inputChars)

		entry := map[string]interface{}{
			"id":          id,
			"timestamp":   timestamp,
			"provider":    provider,
			"model":       model,
			"latency_ms":  latencyMs,
			"success":     success,
			"input_chars": inputChars,
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if voice.Valid {
			entry["voice"] = voice.String
		}
		history = append(history, entry)
	}

	// Get stats
	var totalRequests int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'tts'").Scan(&totalRequests)
	var successCount int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'tts' AND success = 1").Scan(&successCount)
	var avgLatency float64
	h.DB.QueryRow("SELECT COALESCE(AVG(latency_ms), 0) FROM requests WHERE request_type = 'tts'").Scan(&avgLatency)
	var totalChars int
	h.DB.QueryRow("SELECT COALESCE(SUM(input_chars), 0) FROM requests WHERE request_type = 'tts'").Scan(&totalChars)

	result := map[string]interface{}{
		"history":        history,
		"total_requests": totalRequests,
		"success_count":  successCount,
		"avg_latency_ms": avgLatency,
		"total_chars":    totalChars,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleSTTHistory handles GET /api/stt-history requests.
func (h *HistoryHandler) HandleSTTHistory(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	rows, _ := h.DB.Query(`
		SELECT id, timestamp, provider, model, latency_ms, success, error, sensitive, input_chars, output_tokens, response_body
		FROM requests WHERE request_type = 'stt' ORDER BY id DESC LIMIT 50
	`)
	defer rows.Close()

	history := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var timestamp, provider, model string
		var latencyMs int64
		var success, sensitive bool
		var errStr, responseBody sql.NullString
		var inputChars, outputTokens int
		rows.Scan(&id, &timestamp, &provider, &model, &latencyMs, &success, &errStr, &sensitive, &inputChars, &outputTokens, &responseBody)

		entry := map[string]interface{}{
			"id":           id,
			"timestamp":    timestamp,
			"provider":     provider,
			"model":        model,
			"latency_ms":   latencyMs,
			"success":      success,
			"sensitive":    sensitive,
			"file_size":    inputChars,
			"output_chars": outputTokens,
		}
		if errStr.Valid {
			entry["error"] = errStr.String
		}
		if responseBody.Valid && responseBody.String != "" {
			entry["transcription"] = responseBody.String
		}
		history = append(history, entry)
	}

	// Get stats
	var totalRequests int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt'").Scan(&totalRequests)
	var successCount int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND success = 1").Scan(&successCount)
	var avgLatency float64
	h.DB.QueryRow("SELECT COALESCE(AVG(latency_ms), 0) FROM requests WHERE request_type = 'stt'").Scan(&avgLatency)
	var localCount, cloudCount int
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND provider = 'local'").Scan(&localCount)
	h.DB.QueryRow("SELECT COUNT(*) FROM requests WHERE request_type = 'stt' AND provider = 'openai'").Scan(&cloudCount)

	result := map[string]interface{}{
		"history":        history,
		"total_requests": totalRequests,
		"success_count":  successCount,
		"avg_latency_ms": avgLatency,
		"local_count":    localCount,
		"cloud_count":    cloudCount,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleRequestDetail handles GET /api/request requests.
func (h *HistoryHandler) HandleRequestDetail(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query().Get("id")
	if idStr == "" {
		http.Error(w, "id parameter required", http.StatusBadRequest)
		return
	}

	var id int64
	if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil {
		http.Error(w, "Invalid id", http.StatusBadRequest)
		return
	}

	h.DBMutex.Lock()
	var entry struct {
		ID             int64          `json:"id"`
		Timestamp      string         `json:"timestamp"`
		Provider       string         `json:"provider"`
		Model          string         `json:"model"`
		RequestedModel sql.NullString `json:"-"`
		Sensitive      bool           `json:"sensitive"`
		Precision      sql.NullString `json:"-"`
		Usecase        sql.NullString `json:"-"`
		Cached         bool           `json:"cached"`
		InputTokens    int            `json:"input_tokens"`
		OutputTokens   int            `json:"output_tokens"`
		LatencyMs      int64          `json:"latency_ms"`
		CostUSD        float64        `json:"cost_usd"`
		Success        bool           `json:"success"`
		Error          sql.NullString `json:"-"`
		CacheKey       sql.NullString `json:"-"`
		HasImages      bool           `json:"has_images"`
		RequestBody    sql.NullString `json:"-"`
		ResponseBody   sql.NullString `json:"-"`
		ClientIP       sql.NullString `json:"-"`
	}

	err := h.DB.QueryRow(`
		SELECT id, timestamp, provider, model, requested_model, sensitive, precision, usecase,
		       cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images,
		       request_body, response_body, client_ip
		FROM requests WHERE id = ?
	`, id).Scan(&entry.ID, &entry.Timestamp, &entry.Provider, &entry.Model,
		&entry.RequestedModel, &entry.Sensitive, &entry.Precision, &entry.Usecase, &entry.Cached,
		&entry.InputTokens, &entry.OutputTokens, &entry.LatencyMs, &entry.CostUSD,
		&entry.Success, &entry.Error, &entry.CacheKey, &entry.HasImages,
		&entry.RequestBody, &entry.ResponseBody, &entry.ClientIP)
	h.DBMutex.Unlock()

	if err == sql.ErrNoRows {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Build response with nullable fields
	response := map[string]interface{}{
		"id":            entry.ID,
		"timestamp":     entry.Timestamp,
		"provider":      entry.Provider,
		"model":         entry.Model,
		"sensitive":     entry.Sensitive,
		"cached":        entry.Cached,
		"input_tokens":  entry.InputTokens,
		"output_tokens": entry.OutputTokens,
		"latency_ms":    entry.LatencyMs,
		"cost_usd":      entry.CostUSD,
		"success":       entry.Success,
		"has_images":    entry.HasImages,
	}

	if entry.RequestedModel.Valid {
		response["requested_model"] = entry.RequestedModel.String
	}
	if entry.Precision.Valid {
		response["precision"] = entry.Precision.String
	}
	if entry.Usecase.Valid {
		response["usecase"] = entry.Usecase.String
	}
	if entry.Error.Valid && entry.Error.String != "" {
		response["error"] = entry.Error.String
	}
	if entry.ClientIP.Valid {
		response["client_ip"] = entry.ClientIP.String
	}

	if entry.CacheKey.Valid {
		response["cache_key"] = entry.CacheKey.String
	}

	// Get request body - try cache first, then DB
	var reqBody []byte
	if entry.CacheKey.Valid && h.Cache != nil {
		reqBody, _ = h.Cache.GetRequest(entry.CacheKey.String)
	}
	if reqBody == nil && entry.RequestBody.Valid {
		reqBody = []byte(entry.RequestBody.String)
	}
	if reqBody != nil {
		response["request"] = parseRequestBody(reqBody)
	}

	// Get response - try cache first, then DB
	var respBody []byte
	if entry.CacheKey.Valid && h.Cache != nil {
		respBody, _ = h.Cache.GetResponse(entry.CacheKey.String)
	}
	if respBody == nil && entry.ResponseBody.Valid {
		respBody = []byte(entry.ResponseBody.String)
	}
	if respBody != nil {
		var resp map[string]interface{}
		if json.Unmarshal(respBody, &resp) == nil {
			response["response"] = resp
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// parseRequestBody parses request body JSON into a displayable format.
func parseRequestBody(reqBody []byte) map[string]interface{} {
	// Try parsing as Chat Completions format first
	var chatReq struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string      `json:"role"`
			Content interface{} `json:"content"`
		} `json:"messages"`
		Sensitive  *bool       `json:"sensitive,omitempty"`
		Precision  string      `json:"precision,omitempty"`
		Usecase    string      `json:"usecase,omitempty"`
		NoCache    bool        `json:"no_cache,omitempty"`
		Tools      interface{} `json:"tools,omitempty"`
		ToolChoice interface{} `json:"tool_choice,omitempty"`
	}
	if json.Unmarshal(reqBody, &chatReq) == nil && len(chatReq.Messages) > 0 {
		var displayMessages []map[string]interface{}
		for _, msg := range chatReq.Messages {
			displayMsg := map[string]interface{}{
				"role": msg.Role,
			}
			if msg.Content != nil {
				displayMsg["content"] = msg.Content
			}
			displayMessages = append(displayMessages, displayMsg)
		}
		result := map[string]interface{}{
			"model":       chatReq.Model,
			"messages":    displayMessages,
			"tools":       chatReq.Tools,
			"tool_choice": chatReq.ToolChoice,
		}
		if chatReq.Sensitive != nil {
			result["sensitive"] = *chatReq.Sensitive
		}
		if chatReq.Precision != "" {
			result["precision"] = chatReq.Precision
		}
		if chatReq.Usecase != "" {
			result["usecase"] = chatReq.Usecase
		}
		if chatReq.NoCache {
			result["no_cache"] = chatReq.NoCache
		}
		return result
	}

	// Try parsing as Responses API format
	var respReq struct {
		Model        string      `json:"model"`
		Input        interface{} `json:"input"`
		Instructions string      `json:"instructions,omitempty"`
		Tools        []struct {
			Type        string                 `json:"type"`
			Name        string                 `json:"name,omitempty"`
			Description string                 `json:"description,omitempty"`
			Parameters  map[string]interface{} `json:"parameters,omitempty"`
			Function    *struct {
				Name        string                 `json:"name"`
				Description string                 `json:"description,omitempty"`
				Parameters  map[string]interface{} `json:"parameters,omitempty"`
			} `json:"function,omitempty"`
		} `json:"tools,omitempty"`
		ToolChoice interface{} `json:"tool_choice,omitempty"`
		Sensitive  *bool       `json:"sensitive,omitempty"`
		Precision  string      `json:"precision,omitempty"`
		Usecase    string      `json:"usecase,omitempty"`
	}
	if json.Unmarshal(reqBody, &respReq) == nil && (respReq.Input != nil || len(respReq.Tools) > 0) {
		var displayMessages []map[string]interface{}
		if respReq.Instructions != "" {
			displayMessages = append(displayMessages, map[string]interface{}{
				"role":    "system",
				"content": respReq.Instructions,
			})
		}
		switch input := respReq.Input.(type) {
		case string:
			displayMessages = append(displayMessages, map[string]interface{}{
				"role":    "user",
				"content": input,
			})
		case []interface{}:
			for _, item := range input {
				if m, ok := item.(map[string]interface{}); ok {
					displayMsg := map[string]interface{}{}
					if role, ok := m["role"].(string); ok {
						displayMsg["role"] = role
					}
					if content, ok := m["content"]; ok {
						displayMsg["content"] = content
					}
					if len(displayMsg) > 0 {
						displayMessages = append(displayMessages, displayMsg)
					}
				}
			}
		}

		// Convert tools to display format
		var displayTools []map[string]interface{}
		for _, tool := range respReq.Tools {
			displayTool := map[string]interface{}{
				"type": tool.Type,
			}
			name := tool.Name
			desc := tool.Description
			params := tool.Parameters

			if tool.Function != nil {
				if tool.Function.Name != "" {
					name = tool.Function.Name
				}
				if tool.Function.Description != "" {
					desc = tool.Function.Description
				}
				if tool.Function.Parameters != nil {
					params = tool.Function.Parameters
				}
			}

			if name != "" || desc != "" || params != nil {
				displayTool["function"] = map[string]interface{}{
					"name":        name,
					"description": desc,
					"parameters":  params,
				}
			}
			if name != "" {
				displayTool["name"] = name
			}
			displayTools = append(displayTools, displayTool)
		}

		sensitive := false
		if respReq.Sensitive != nil {
			sensitive = *respReq.Sensitive
		}

		return map[string]interface{}{
			"model":       respReq.Model,
			"messages":    displayMessages,
			"sensitive":   sensitive,
			"precision":   respReq.Precision,
			"usecase":     respReq.Usecase,
			"tools":       displayTools,
			"tool_choice": respReq.ToolChoice,
		}
	}

	return nil
}

// HandleAnalytics handles GET /api/analytics requests.
func (h *HistoryHandler) HandleAnalytics(w http.ResponseWriter, r *http.Request) {
	h.DBMutex.Lock()
	defer h.DBMutex.Unlock()

	w.Header().Set("Content-Type", "application/json")

	analytics := map[string]interface{}{}

	// Cost breakdown by usecase
	rows, _ := h.DB.Query(`
		SELECT COALESCE(usecase, 'unspecified'), COALESCE(SUM(cost_usd), 0), COUNT(*)
		FROM requests
		WHERE usecase IS NOT NULL AND usecase != ''
		GROUP BY usecase
		ORDER BY SUM(cost_usd) DESC
	`)
	defer rows.Close()

	costByUsecase := []map[string]interface{}{}
	for rows.Next() {
		var usecase string
		var cost float64
		var count int
		rows.Scan(&usecase, &cost, &count)
		costByUsecase = append(costByUsecase, map[string]interface{}{
			"usecase": usecase,
			"cost":    cost,
			"count":   count,
		})
	}
	analytics["cost_by_usecase"] = costByUsecase

	// Query count matrix by model, sensitivity, precision
	rows, _ = h.DB.Query(`
		SELECT model, sensitive, COALESCE(precision, 'unspecified'), COUNT(*), COALESCE(SUM(cost_usd), 0)
		FROM requests
		GROUP BY model, sensitive, precision
		ORDER BY model, sensitive, precision
	`)
	defer rows.Close()

	queryMatrix := []map[string]interface{}{}
	for rows.Next() {
		var model string
		var sensitive bool
		var precision string
		var count int
		var cost float64
		rows.Scan(&model, &sensitive, &precision, &count, &cost)
		queryMatrix = append(queryMatrix, map[string]interface{}{
			"model":     model,
			"sensitive": sensitive,
			"precision": precision,
			"count":     count,
			"cost":      cost,
		})
	}
	analytics["query_matrix"] = queryMatrix

	// Get all distinct models
	rows, _ = h.DB.Query(`SELECT DISTINCT model FROM requests ORDER BY model`)
	defer rows.Close()

	models := []string{}
	for rows.Next() {
		var model string
		rows.Scan(&model)
		models = append(models, model)
	}
	analytics["models"] = models

	// Volume over time
	timeRange := r.URL.Query().Get("range")
	if timeRange == "" {
		timeRange = "7d"
	}

	var volumeQuery string
	switch timeRange {
	case "1h":
		volumeQuery = `
			SELECT
				strftime('%Y-%m-%d %H:', timestamp) || printf('%02d', (CAST(strftime('%M', timestamp) AS INTEGER) / 5) * 5) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '-1 hour')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	case "1d":
		volumeQuery = `
			SELECT
				strftime('%Y-%m-%d %H:00', timestamp) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= datetime('now', 'localtime', '-1 day')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	case "30d":
		volumeQuery = `
			SELECT
				DATE(timestamp) as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= DATE('now', 'localtime', '-30 days')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	default: // 7d
		volumeQuery = `
			SELECT
				DATE(timestamp) || ' ' || printf('%02d', (CAST(strftime('%H', timestamp) AS INTEGER) / 4) * 4) || ':00' as time_block,
				model,
				COUNT(*),
				COALESCE(SUM(cost_usd), 0)
			FROM requests
			WHERE timestamp >= DATE('now', 'localtime', '-7 days')
			GROUP BY time_block, model
			ORDER BY time_block ASC, model
		`
	}

	rows, _ = h.DB.Query(volumeQuery)
	defer rows.Close()

	volumeOverTime := []map[string]interface{}{}
	for rows.Next() {
		var timeBlock, model string
		var count int
		var cost float64
		rows.Scan(&timeBlock, &model, &count, &cost)
		volumeOverTime = append(volumeOverTime, map[string]interface{}{
			"day":   timeBlock,
			"model": model,
			"count": count,
			"cost":  cost,
		})
	}
	analytics["volume_over_time"] = volumeOverTime
	analytics["time_range"] = timeRange

	// Hourly distribution
	rows, _ = h.DB.Query(`
		SELECT strftime('%H', timestamp) as hour, COUNT(*)
		FROM requests
		WHERE timestamp >= DATE('now', 'localtime', '-7 days')
		GROUP BY hour
		ORDER BY hour
	`)
	defer rows.Close()

	hourlyDistribution := []map[string]interface{}{}
	for rows.Next() {
		var hour string
		var count int
		rows.Scan(&hour, &count)
		hourlyDistribution = append(hourlyDistribution, map[string]interface{}{
			"hour":  hour,
			"count": count,
		})
	}
	analytics["hourly_distribution"] = hourlyDistribution

	// Model-specific stats
	modelParam := r.URL.Query().Get("model")
	if modelParam != "" {
		rows, _ = h.DB.Query(`
			SELECT DATE(timestamp) as day, COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms)
			FROM requests
			WHERE model = ? AND timestamp >= DATE('now', 'localtime', '-30 days')
			GROUP BY DATE(timestamp)
			ORDER BY day ASC
		`, modelParam)
		defer rows.Close()

		modelDailyVolume := []map[string]interface{}{}
		for rows.Next() {
			var day string
			var count int
			var cost, avgLatency float64
			rows.Scan(&day, &count, &cost, &avgLatency)
			modelDailyVolume = append(modelDailyVolume, map[string]interface{}{
				"day":         day,
				"count":       count,
				"cost":        cost,
				"avg_latency": avgLatency,
			})
		}
		analytics["model_daily_volume"] = modelDailyVolume

		var totalCount int
		var totalCost, avgLatency float64
		var minLatency, maxLatency int64
		h.DB.QueryRow(`
			SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), AVG(latency_ms), MIN(latency_ms), MAX(latency_ms)
			FROM requests WHERE model = ?
		`, modelParam).Scan(&totalCount, &totalCost, &avgLatency, &minLatency, &maxLatency)

		analytics["model_summary"] = map[string]interface{}{
			"model":       modelParam,
			"total_count": totalCount,
			"total_cost":  totalCost,
			"avg_latency": avgLatency,
			"min_latency": minLatency,
			"max_latency": maxLatency,
		}
	}

	json.NewEncoder(w).Encode(analytics)
}
