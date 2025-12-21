// Package repository provides database adapter implementations.
package repository

import (
	"database/sql"
	"log"
	"sync"
	"time"

	"llm-proxy/internal/domain"
)

// SQLiteLogger implements RequestLogger port with SQLite storage.
type SQLiteLogger struct {
	db    *sql.DB
	mutex sync.Mutex
}

// NewSQLiteLogger creates a new SQLite request logger.
func NewSQLiteLogger(db *sql.DB) *SQLiteLogger {
	return &SQLiteLogger{db: db}
}

// LogRequest stores a request log entry and returns its ID.
func (r *SQLiteLogger) LogRequest(entry *domain.RequestLog) int64 {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	// Default request type to "llm" if not set
	if entry.RequestType == "" {
		entry.RequestType = "llm"
	}

	result, err := r.db.Exec(`
		INSERT INTO requests (timestamp, request_type, provider, model, requested_model, sensitive, precision, usecase, cached, input_tokens, output_tokens, latency_ms, cost_usd, success, error, cache_key, has_images, request_body, response_body, voice, audio_duration_ms, input_chars, is_replay, client_ip)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, entry.Timestamp.Format(time.RFC3339), entry.RequestType, entry.Provider, entry.Model, entry.RequestedModel,
		entry.Sensitive, entry.Precision, entry.Usecase, entry.Cached, entry.InputTokens, entry.OutputTokens,
		entry.LatencyMs, entry.CostUSD, entry.Success, entry.Error, entry.CacheKey, entry.HasImages,
		string(entry.RequestBody), string(entry.ResponseBody), entry.Voice, entry.AudioDurationMs, entry.InputChars, entry.IsReplay, entry.ClientIP)

	if err != nil {
		log.Printf("Failed to log request: %v", err)
		return 0
	}

	id, err := result.LastInsertId()
	if err != nil {
		log.Printf("Failed to get last insert ID: %v", err)
		return 0
	}
	return id
}

// SQLiteBudgetRepository implements BudgetRepository port with SQLite storage.
type SQLiteBudgetRepository struct {
	db    *sql.DB
	mutex sync.RWMutex
}

// NewSQLiteBudgetRepository creates a new SQLite budget repository.
func NewSQLiteBudgetRepository(db *sql.DB) *SQLiteBudgetRepository {
	return &SQLiteBudgetRepository{db: db}
}

// GetProviderBudget retrieves the budget configuration for a provider.
func (r *SQLiteBudgetRepository) GetProviderBudget(provider string) (*domain.ProviderBudget, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var budget domain.ProviderBudget
	var enabled int
	var createdAtStr, updatedAtStr string
	err := r.db.QueryRow(`
		SELECT provider, budget_usd, month_start_day, enabled, created_at, updated_at
		FROM provider_budgets
		WHERE provider = ?
	`, provider).Scan(&budget.Provider, &budget.BudgetUSD, &budget.MonthStartDay, &enabled, &createdAtStr, &updatedAtStr)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	budget.Enabled = enabled == 1
	if createdAtStr != "" {
		budget.CreatedAt, _ = time.Parse(time.RFC3339, createdAtStr)
	}
	if updatedAtStr != "" {
		budget.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAtStr)
	}
	return &budget, nil
}

// SetProviderBudget sets or updates the budget for a provider.
func (r *SQLiteBudgetRepository) SetProviderBudget(budget *domain.ProviderBudget) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	enabled := 0
	if budget.Enabled {
		enabled = 1
	}

	// Check if record exists
	var existingCreatedAt string
	err := r.db.QueryRow(`SELECT created_at FROM provider_budgets WHERE provider = ?`, budget.Provider).Scan(&existingCreatedAt)
	
	if err == sql.ErrNoRows {
		// New record - database will set created_at and updated_at automatically
		_, err = r.db.Exec(`
			INSERT INTO provider_budgets (provider, budget_usd, month_start_day, enabled)
			VALUES (?, ?, ?, ?)
		`, budget.Provider, budget.BudgetUSD, budget.MonthStartDay, enabled)
	} else if err == nil {
		// Existing record - trigger will update updated_at automatically
		_, err = r.db.Exec(`
			UPDATE provider_budgets 
			SET budget_usd = ?, month_start_day = ?, enabled = ?
			WHERE provider = ?
		`, budget.BudgetUSD, budget.MonthStartDay, enabled, budget.Provider)
	}
	return err
}

// DeleteProviderBudget removes the budget for a provider.
func (r *SQLiteBudgetRepository) DeleteProviderBudget(provider string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	_, err := r.db.Exec(`DELETE FROM provider_budgets WHERE provider = ?`, provider)
	return err
}

// GetAllProviderBudgets retrieves all provider budgets.
func (r *SQLiteBudgetRepository) GetAllProviderBudgets() ([]*domain.ProviderBudget, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	rows, err := r.db.Query(`
		SELECT provider, budget_usd, month_start_day, enabled, created_at, updated_at
		FROM provider_budgets
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var budgets []*domain.ProviderBudget
	for rows.Next() {
		var budget domain.ProviderBudget
		var enabled int
		var createdAtStr, updatedAtStr string
		if err := rows.Scan(&budget.Provider, &budget.BudgetUSD, &budget.MonthStartDay, &enabled, &createdAtStr, &updatedAtStr); err != nil {
			return nil, err
		}
		budget.Enabled = enabled == 1
		if createdAtStr != "" {
			budget.CreatedAt, _ = time.Parse(time.RFC3339, createdAtStr)
		}
		if updatedAtStr != "" {
			budget.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAtStr)
		}
		budgets = append(budgets, &budget)
	}

	return budgets, rows.Err()
}

// GetGlobalBudget retrieves the active global budget (most recent enabled one).
func (r *SQLiteBudgetRepository) GetGlobalBudget() (*domain.GlobalBudget, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var budget domain.GlobalBudget
	var enabled int
	var createdAtStr, updatedAtStr string
	err := r.db.QueryRow(`
		SELECT id, budget_usd, month_start_day, enabled, created_at, updated_at
		FROM global_budgets
		WHERE enabled = 1
		ORDER BY created_at DESC
		LIMIT 1
	`).Scan(&budget.ID, &budget.BudgetUSD, &budget.MonthStartDay, &enabled, &createdAtStr, &updatedAtStr)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	budget.Enabled = enabled == 1
	if createdAtStr != "" {
		budget.CreatedAt, _ = time.Parse(time.RFC3339, createdAtStr)
	}
	if updatedAtStr != "" {
		budget.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAtStr)
	}

	return &budget, nil
}

// GetLatestGlobalBudget retrieves the most recent global budget regardless of enabled status.
func (r *SQLiteBudgetRepository) GetLatestGlobalBudget() (*domain.GlobalBudget, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var budget domain.GlobalBudget
	var enabled int
	var createdAtStr, updatedAtStr string
	err := r.db.QueryRow(`
		SELECT id, budget_usd, month_start_day, enabled, created_at, updated_at
		FROM global_budgets
		ORDER BY created_at DESC
		LIMIT 1
	`).Scan(&budget.ID, &budget.BudgetUSD, &budget.MonthStartDay, &enabled, &createdAtStr, &updatedAtStr)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	budget.Enabled = enabled == 1
	if createdAtStr != "" {
		budget.CreatedAt, _ = time.Parse(time.RFC3339, createdAtStr)
	}
	if updatedAtStr != "" {
		budget.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAtStr)
	}

	return &budget, nil
}

// SetGlobalBudget creates a new global budget or updates the most recent one.
func (r *SQLiteBudgetRepository) SetGlobalBudget(budget *domain.GlobalBudget) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	enabled := 0
	if budget.Enabled {
		enabled = 1
	}

	// Check if there's an existing global budget
	var existingID int64
	err := r.db.QueryRow(`
		SELECT id FROM global_budgets ORDER BY created_at DESC LIMIT 1
	`).Scan(&existingID)

	if err == sql.ErrNoRows {
		// No existing budget - create new one
		// Database will set created_at and updated_at automatically
		_, err = r.db.Exec(`
			INSERT INTO global_budgets (budget_usd, month_start_day, enabled)
			VALUES (?, ?, ?)
		`, budget.BudgetUSD, budget.MonthStartDay, enabled)
	} else if err == nil {
		// Update existing budget - trigger will update updated_at automatically
		_, err = r.db.Exec(`
			UPDATE global_budgets 
			SET budget_usd = ?, month_start_day = ?, enabled = ?
			WHERE id = ?
		`, budget.BudgetUSD, budget.MonthStartDay, enabled, existingID)
	}
	return err
}

// GetProviderSpending calculates total spending for a provider within the current period.
func (r *SQLiteBudgetRepository) GetProviderSpending(provider string, periodStart, periodEnd time.Time) (float64, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var spending float64
	err := r.db.QueryRow(`
		SELECT COALESCE(SUM(cost_usd), 0)
		FROM requests
		WHERE provider = ? AND success = 1
		AND timestamp >= ? AND timestamp < ?
	`, provider, periodStart.Format(time.RFC3339), periodEnd.Format(time.RFC3339)).Scan(&spending)

	return spending, err
}

// GetGlobalSpending calculates total spending across all providers within the current period.
func (r *SQLiteBudgetRepository) GetGlobalSpending(periodStart, periodEnd time.Time) (float64, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var spending float64
	err := r.db.QueryRow(`
		SELECT COALESCE(SUM(cost_usd), 0)
		FROM requests
		WHERE success = 1
		AND timestamp >= ? AND timestamp < ?
	`, periodStart.Format(time.RFC3339), periodEnd.Format(time.RFC3339)).Scan(&spending)

	return spending, err
}
