package ports

import (
	"time"

	"llm-proxy/internal/domain"
)

// RequestLogger is a secondary port for persisting request logs.
// This is a minimal interface for the core logging functionality.
type RequestLogger interface {
	// LogRequest stores a request log entry and returns its ID.
	LogRequest(entry *domain.RequestLog) int64
}

// BudgetRepository is a secondary port for managing budget configurations and checking spending.
type BudgetRepository interface {
	// GetProviderBudget retrieves the budget configuration for a provider.
	GetProviderBudget(provider string) (*domain.ProviderBudget, error)

	// SetProviderBudget sets or updates the budget for a provider.
	SetProviderBudget(budget *domain.ProviderBudget) error

	// DeleteProviderBudget removes the budget for a provider.
	DeleteProviderBudget(provider string) error

	// GetAllProviderBudgets retrieves all provider budgets.
	GetAllProviderBudgets() ([]*domain.ProviderBudget, error)

	// GetGlobalBudget retrieves the active global budget (most recent enabled one).
	GetGlobalBudget() (*domain.GlobalBudget, error)

	// GetLatestGlobalBudget retrieves the most recent global budget regardless of enabled status.
	GetLatestGlobalBudget() (*domain.GlobalBudget, error)

	// SetGlobalBudget creates a new global budget.
	SetGlobalBudget(budget *domain.GlobalBudget) error

	// GetProviderSpending calculates total spending for a provider within the current period.
	// periodStart and periodEnd define the time range for the current billing period.
	GetProviderSpending(provider string, periodStart, periodEnd time.Time) (float64, error)

	// GetGlobalSpending calculates total spending across all providers within the current period.
	GetGlobalSpending(periodStart, periodEnd time.Time) (float64, error)
}
