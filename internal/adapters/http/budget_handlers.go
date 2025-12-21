// Package http provides HTTP handler adapters (primary adapters).
package http

import (
	"encoding/json"
	"net/http"
	"time"

	budgetpkg "llm-proxy/internal/adapters/budget"
	"llm-proxy/internal/domain"
	"llm-proxy/internal/ports"
)

// BudgetHandler handles budget management API endpoints.
type BudgetHandler struct {
	BudgetRepo ports.BudgetRepository
}

// NewBudgetHandler creates a new budget handler.
func NewBudgetHandler(repo ports.BudgetRepository) *BudgetHandler {
	return &BudgetHandler{
		BudgetRepo: repo,
	}
}

// HandleGetProviderBudgets handles GET /api/budgets/providers
func (h *BudgetHandler) HandleGetProviderBudgets(w http.ResponseWriter, r *http.Request) {
	budgets, err := h.BudgetRepo.GetAllProviderBudgets()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Ensure we always return an array, not null
	if budgets == nil {
		budgets = []*domain.ProviderBudget{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(budgets)
}

// HandleSetProviderBudget handles POST /api/budgets/providers
func (h *BudgetHandler) HandleSetProviderBudget(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var budget domain.ProviderBudget
	if err := json.NewDecoder(r.Body).Decode(&budget); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if budget.Provider == "" {
		http.Error(w, "provider is required", http.StatusBadRequest)
		return
	}

	if budget.BudgetUSD < 0 {
		http.Error(w, "budget_usd must be non-negative", http.StatusBadRequest)
		return
	}

	if budget.MonthStartDay < 1 || budget.MonthStartDay > 31 {
		http.Error(w, "month_start_day must be between 1 and 31", http.StatusBadRequest)
		return
	}

	if err := h.BudgetRepo.SetProviderBudget(&budget); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(budget)
}

// HandleDeleteProviderBudget handles DELETE /api/budgets/providers/{provider}
func (h *BudgetHandler) HandleDeleteProviderBudget(w http.ResponseWriter, r *http.Request) {
	if r.Method != "DELETE" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	provider := r.URL.Path[len("/api/budgets/providers/"):]
	if provider == "" {
		http.Error(w, "provider is required", http.StatusBadRequest)
		return
	}

	if err := h.BudgetRepo.DeleteProviderBudget(provider); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// HandleGetGlobalBudget handles GET /api/budgets/global
func (h *BudgetHandler) HandleGetGlobalBudget(w http.ResponseWriter, r *http.Request) {
	// Use GetLatestGlobalBudget so we can see and manage disabled budgets
	budget, err := h.BudgetRepo.GetLatestGlobalBudget()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Return null explicitly if no budget exists (this is expected)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(budget)
}

// HandleSetGlobalBudget handles POST /api/budgets/global
func (h *BudgetHandler) HandleSetGlobalBudget(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var budget domain.GlobalBudget
	if err := json.NewDecoder(r.Body).Decode(&budget); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if budget.BudgetUSD < 0 {
		http.Error(w, "budget_usd must be non-negative", http.StatusBadRequest)
		return
	}

	if budget.MonthStartDay < 1 || budget.MonthStartDay > 31 {
		http.Error(w, "month_start_day must be between 1 and 31", http.StatusBadRequest)
		return
	}

	if err := h.BudgetRepo.SetGlobalBudget(&budget); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(budget)
}

// HandleGetBudgetSpending handles GET /api/budgets/spending?provider={provider}
func (h *BudgetHandler) HandleGetBudgetSpending(w http.ResponseWriter, r *http.Request) {
	provider := r.URL.Query().Get("provider")
	now := time.Now()

	var spending float64
	var err error
	var budget *domain.ProviderBudget

	if provider != "" {
		budget, err = h.BudgetRepo.GetProviderBudget(provider)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if budget == nil || !budget.Enabled {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"provider": provider,
				"spending": 0,
				"budget":   nil,
			})
			return
		}
		periodStart, periodEnd := budgetpkg.CalculatePeriod(now, budget.MonthStartDay)
		spending, err = h.BudgetRepo.GetProviderSpending(provider, periodStart, periodEnd)
	} else {
		// Global spending
		globalBudget, err := h.BudgetRepo.GetGlobalBudget()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if globalBudget == nil || !globalBudget.Enabled {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"spending": 0,
				"budget":   nil,
			})
			return
		}
		periodStart, periodEnd := budgetpkg.CalculatePeriod(now, globalBudget.MonthStartDay)
		spending, err = h.BudgetRepo.GetGlobalSpending(periodStart, periodEnd)
		budget = &domain.ProviderBudget{
			BudgetUSD:     globalBudget.BudgetUSD,
			MonthStartDay: globalBudget.MonthStartDay,
		}
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if provider != "" {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"provider": provider,
			"spending": spending,
			"budget":   budget.BudgetUSD,
			"percentage": (spending / budget.BudgetUSD) * 100,
		})
	} else {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"spending": spending,
			"budget":   budget.BudgetUSD,
			"percentage": (spending / budget.BudgetUSD) * 100,
		})
	}
}

