// Package budget provides budget checking functionality.
package budget

import (
	"fmt"
	"time"

	"llm-proxy/internal/ports"
)

// BudgetChecker checks if requests should be allowed based on budget limits.
type BudgetChecker struct {
	repo ports.BudgetRepository
}

// NewBudgetChecker creates a new budget checker.
func NewBudgetChecker(repo ports.BudgetRepository) *BudgetChecker {
	return &BudgetChecker{repo: repo}
}

// CheckBudget verifies if a request to a provider should be allowed based on budget limits.
// Returns an error if the budget is exceeded, nil otherwise.
func (b *BudgetChecker) CheckBudget(provider string) error {
	now := time.Now()

	// Check provider budget
	providerBudget, err := b.repo.GetProviderBudget(provider)
	if err != nil {
		return fmt.Errorf("bfailed to get provider budget: %w", err)
	}

	if providerBudget != nil && providerBudget.Enabled {
		periodStart, periodEnd := CalculatePeriod(now, providerBudget.MonthStartDay)
		spending, err := b.repo.GetProviderSpending(provider, periodStart, periodEnd)
		if err != nil {
			return fmt.Errorf("failed to get provider spending: %w", err)
		}

		if spending >= providerBudget.BudgetUSD {
			return fmt.Errorf("provider budget exceeded: $%.2f spent of $%.2f limit for %s", spending, providerBudget.BudgetUSD, provider)
		}
	}

	// Check global budget
	globalBudget, err := b.repo.GetGlobalBudget()
	if err != nil {
		return fmt.Errorf("failed to get global budget: %w", err)
	}

	if globalBudget != nil && globalBudget.Enabled {
		periodStart, periodEnd := CalculatePeriod(now, globalBudget.MonthStartDay)
		spending, err := b.repo.GetGlobalSpending(periodStart, periodEnd)
		if err != nil {
			return fmt.Errorf("failed to get global spending: %w", err)
		}

		if spending >= globalBudget.BudgetUSD {
			return fmt.Errorf("global budget exceeded: $%.2f spent of $%.2f limit", spending, globalBudget.BudgetUSD)
		}
	}

	return nil
}

// CalculatePeriod calculates the start and end of the current billing period
// based on the current time and the month start day.
// monthStartDay is the day of the month when the period starts (1-31).
func CalculatePeriod(now time.Time, monthStartDay int) (start, end time.Time) {
	// Normalize monthStartDay to valid range (1-31)
	if monthStartDay < 1 {
		monthStartDay = 1
	}
	if monthStartDay > 31 {
		monthStartDay = 31
	}

	// Get current year and month
	year := now.Year()
	month := now.Month()
	day := now.Day()

	// Helper function to get the last day of a month
	lastDayOfMonth := func(y int, m time.Month) int {
		// Get first day of next month, then subtract one day
		nextMonth := time.Date(y, m+1, 1, 0, 0, 0, 0, now.Location())
		lastDay := nextMonth.AddDate(0, 0, -1)
		return lastDay.Day()
	}

	// Adjust monthStartDay to not exceed the last day of the target month
	adjustDayForMonth := func(y int, m time.Month, d int) int {
		lastDay := lastDayOfMonth(y, m)
		if d > lastDay {
			return lastDay
		}
		return d
	}

	// Determine the start of the current period
	// If current day >= monthStartDay, period started this month
	// Otherwise, period started last month
	if day >= monthStartDay {
		// Period started this month
		adjustedDay := adjustDayForMonth(year, month, monthStartDay)
		start = time.Date(year, month, adjustedDay, 0, 0, 0, 0, now.Location())
		// End is start of next period (next month, same day)
		nextMonth := month + 1
		nextYear := year
		if nextMonth > 12 {
			nextMonth = 1
			nextYear = year + 1
		}
		adjustedNextDay := adjustDayForMonth(nextYear, nextMonth, monthStartDay)
		end = time.Date(nextYear, nextMonth, adjustedNextDay, 0, 0, 0, 0, now.Location())
	} else {
		// Period started last month
		prevMonth := month - 1
		prevYear := year
		if prevMonth < 1 {
			prevMonth = 12
			prevYear = year - 1
		}
		adjustedPrevDay := adjustDayForMonth(prevYear, prevMonth, monthStartDay)
		start = time.Date(prevYear, prevMonth, adjustedPrevDay, 0, 0, 0, 0, now.Location())
		// End is start of current period (this month, same day)
		adjustedDay := adjustDayForMonth(year, month, monthStartDay)
		end = time.Date(year, month, adjustedDay, 0, 0, 0, 0, now.Location())
	}

	return start, end
}

