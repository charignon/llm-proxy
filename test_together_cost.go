package main

import (
	"fmt"
	"strings"
)

// Test the cost calculation logic
func testCalculateCost() {
	// Together.ai models from pricing map
	modelPricing := map[string][2]float64{
		"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {0.27, 0.85},
		"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":           {2.00, 2.00},
		"moonshotai/Kimi-K2-Thinking":                       {1.20, 4.00},
	}

	// Test models that might be requested
	testModels := []string{
		"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  // Exact match
		"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",            // Exact match
		"moonshotai/Kimi-K2-Thinking",                        // Exact match
		"Meta-Llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  // Different case
		"meta-llama/llama-4-maverick-17b-128e-instruct-fp8",  // Lowercase
		"Llama-4-Maverick-17B-128E-Instruct-FP8",             // Without prefix
		"unknown-model",                                      // Not in map
	}

	for _, model := range testModels {
		cost := calculateCostTest(model, 1000, 1000, modelPricing)
		fmt.Printf("Model: %-55s Cost: $%.6f\n", model, cost)
	}
}

func calculateCostTest(model string, inputTokens, outputTokens int, pricingMap map[string][2]float64) float64 {
	pricing, ok := pricingMap[model]
	if !ok {
		// Try prefix match (current logic)
		for m, p := range pricingMap {
			if strings.HasPrefix(model, m) {
				pricing = p
				ok = true
				break
			}
		}
		// Try reverse prefix match (what might be needed)
		if !ok {
			for m, p := range pricingMap {
				if strings.HasPrefix(m, model) {
					pricing = p
					ok = true
					break
				}
			}
		}
	}
	if !ok {
		return 0
	}
	return (float64(inputTokens) * pricing[0] / 1_000_000) + (float64(outputTokens) * pricing[1] / 1_000_000)
}

func main() {
	testCalculateCost()
}