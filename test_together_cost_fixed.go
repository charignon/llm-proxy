package main

import (
	"fmt"
	"strings"
)

// Test the updated cost calculation logic with case-insensitive matching
func testCalculateCostFixed() {
	// Together.ai models from pricing map
	modelPricing := map[string][2]float64{
		"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {0.27, 0.85},
		"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":           {2.00, 2.00},
		"moonshotai/Kimi-K2-Thinking":                       {1.20, 4.00},
	}

	// Test models that might be requested
	testModels := []string{
		"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  // Exact match
		"Meta-Llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  // Different case
		"meta-llama/llama-4-maverick-17b-128e-instruct-fp8",  // Lowercase
		"META-LLAMA/LLAMA-4-MAVERICK-17B-128E-INSTRUCT-FP8",  // Uppercase
		"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",            // Exact match
		"qwen/qwen3-coder-480b-a35b-instruct-fp8",            // Lowercase
		"MOONSHOTAI/KIMI-K2-THINKING",                        // Uppercase variant
		"unknown-model",                                      // Not in map
	}

	fmt.Println("Testing with case-insensitive matching:")
	for _, model := range testModels {
		cost := calculateCostFixedTest(model, 1000, 1000, modelPricing)
		fmt.Printf("Model: %-55s Cost: $%.6f\n", model, cost)
	}
}

func calculateCostFixedTest(model string, inputTokens, outputTokens int, pricingMap map[string][2]float64) float64 {
	pricing, ok := pricingMap[model]
	if !ok {
		// Try case-insensitive match
		for m, p := range pricingMap {
			if strings.EqualFold(model, m) {
				pricing = p
				ok = true
				break
			}
		}
	}
	if !ok {
		return 0
	}
	return (float64(inputTokens) * pricing[0] / 1_000_000) + (float64(outputTokens) * pricing[1] / 1_000_000)
}