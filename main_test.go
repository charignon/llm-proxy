package main

import "testing"

func setupAssistantModelTestDB(t *testing.T) {
	t.Helper()

	originalDataDir := dataDir
	originalDB := db
	originalAssistantModels := getAllAssistantModels()

	dataDir = t.TempDir()
	db = nil

	assistantModelsMutex.Lock()
	assistantModels = make(map[string]AssistantModelConfig)
	assistantModelsMutex.Unlock()

	if err := initDB(); err != nil {
		t.Fatalf("initDB returned error: %v", err)
	}

	t.Cleanup(func() {
		tempDB := db
		if tempDB != nil && tempDB != originalDB {
			_ = tempDB.Close()
		}

		db = originalDB
		dataDir = originalDataDir

		assistantModelsMutex.Lock()
		assistantModels = originalAssistantModels
		assistantModelsMutex.Unlock()
	})
}

func TestInitDBSeedsAssistantMLXDefault(t *testing.T) {
	setupAssistantModelTestDB(t)

	loadAssistantModels()

	want := assistantAliasDefaults["assistant-mlx"]
	provider, model, found := getAssistantModel("assistant-mlx")
	if !found {
		t.Fatal("assistant-mlx alias was not loaded from defaults")
	}
	if provider != want.Provider {
		t.Fatalf("assistant-mlx provider = %q, want %q", provider, want.Provider)
	}
	if model != want.Model {
		t.Fatalf("assistant-mlx model = %q, want %q", model, want.Model)
	}
}

func TestSetAssistantModelAcceptsAssistantMLX(t *testing.T) {
	setupAssistantModelTestDB(t)

	want := AssistantModelConfig{
		Provider: "mlx",
		Model:    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
	}
	if err := setAssistantModel("assistant-mlx", want.Provider, want.Model); err != nil {
		t.Fatalf("setAssistantModel returned error: %v", err)
	}

	provider, model, found := getAssistantModel("assistant-mlx")
	if !found {
		t.Fatal("assistant-mlx alias was not saved")
	}
	if provider != want.Provider {
		t.Fatalf("assistant-mlx provider = %q, want %q", provider, want.Provider)
	}
	if model != want.Model {
		t.Fatalf("assistant-mlx model = %q, want %q", model, want.Model)
	}
}
