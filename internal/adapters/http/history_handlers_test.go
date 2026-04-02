package http

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleCancelPendingRequest(t *testing.T) {
	t.Parallel()

	called := ""
	handler := &HistoryHandler{
		CancelPending: func(id string) bool {
			called = id
			return id == "req-42"
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/api/pending/req-42/cancel", nil)
	rr := httptest.NewRecorder()

	handler.HandleCancelPendingRequest(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
	if called != "req-42" {
		t.Fatalf("CancelPending called with %q, want req-42", called)
	}

	var resp map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp["status"] != "cancelling" {
		t.Fatalf("status payload = %q, want cancelling", resp["status"])
	}
}
