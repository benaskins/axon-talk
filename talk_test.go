package talk

import (
	"errors"
	"fmt"
	"testing"
)

func TestNewRequest(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: "hello"}}
	req := NewRequest("model-1", msgs)

	if req.Model != "model-1" {
		t.Errorf("model = %q, want model-1", req.Model)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(req.Messages))
	}
	if req.Options == nil {
		t.Error("options should be initialised")
	}
}

func TestNewRequest_WithOptions(t *testing.T) {
	withMaxTokens := func(n int) RequestOption {
		return func(r *Request) { r.Options["max_tokens"] = n }
	}
	withStream := func(r *Request) { r.Stream = true }

	req := NewRequest("model-1", nil, withMaxTokens(1024), withStream)

	if req.Options["max_tokens"] != 1024 {
		t.Errorf("max_tokens = %v, want 1024", req.Options["max_tokens"])
	}
	if !req.Stream {
		t.Error("stream should be true")
	}
}

func TestStatusError(t *testing.T) {
	err := &StatusError{StatusCode: 429, Body: "rate limited", Provider: "openai"}

	if err.Error() != "openai: status 429: rate limited" {
		t.Errorf("error = %q", err.Error())
	}

	// Should be unwrappable via errors.As
	var target *StatusError
	if !errors.As(err, &target) {
		t.Fatal("errors.As should match *StatusError")
	}
	if target.StatusCode != 429 {
		t.Errorf("status = %d, want 429", target.StatusCode)
	}
}

func TestStatusError_WrappedInFmt(t *testing.T) {
	inner := &StatusError{StatusCode: 503, Body: "unavailable", Provider: "anthropic"}
	wrapped := fmt.Errorf("chat failed: %w", inner)

	var target *StatusError
	if !errors.As(wrapped, &target) {
		t.Fatal("errors.As should unwrap through fmt.Errorf")
	}
	if target.StatusCode != 503 {
		t.Errorf("status = %d, want 503", target.StatusCode)
	}
}

func TestWithStructuredOutput(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "number"},
		},
		"required": []any{"name"},
	}

	req := NewRequest("model-1", nil, WithStructuredOutput(schema))

	got, ok := req.Options["structured_output"].(map[string]any)
	if !ok {
		t.Fatal("structured_output not set")
	}
	if got["type"] != "object" {
		t.Errorf("type = %v, want object", got["type"])
	}
}

func TestRoleConstants(t *testing.T) {
	if RoleSystem != "system" {
		t.Errorf("RoleSystem = %q", RoleSystem)
	}
	if RoleUser != "user" {
		t.Errorf("RoleUser = %q", RoleUser)
	}
	if RoleAssistant != "assistant" {
		t.Errorf("RoleAssistant = %q", RoleAssistant)
	}
	if RoleTool != "tool" {
		t.Errorf("RoleTool = %q", RoleTool)
	}
}
