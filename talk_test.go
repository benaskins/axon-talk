package talk

import "testing"

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
