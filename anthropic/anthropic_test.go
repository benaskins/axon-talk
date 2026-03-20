package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	loop "github.com/benaskins/axon-loop"
	tool "github.com/benaskins/axon-tool"
)

func TestClientImplementsLLMClient(t *testing.T) {
	var _ loop.LLMClient = NewClient("http://example.com", "key")
}

func TestChat_BasicResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("anthropic-version = %q", r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}

		json.NewEncoder(w).Encode(messagesResponse{
			Content: []contentBlock{{
				Type: "text",
				Text: "4",
			}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2?"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	var got loop.Response
	err := client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if got.Content != "4" {
		t.Errorf("content = %q, want %q", got.Content, "4")
	}
	if !got.Done {
		t.Error("done should be true")
	}
}

func TestChat_SystemPromptExtracted(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model: "claude-opus-4-6",
		Messages: []loop.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "hi"},
		},
		Options: map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotBody.System) != 1 || gotBody.System[0].Text != "You are helpful." {
		t.Errorf("system = %+v, want [{text: You are helpful.}]", gotBody.System)
	}
	if len(gotBody.Messages) != 1 {
		t.Fatalf("got %d messages, want 1 (system should be extracted)", len(gotBody.Messages))
	}
	if gotBody.Messages[0].Role != "user" {
		t.Errorf("first message role = %q, want user", gotBody.Messages[0].Role)
	}
}

func TestChat_ModelInRequest(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotBody.Model != "claude-sonnet-4-6" {
		t.Errorf("model = %q, want claude-sonnet-4-6", gotBody.Model)
	}
}

func TestChat_WithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(messagesResponse{
			Content: []contentBlock{{
				Type:  "tool_use",
				ID:    "toolu_01",
				Name:  "get_weather",
				Input: map[string]any{"city": "Sydney"},
			}},
			StopReason: "tool_use",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "Weather in Sydney?"}},
		Tools: []tool.ToolDef{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"city"},
				Properties: map[string]tool.PropertySchema{
					"city": {Type: "string", Description: "City name"},
				},
			},
		}},
		Options: map[string]any{"max_tokens": 1024},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if len(got.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", got.ToolCalls[0].Name)
	}
	if got.ToolCalls[0].Arguments["city"] != "Sydney" {
		t.Errorf("city = %v, want Sydney", got.ToolCalls[0].Arguments["city"])
	}
}

func TestChat_ToolResultsInMessages(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "It's sunny."}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model: "claude-opus-4-6",
		Messages: []loop.Message{
			{Role: "user", Content: "Weather?"},
			{Role: "assistant", ToolCalls: []loop.ToolCall{
				{Name: "get_weather", Arguments: map[string]any{"city": "Sydney"}},
			}},
			{Role: "tool", Content: "Sunny, 22°C"},
		},
		Options: map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotBody.Messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(gotBody.Messages))
	}

	// Assistant message should have tool_use content block
	assistantMsg := gotBody.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Errorf("msg[1] role = %q, want assistant", assistantMsg.Role)
	}
	if len(assistantMsg.Content) == 0 {
		t.Fatal("assistant message should have content blocks")
	}
	if assistantMsg.Content[0].Type != "tool_use" {
		t.Errorf("assistant content[0] type = %q, want tool_use", assistantMsg.Content[0].Type)
	}

	// Tool message should have tool_result content block
	toolMsg := gotBody.Messages[2]
	if toolMsg.Role != "user" {
		t.Errorf("msg[2] role = %q, want user (tool results are user messages in Anthropic API)", toolMsg.Role)
	}
	if len(toolMsg.Content) == 0 {
		t.Fatal("tool message should have content blocks")
	}
	if toolMsg.Content[0].Type != "tool_result" {
		t.Errorf("tool content[0] type = %q, want tool_result", toolMsg.Content[0].Type)
	}
}

func TestChat_ToolsSentInRequest(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Tools: []tool.ToolDef{{
			Name:        "search",
			Description: "Search the web",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"query"},
				Properties: map[string]tool.PropertySchema{
					"query": {Type: "string", Description: "Search query"},
					"limit": {Type: "number", Description: "Max results"},
				},
			},
		}},
		Options: map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotBody.Tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(gotBody.Tools))
	}
	if gotBody.Tools[0].Name != "search" {
		t.Errorf("name = %q, want search", gotBody.Tools[0].Name)
	}
	if gotBody.Tools[0].InputSchema.Properties["query"].Type != "string" {
		t.Errorf("query type = %q", gotBody.Tools[0].InputSchema.Properties["query"].Type)
	}
}

func TestChat_MaxTokensFromOptions(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 500},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotBody.MaxTokens != 500 {
		t.Errorf("max_tokens = %d, want 500", gotBody.MaxTokens)
	}
}

func TestChat_DefaultMaxTokens(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotBody.MaxTokens != 4096 {
		t.Errorf("max_tokens = %d, want 4096 (default)", gotBody.MaxTokens)
	}
}

func TestChat_GatewayToken(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key", WithGatewayToken("my-gw-token"))
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotHeader != "Bearer my-gw-token" {
		t.Errorf("cf-aig-authorization = %q, want %q", gotHeader, "Bearer my-gw-token")
	}
}

func TestChat_NoGatewayToken(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotHeader != "" {
		t.Errorf("cf-aig-authorization should be empty when no gateway token set, got %q", gotHeader)
	}
}

func TestChat_SystemOnlyMessages(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "What shall we discuss?"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model: "claude-opus-4-6",
		Messages: []loop.Message{
			{Role: "system", Content: "You are a journalist."},
		},
		Options: map[string]any{"max_tokens": 1024},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if len(gotBody.System) != 1 {
		t.Errorf("system blocks = %d, want 1", len(gotBody.System))
	}
	if len(gotBody.Messages) != 1 {
		t.Fatalf("messages = %d, want 1 (placeholder)", len(gotBody.Messages))
	}
	if gotBody.Messages[0].Role != "user" {
		t.Errorf("placeholder role = %q, want user", gotBody.Messages[0].Role)
	}
	if got.Content != "What shall we discuss?" {
		t.Errorf("content = %q", got.Content)
	}
}

func TestChat_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "bad-key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	err := client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

func TestChat_MultipleContentBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(messagesResponse{
			Content: []contentBlock{
				{Type: "text", Text: "Let me check. "},
				{Type: "tool_use", ID: "toolu_01", Name: "search", Input: map[string]any{"q": "test"}},
			},
			StopReason: "tool_use",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "search test"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if got.Content != "Let me check. " {
		t.Errorf("content = %q, want %q", got.Content, "Let me check. ")
	}
	if len(got.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "search" {
		t.Errorf("tool name = %q, want search", got.ToolCalls[0].Name)
	}
}

func TestChat_RequestURL(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL+"/v1/account123/axon-gate/anthropic", "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotPath != "/v1/account123/axon-gate/anthropic/v1/messages" {
		t.Errorf("path = %q, want /v1/account123/axon-gate/anthropic/v1/messages", gotPath)
	}
}

func TestChat_TemperatureFromOptions(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 1024, "temperature": 0.7},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotBody.Temperature == nil || *gotBody.Temperature != 0.7 {
		t.Errorf("temperature = %v, want 0.7", gotBody.Temperature)
	}
}

func TestChat_MultipleToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(messagesResponse{
			Content: []contentBlock{
				{Type: "tool_use", ID: "toolu_01", Name: "get_weather", Input: map[string]any{"city": "Sydney"}},
				{Type: "tool_use", ID: "toolu_02", Name: "get_weather", Input: map[string]any{"city": "Melbourne"}},
			},
			StopReason: "tool_use",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model:    "claude-opus-4-6",
		Messages: []loop.Message{{Role: "user", Content: "Weather in Sydney and Melbourne?"}},
		Options:  map[string]any{"max_tokens": 1024},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if len(got.ToolCalls) != 2 {
		t.Fatalf("got %d tool calls, want 2", len(got.ToolCalls))
	}
	if got.ToolCalls[1].Arguments["city"] != "Melbourne" {
		t.Errorf("second call city = %v, want Melbourne", got.ToolCalls[1].Arguments["city"])
	}
}

func TestChat_MultipleToolResultsInMessages(t *testing.T) {
	var gotBody messagesRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(messagesResponse{
			Content:    []contentBlock{{Type: "text", Text: "Both sunny."}},
			StopReason: "end_turn",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &loop.Request{
		Model: "claude-opus-4-6",
		Messages: []loop.Message{
			{Role: "user", Content: "Weather?"},
			{Role: "assistant", ToolCalls: []loop.ToolCall{
				{Name: "get_weather", Arguments: map[string]any{"city": "Sydney"}},
				{Name: "get_weather", Arguments: map[string]any{"city": "Melbourne"}},
			}},
			{Role: "tool", Content: "Sunny, 22°C"},
			{Role: "tool", Content: "Sunny, 19°C"},
		},
		Options: map[string]any{"max_tokens": 1024},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	// In Anthropic API, multiple tool results for the same assistant turn
	// should be combined into a single user message with multiple tool_result blocks.
	// Messages: user, assistant (with tool_use blocks), user (with tool_result blocks)
	if len(gotBody.Messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(gotBody.Messages))
	}
	toolResultMsg := gotBody.Messages[2]
	if toolResultMsg.Role != "user" {
		t.Errorf("tool result msg role = %q, want user", toolResultMsg.Role)
	}
	if len(toolResultMsg.Content) != 2 {
		t.Fatalf("tool result msg has %d content blocks, want 2", len(toolResultMsg.Content))
	}
	if toolResultMsg.Content[0].Type != "tool_result" {
		t.Errorf("content[0] type = %q, want tool_result", toolResultMsg.Content[0].Type)
	}
	if toolResultMsg.Content[1].Type != "tool_result" {
		t.Errorf("content[1] type = %q, want tool_result", toolResultMsg.Content[1].Type)
	}
}
