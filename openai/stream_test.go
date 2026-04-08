package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
)

func sseServer(lines ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range lines {
			fmt.Fprintf(w, "data: %s\n\n", line)
		}
	}))
}

func TestChat_StreamContentTokens(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"content":" world"}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var tokens []string
	var gotDone bool
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Content != "" {
			tokens = append(tokens, resp.Content)
		}
		if resp.Done {
			gotDone = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if got := strings.Join(tokens, ""); got != "Hello world" {
		t.Errorf("content = %q, want %q", got, "Hello world")
	}
	if !gotDone {
		t.Error("expected done")
	}
}

func TestChat_StreamThinking(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"reasoning_content":"Think"}}]}`,
		`{"choices":[{"delta":{"reasoning_content":"ing"}}]}`,
		`{"choices":[{"delta":{"content":"Answer"}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "o3",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var thinking, content []string
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Thinking != "" {
			thinking = append(thinking, resp.Thinking)
		}
		if resp.Content != "" {
			content = append(content, resp.Content)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if got := strings.Join(thinking, ""); got != "Thinking" {
		t.Errorf("thinking = %q, want %q", got, "Thinking")
	}
	if got := strings.Join(content, ""); got != "Answer" {
		t.Errorf("content = %q, want %q", got, "Answer")
	}
}

func TestChat_StreamStructuredToolCalls(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Sydney\"}"}}]}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "weather?"}},
		Stream:   true,
		Tools: []tool.ToolDef{{
			Name: "get_weather",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"city": {Type: "string"},
				},
			},
		}},
	}

	var responses []talk.Response
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		responses = append(responses, resp)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	var toolCalls []talk.ToolCall
	for _, r := range responses {
		toolCalls = append(toolCalls, r.ToolCalls...)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(toolCalls))
	}
	if toolCalls[0].Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", toolCalls[0].Name)
	}
	if toolCalls[0].Arguments["city"] != "Sydney" {
		t.Errorf("city = %v, want Sydney", toolCalls[0].Arguments["city"])
	}
}

func TestChat_StreamMultipleToolCalls(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"search","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"go\"}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"q\":\"rust\"}"}}]}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "search"}},
		Stream:   true,
	}

	var toolCalls []talk.ToolCall
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		toolCalls = append(toolCalls, resp.ToolCalls...)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(toolCalls) != 2 {
		t.Fatalf("got %d tool calls, want 2", len(toolCalls))
	}
	if toolCalls[0].Arguments["q"] != "go" {
		t.Errorf("first q = %v, want go", toolCalls[0].Arguments["q"])
	}
	if toolCalls[1].Arguments["q"] != "rust" {
		t.Errorf("second q = %v, want rust", toolCalls[1].Arguments["q"])
	}
}

func TestChat_StreamSetsStreamInRequest(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"ok"}}]}`)
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotBody.Stream == nil || !*gotBody.Stream {
		t.Error("stream should be true in request body")
	}
}

func TestChat_StreamNormalizesToolCallArgs(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"set_temp","arguments":"{\"degrees\":\"22\"}"}}]}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "set temp"}},
		Stream:   true,
		Tools: []tool.ToolDef{{
			Name: "set_temp",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"degrees": {Type: "number"},
				},
			},
		}},
	}

	var toolCalls []talk.ToolCall
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		toolCalls = append(toolCalls, resp.ToolCalls...)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(toolCalls))
	}
	if v, ok := toolCalls[0].Arguments["degrees"].(float64); !ok || v != 22 {
		t.Errorf("degrees = %v (%T), want 22 (float64)", toolCalls[0].Arguments["degrees"], toolCalls[0].Arguments["degrees"])
	}
}

func TestChat_StreamUsageOnSeparateChunk(t *testing.T) {
	// With stream_options.include_usage, usage arrives on a final chunk
	// with an empty choices array.
	server := sseServer(
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"content":" world"}}]}`,
		`{"choices":[],"usage":{"prompt_tokens":15,"completion_tokens":3,"total_tokens":18}}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var final talk.Response
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Done {
			final = resp
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if final.Usage == nil {
		t.Fatal("expected usage on done response")
	}
	if final.Usage.InputTokens != 15 {
		t.Errorf("InputTokens = %d, want 15", final.Usage.InputTokens)
	}
	if final.Usage.OutputTokens != 3 {
		t.Errorf("OutputTokens = %d, want 3", final.Usage.OutputTokens)
	}
}

func TestChat_StreamSendsStreamOptions(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"ok"}}]}`)
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	so, ok := gotBody["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("stream_options should be set in request")
	}
	if so["include_usage"] != true {
		t.Errorf("include_usage = %v, want true", so["include_usage"])
	}
}

func TestChat_StreamNoUsageWhenAbsent(t *testing.T) {
	server := sseServer(
		`{"choices":[{"delta":{"content":"hi"}}]}`,
		"[DONE]",
	)
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var final talk.Response
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Done {
			final = resp
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if final.Usage != nil {
		t.Error("usage should be nil when not in stream")
	}
}

func TestChat_StreamAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error": {"message": "Rate limit exceeded"}}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	err := client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for 429 response")
	}
}
