package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
)

func TestChat_StreamThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null}}`,
			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" about this."}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
			`event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 4."}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":1}`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`event: message_stop
data: {"type":"message_stop"}`,
		}

		for _, ev := range events {
			fmt.Fprintf(w, "%s\n\n", ev)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	think := true
	req := &talk.Request{
		Model:    "claude-opus-4-6",
		Messages: []talk.Message{{Role: "user", Content: "What is 2+2?"}},
		Stream:   true,
		Think:    &think,
		Options:  map[string]any{"max_tokens": 16000},
	}

	var thinking []string
	var content []string
	var done bool
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Thinking != "" {
			thinking = append(thinking, resp.Thinking)
		}
		if resp.Content != "" {
			content = append(content, resp.Content)
		}
		if resp.Done {
			done = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	joinedThinking := strings.Join(thinking, "")
	if joinedThinking != "Let me think about this." {
		t.Errorf("thinking = %q, want %q", joinedThinking, "Let me think about this.")
	}
	joinedContent := strings.Join(content, "")
	if joinedContent != "The answer is 4." {
		t.Errorf("content = %q, want %q", joinedContent, "The answer is 4.")
	}
	if !done {
		t.Error("should receive done=true")
	}
}

func TestChat_StreamBasic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null}}`,
			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`event: message_stop
data: {"type":"message_stop"}`,
		}

		for _, ev := range events {
			fmt.Fprintf(w, "%s\n\n", ev)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &talk.Request{
		Model:    "claude-opus-4-6",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
		Options:  map[string]any{"max_tokens": 1024},
	}

	var tokens []string
	var done bool
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if resp.Content != "" {
			tokens = append(tokens, resp.Content)
		}
		if resp.Done {
			done = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	joined := strings.Join(tokens, "")
	if joined != "Hello world" {
		t.Errorf("content = %q, want %q", joined, "Hello world")
	}
	if !done {
		t.Error("should receive done=true")
	}
}

func TestChat_StreamToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[]}}`,
			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather"}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Sydney\"}"}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
			`event: message_stop
data: {"type":"message_stop"}`,
		}

		for _, ev := range events {
			fmt.Fprintf(w, "%s\n\n", ev)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &talk.Request{
		Model:    "claude-opus-4-6",
		Messages: []talk.Message{{Role: "user", Content: "Weather?"}},
		Stream:   true,
		Tools: []tool.ToolDef{{
			Name: "get_weather",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"city"},
				Properties: map[string]tool.PropertySchema{
					"city": {Type: "string"},
				},
			},
		}},
		Options: map[string]any{"max_tokens": 1024},
	}

	var toolCalls []talk.ToolCall
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		if len(resp.ToolCalls) > 0 {
			toolCalls = resp.ToolCalls
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
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

func TestChat_StreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		fmt.Fprintf(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &talk.Request{
		Model:    "claude-opus-4-6",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
		Options:  map[string]any{"max_tokens": 1024},
	}

	err := client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for stream error event")
	}
	if !strings.Contains(err.Error(), "overloaded") {
		t.Errorf("error = %q, want to contain 'overloaded'", err.Error())
	}
}

func TestChat_StreamTextThenToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		events := []string{
			`event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[]}}`,
			`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
			`event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"search"}}`,
			`event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"test\"}"}}`,
			`event: content_block_stop
data: {"type":"content_block_stop","index":1}`,
			`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
			`event: message_stop
data: {"type":"message_stop"}`,
		}

		for _, ev := range events {
			fmt.Fprintf(w, "%s\n\n", ev)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := NewClient(server.URL, "key")
	req := &talk.Request{
		Model:    "claude-opus-4-6",
		Messages: []talk.Message{{Role: "user", Content: "search test"}},
		Stream:   true,
		Options:  map[string]any{"max_tokens": 1024},
	}

	var content string
	var toolCalls []talk.ToolCall
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		content += resp.Content
		if len(resp.ToolCalls) > 0 {
			toolCalls = resp.ToolCalls
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if content != "Let me check." {
		t.Errorf("content = %q, want %q", content, "Let me check.")
	}
	if len(toolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(toolCalls))
	}
	if toolCalls[0].Name != "search" {
		t.Errorf("name = %q, want search", toolCalls[0].Name)
	}
}
