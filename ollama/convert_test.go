package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
	ollamaapi "github.com/ollama/ollama/api"
)

func TestToMessages_Basic(t *testing.T) {
	msgs := []talk.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
	}

	got := toMessages(msgs)

	if len(got) != 2 {
		t.Fatalf("got %d messages, want 2", len(got))
	}
	if got[0].Role != "user" || got[0].Content != "hello" {
		t.Errorf("msg[0] = %+v", got[0])
	}
	if got[1].Role != "assistant" || got[1].Content != "hi there" {
		t.Errorf("msg[1] = %+v", got[1])
	}
}

func TestToMessages_WithThinking(t *testing.T) {
	msgs := []talk.Message{
		{Role: "assistant", Content: "answer", Thinking: "let me think"},
	}

	got := toMessages(msgs)

	if got[0].Thinking != "let me think" {
		t.Errorf("thinking = %q, want %q", got[0].Thinking, "let me think")
	}
}

func TestToMessages_WithToolCalls(t *testing.T) {
	msgs := []talk.Message{
		{
			Role: "assistant",
			ToolCalls: []talk.ToolCall{
				{Name: "get_weather", Arguments: map[string]any{"city": "Sydney"}},
			},
		},
	}

	got := toMessages(msgs)

	if len(got[0].ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got[0].ToolCalls))
	}
	tc := got[0].ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("name = %q, want %q", tc.Function.Name, "get_weather")
	}
	args := tc.Function.Arguments.ToMap()
	if args["city"] != "Sydney" {
		t.Errorf("city = %v, want Sydney", args["city"])
	}
}

func TestToMessages_Empty(t *testing.T) {
	got := toMessages(nil)
	if len(got) != 0 {
		t.Errorf("got %d messages, want 0", len(got))
	}
}

func TestToToolCalls(t *testing.T) {
	calls := []talk.ToolCall{
		{Name: "search", Arguments: map[string]any{"query": "test", "limit": float64(10)}},
		{Name: "noop", Arguments: map[string]any{}},
	}

	got := toToolCalls(calls)

	if len(got) != 2 {
		t.Fatalf("got %d calls, want 2", len(got))
	}
	if got[0].Function.Name != "search" {
		t.Errorf("name = %q, want %q", got[0].Function.Name, "search")
	}
	args := got[0].Function.Arguments.ToMap()
	if args["query"] != "test" {
		t.Errorf("query = %v, want test", args["query"])
	}
}

func TestToTools(t *testing.T) {
	defs := []tool.ToolDef{
		{
			Name:        "get_weather",
			Description: "Get the weather for a city",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"city"},
				Properties: map[string]tool.PropertySchema{
					"city": {Type: "string", Description: "City name"},
				},
			},
		},
	}

	got := toTools(defs)

	if len(got) != 1 {
		t.Fatalf("got %d tools, want 1", len(got))
	}
	if got[0].Type != "function" {
		t.Errorf("type = %q, want function", got[0].Type)
	}
	if got[0].Function.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", got[0].Function.Name)
	}
	if got[0].Function.Description != "Get the weather for a city" {
		t.Errorf("description = %q", got[0].Function.Description)
	}
	if got[0].Function.Parameters.Type != "object" {
		t.Errorf("params type = %q", got[0].Function.Parameters.Type)
	}
	if len(got[0].Function.Parameters.Required) != 1 || got[0].Function.Parameters.Required[0] != "city" {
		t.Errorf("required = %v", got[0].Function.Parameters.Required)
	}
}

func TestFromResponse_Basic(t *testing.T) {
	resp := ollamaapi.ChatResponse{
		Done: true,
		Message: ollamaapi.Message{
			Content: "hello",
		},
	}

	got := fromResponse(resp)

	if got.Content != "hello" {
		t.Errorf("content = %q, want hello", got.Content)
	}
	if !got.Done {
		t.Error("done should be true")
	}
	if len(got.ToolCalls) != 0 {
		t.Errorf("got %d tool calls, want 0", len(got.ToolCalls))
	}
}

func TestFromResponse_WithThinking(t *testing.T) {
	resp := ollamaapi.ChatResponse{
		Message: ollamaapi.Message{
			Content:  "answer",
			Thinking: "reasoning",
		},
	}

	got := fromResponse(resp)

	if got.Thinking != "reasoning" {
		t.Errorf("thinking = %q, want reasoning", got.Thinking)
	}
}

func TestFromResponse_WithToolCalls(t *testing.T) {
	args := ollamaapi.NewToolCallFunctionArguments()
	args.Set("city", "Melbourne")

	resp := ollamaapi.ChatResponse{
		Message: ollamaapi.Message{
			ToolCalls: []ollamaapi.ToolCall{
				{
					Function: ollamaapi.ToolCallFunction{
						Name:      "get_weather",
						Arguments: args,
					},
				},
			},
		},
	}

	got := fromResponse(resp)

	if len(got.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", got.ToolCalls[0].Name)
	}
	if got.ToolCalls[0].Arguments["city"] != "Melbourne" {
		t.Errorf("city = %v, want Melbourne", got.ToolCalls[0].Arguments["city"])
	}
}

func TestChat_StreamsResponses(t *testing.T) {
	// Mock Ollama server that returns two streamed JSON lines
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %q, want /api/chat", r.URL.Path)
		}

		var req ollamaapi.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "llama3" {
			t.Errorf("model = %q, want llama3", req.Model)
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, _ := w.(http.Flusher)

		chunk1 := ollamaapi.ChatResponse{Message: ollamaapi.Message{Content: "hello "}}
		json.NewEncoder(w).Encode(chunk1)
		flusher.Flush()

		chunk2 := ollamaapi.ChatResponse{Done: true, Message: ollamaapi.Message{Content: "world"}}
		json.NewEncoder(w).Encode(chunk2)
		flusher.Flush()
	}))
	defer server.Close()

	base, _ := url.Parse(server.URL)
	api := ollamaapi.NewClient(base, server.Client())
	client := NewClient(api)

	req := &talk.Request{
		Model:    "llama3",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var responses []talk.Response
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		responses = append(responses, resp)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if len(responses) != 2 {
		t.Fatalf("got %d responses, want 2", len(responses))
	}
	if responses[0].Content != "hello " {
		t.Errorf("response[0].Content = %q, want %q", responses[0].Content, "hello ")
	}
	if responses[1].Content != "world" || !responses[1].Done {
		t.Errorf("response[1] = %+v", responses[1])
	}
}

func TestChat_WithOptions(t *testing.T) {
	var gotOptions map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]json.RawMessage
		json.NewDecoder(r.Body).Decode(&raw)
		if opts, ok := raw["options"]; ok {
			json.Unmarshal(opts, &gotOptions)
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		resp := ollamaapi.ChatResponse{Done: true, Message: ollamaapi.Message{Content: "ok"}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	base, _ := url.Parse(server.URL)
	client := NewClient(ollamaapi.NewClient(base, server.Client()))

	req := &talk.Request{
		Model:    "llama3",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"temperature": float64(0)},
	}

	err := client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if gotOptions == nil {
		t.Error("options not sent to server")
	}
	if gotOptions["temperature"] != float64(0) {
		t.Errorf("temperature = %v, want 0", gotOptions["temperature"])
	}
}

func TestChat_WithThinkFlag(t *testing.T) {
	var gotThink bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]any
		json.NewDecoder(r.Body).Decode(&raw)
		if think, ok := raw["think"]; ok {
			gotThink = think.(bool)
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		resp := ollamaapi.ChatResponse{Done: true, Message: ollamaapi.Message{Content: "thought"}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	base, _ := url.Parse(server.URL)
	client := NewClient(ollamaapi.NewClient(base, server.Client()))

	think := true
	req := &talk.Request{
		Model:    "llama3",
		Messages: []talk.Message{{Role: "user", Content: "think about this"}},
		Think:    &think,
	}

	err := client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if !gotThink {
		t.Error("think flag not sent to server")
	}
}
