package cloudflare

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
	var _ loop.LLMClient = NewClient("http://example.com", "token", WithHTTPClient(http.DefaultClient))
}

func TestChat_BasicResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}

		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result: chatCompletion{
				Choices: []choice{{
					Message: responseMessage{Content: "4"},
				}},
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-token")
	req := &loop.Request{
		Model:    "@cf/qwen/qwen3-30b-a3b-fp8",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2?"}},
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

func TestChat_WithThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result: chatCompletion{
				Choices: []choice{{
					Message: responseMessage{
						Content:          "4",
						ReasoningContent: "2+2 equals 4",
					},
				}},
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2?"}},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})
	if got.Thinking != "2+2 equals 4" {
		t.Errorf("thinking = %q, want %q", got.Thinking, "2+2 equals 4")
	}
}

func TestChat_NoThinkPrefix(t *testing.T) {
	var gotMessages []message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		json.NewDecoder(r.Body).Decode(&body)
		gotMessages = body.Messages

		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "4"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	think := false
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2?"}},
		Think:    &think,
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotMessages) != 1 {
		t.Fatalf("got %d messages, want 1", len(gotMessages))
	}
	if gotMessages[0].Content != "/no_think What is 2+2?" {
		t.Errorf("content = %q, want %q", gotMessages[0].Content, "/no_think What is 2+2?")
	}
}

func TestChat_ThinkEnabled_NoPrefix(t *testing.T) {
	var gotMessages []message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		json.NewDecoder(r.Body).Decode(&body)
		gotMessages = body.Messages

		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "4"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	think := true
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2?"}},
		Think:    &think,
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotMessages[0].Content != "What is 2+2?" {
		t.Errorf("content = %q, should not have /no_think prefix", gotMessages[0].Content)
	}
}

func TestChat_WithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result: chatCompletion{
				Choices: []choice{{
					Message: responseMessage{
						ToolCalls: []responseToolCall{{
							Function: responseToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"city":"Sydney"}`,
							},
						}},
					},
				}},
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
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

func TestChat_ToolsSentInRequest(t *testing.T) {
	var gotTools []toolDef
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		json.NewDecoder(r.Body).Decode(&body)
		gotTools = body.Tools

		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "ok"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
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
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotTools) != 1 {
		t.Fatalf("got %d tools, want 1", len(gotTools))
	}
	if gotTools[0].Type != "function" {
		t.Errorf("type = %q, want function", gotTools[0].Type)
	}
	if gotTools[0].Function.Name != "search" {
		t.Errorf("name = %q, want search", gotTools[0].Function.Name)
	}
	if len(gotTools[0].Function.Parameters.Required) != 1 {
		t.Errorf("required = %v", gotTools[0].Function.Parameters.Required)
	}
	if gotTools[0].Function.Parameters.Properties["query"].Type != "string" {
		t.Errorf("query type = %q", gotTools[0].Function.Parameters.Properties["query"].Type)
	}
}

func TestChat_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error": "unauthorized"}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "bad-token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	err := client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

func TestChat_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if !got.Done {
		t.Error("empty choices should return done=true")
	}
	if got.Content != "" {
		t.Errorf("content = %q, want empty", got.Content)
	}
}

func TestChat_ModelInURL(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "ok"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL+"/v1/gateway", "token")
	req := &loop.Request{
		Model:    "@cf/qwen/qwen3-30b-a3b-fp8",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotPath != "/v1/gateway/@cf/qwen/qwen3-30b-a3b-fp8" {
		t.Errorf("path = %q, want /v1/gateway/@cf/qwen/qwen3-30b-a3b-fp8", gotPath)
	}
}

func TestChat_MessagesWithToolCalls(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "done"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model: "model",
		Messages: []loop.Message{
			{Role: "user", Content: "Weather?"},
			{Role: "assistant", ToolCalls: []loop.ToolCall{
				{Name: "get_weather", Arguments: map[string]any{"city": "Sydney"}},
			}},
			{Role: "tool", Content: "Sunny, 22°C"},
		},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if len(gotBody.Messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(gotBody.Messages))
	}
	if len(gotBody.Messages[1].ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(gotBody.Messages[1].ToolCalls))
	}
	if gotBody.Messages[1].ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q", gotBody.Messages[1].ToolCalls[0].Function.Name)
	}
}

func TestChat_MaxTokensFromOptions(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result:  chatCompletion{Choices: []choice{{Message: responseMessage{Content: "ok"}}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 500},
	}

	client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })

	if gotBody.MaxTokens != 500 {
		t.Errorf("max_tokens = %d, want 500", gotBody.MaxTokens)
	}
}

func TestChat_APISuccessFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{Success: false})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "hi"}},
	}

	err := client.Chat(context.Background(), req, func(resp loop.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for success=false")
	}
}

func TestChat_ToolCallInContent_Fallback(t *testing.T) {
	// Model returns a tool call as JSON in content text instead of
	// using the structured tool_calls field. The stream.ToolCallMatcher
	// should extract it.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result: chatCompletion{
				Choices: []choice{{
					Message: responseMessage{
						Content: `{"name": "get_weather", "arguments": {"city": "Sydney"}}`,
					},
				}},
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "Weather in Sydney?"}},
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
	if got.Content != "" {
		t.Errorf("content should be cleared when tool call extracted, got %q", got.Content)
	}
}

func TestChat_ToolCallInCodeFence_Fallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			Success: true,
			Result: chatCompletion{
				Choices: []choice{{
					Message: responseMessage{
						Content: "```json\n{\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}\n```",
					},
				}},
			},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &loop.Request{
		Model:    "model",
		Messages: []loop.Message{{Role: "user", Content: "search for test"}},
	}

	var got loop.Response
	client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})

	if len(got.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "search" {
		t.Errorf("name = %q, want search", got.ToolCalls[0].Name)
	}
}
