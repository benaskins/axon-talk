package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
)

func TestClientImplementsLLMClient(t *testing.T) {
	var _ talk.LLMClient = NewClient("http://example.com", "token")
}

func TestChat_BasicResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}

		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{
				Message: responseMessage{Content: "4"},
			}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "What is 2+2?"}},
	}

	var got talk.Response
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
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

func TestChat_ModelInRequestBody(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o-mini",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotBody.Model != "gpt-4o-mini" {
		t.Errorf("model = %q, want gpt-4o-mini", gotBody.Model)
	}
}

func TestChat_EndpointPath(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotPath != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", gotPath)
	}
}

func TestChat_WithThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{
				Message: responseMessage{
					Content:          "4",
					ReasoningContent: "2+2 equals 4",
				},
			}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "o3",
		Messages: []talk.Message{{Role: "user", Content: "What is 2+2?"}},
	}

	var got talk.Response
	client.Chat(context.Background(), req, func(resp talk.Response) error {
		got = resp
		return nil
	})
	if got.Thinking != "2+2 equals 4" {
		t.Errorf("thinking = %q, want %q", got.Thinking, "2+2 equals 4")
	}
}

func TestChat_WithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(chatCompletion{
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
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "Weather in Sydney?"}},
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

	var got talk.Response
	client.Chat(context.Background(), req, func(resp talk.Response) error {
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

func TestChat_NormalizesToolCallArgs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{
				Message: responseMessage{
					ToolCalls: []responseToolCall{{
						Function: responseToolCallFunction{
							Name:      "set_temp",
							Arguments: `{"degrees":"22","enabled":"true"}`,
						},
					}},
				},
			}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "Set temp to 22"}},
		Tools: []tool.ToolDef{{
			Name: "set_temp",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"degrees": {Type: "number"},
					"enabled": {Type: "boolean"},
				},
			},
		}},
	}

	var got talk.Response
	client.Chat(context.Background(), req, func(resp talk.Response) error {
		got = resp
		return nil
	})

	if len(got.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(got.ToolCalls))
	}
	if v, ok := got.ToolCalls[0].Arguments["degrees"].(float64); !ok || v != 22 {
		t.Errorf("degrees = %v (%T), want 22 (float64)", got.ToolCalls[0].Arguments["degrees"], got.ToolCalls[0].Arguments["degrees"])
	}
	if v, ok := got.ToolCalls[0].Arguments["enabled"].(bool); !ok || !v {
		t.Errorf("enabled = %v (%T), want true (bool)", got.ToolCalls[0].Arguments["enabled"], got.ToolCalls[0].Arguments["enabled"])
	}
}

func TestChat_RichSchemaInRequest(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Tools: []tool.ToolDef{{
			Name: "create_event",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"title", "priority"},
				Properties: map[string]tool.PropertySchema{
					"title": {Type: "string", Description: "Event title"},
					"priority": {
						Type:    "string",
						Enum:    []any{"low", "medium", "high"},
						Default: "medium",
					},
					"tags": {
						Type: "array",
						Items: &tool.PropertySchema{
							Type: "string",
						},
					},
					"location": {
						Type: "object",
						Properties: map[string]tool.PropertySchema{
							"lat": {Type: "number"},
							"lng": {Type: "number"},
						},
						Required: []string{"lat", "lng"},
					},
				},
			},
		}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if len(gotBody.Tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(gotBody.Tools))
	}
	props := gotBody.Tools[0].Function.Parameters.Properties

	if len(props["priority"].Enum) != 3 {
		t.Errorf("priority enum = %v, want 3 values", props["priority"].Enum)
	}
	if props["priority"].Default != "medium" {
		t.Errorf("priority default = %v, want medium", props["priority"].Default)
	}
	if props["tags"].Items == nil || props["tags"].Items.Type != "string" {
		t.Errorf("tags items = %v, want {type: string}", props["tags"].Items)
	}
	if props["location"].Properties["lat"].Type != "number" {
		t.Errorf("location.lat type = %q, want number", props["location"].Properties["lat"].Type)
	}
	if len(props["location"].Required) != 2 {
		t.Errorf("location required = %v, want [lat lng]", props["location"].Required)
	}
}

func TestChat_ToolsSentInRequest(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
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

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if len(gotBody.Tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(gotBody.Tools))
	}
	if gotBody.Tools[0].Type != "function" {
		t.Errorf("type = %q, want function", gotBody.Tools[0].Type)
	}
	if gotBody.Tools[0].Function.Name != "search" {
		t.Errorf("name = %q, want search", gotBody.Tools[0].Function.Name)
	}
	if len(gotBody.Tools[0].Function.Parameters.Required) != 1 {
		t.Errorf("required = %v", gotBody.Tools[0].Function.Parameters.Required)
	}
	if gotBody.Tools[0].Function.Parameters.Properties["query"].Type != "string" {
		t.Errorf("query type = %q", gotBody.Tools[0].Function.Parameters.Properties["query"].Type)
	}
}

func TestChat_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error": {"message": "Incorrect API key"}}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "bad-token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	err := client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

func TestChat_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(chatCompletion{Choices: []choice{}})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	var got talk.Response
	client.Chat(context.Background(), req, func(resp talk.Response) error {
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

func TestChat_ToolCallIDs(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model: "gpt-4o",
		Messages: []talk.Message{
			{Role: "user", Content: "search for go"},
			{Role: "assistant", ToolCalls: []talk.ToolCall{
				{Name: "search", Arguments: map[string]any{"q": "go"}},
			}},
			{Role: "tool", Content: "results: golang.org"},
		},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	assistantMsg := gotBody.Messages[1]
	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].ID == "" {
		t.Error("tool call ID should not be empty")
	}

	toolMsg := gotBody.Messages[2]
	if toolMsg.ToolCallID == "" {
		t.Error("tool_call_id should not be empty")
	}
	if toolMsg.ToolCallID != assistantMsg.ToolCalls[0].ID {
		t.Errorf("tool_call_id %q doesn't match call ID %q", toolMsg.ToolCallID, assistantMsg.ToolCalls[0].ID)
	}
}

func TestChat_GatewayToken(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token", WithGatewayToken("my-gw-token"))
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotHeader != "Bearer my-gw-token" {
		t.Errorf("cf-aig-authorization = %q, want %q", gotHeader, "Bearer my-gw-token")
	}
}

func TestChat_NoGatewayToken(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotHeader != "" {
		t.Errorf("cf-aig-authorization should be empty when no gateway token set, got %q", gotHeader)
	}
}

func TestChat_StructuredOutput(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: `{"name":"Alice","age":30}`}}},
		})
	}))
	defer server.Close()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "number"},
		},
		"required": []any{"name"},
	}

	client := NewClient(server.URL, "token")
	req := talk.NewRequest("gpt-4o",
		[]talk.Message{{Role: "user", Content: "Who is Alice?"}},
		talk.WithStructuredOutput(schema),
	)

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotBody.ResponseFormat == nil {
		t.Fatal("response_format should be set")
	}
	if gotBody.ResponseFormat.Type != "json_schema" {
		t.Errorf("type = %q, want json_schema", gotBody.ResponseFormat.Type)
	}
	if gotBody.ResponseFormat.JSONSchema == nil {
		t.Fatal("json_schema should be set")
	}
	if !gotBody.ResponseFormat.JSONSchema.Strict {
		t.Error("strict should be true")
	}
	if gotBody.ResponseFormat.JSONSchema.Schema["type"] != "object" {
		t.Errorf("schema type = %v", gotBody.ResponseFormat.JSONSchema.Schema["type"])
	}
}

func TestChat_ParallelToolCalls(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")

	t.Run("enabled", func(t *testing.T) {
		req := &talk.Request{
			Model:    "gpt-4o",
			Messages: []talk.Message{{Role: "user", Content: "hi"}},
			Options:  map[string]any{"parallel_tool_calls": true},
		}
		client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
		if gotBody.ParallelToolCalls == nil || *gotBody.ParallelToolCalls != true {
			t.Errorf("parallel_tool_calls = %v, want true", gotBody.ParallelToolCalls)
		}
	})

	t.Run("disabled", func(t *testing.T) {
		req := &talk.Request{
			Model:    "gpt-4o",
			Messages: []talk.Message{{Role: "user", Content: "hi"}},
			Options:  map[string]any{"parallel_tool_calls": false},
		}
		client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
		if gotBody.ParallelToolCalls == nil || *gotBody.ParallelToolCalls != false {
			t.Errorf("parallel_tool_calls = %v, want false", gotBody.ParallelToolCalls)
		}
	})

	t.Run("omitted", func(t *testing.T) {
		gotBody = chatRequest{}
		req := &talk.Request{
			Model:    "gpt-4o",
			Messages: []talk.Message{{Role: "user", Content: "hi"}},
		}
		client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })
		if gotBody.ParallelToolCalls != nil {
			t.Errorf("parallel_tool_calls = %v, want nil", gotBody.ParallelToolCalls)
		}
	})
}

func TestChat_MaxTokensFromOptions(t *testing.T) {
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"max_tokens": 500},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if gotBody.MaxTokens != 500 {
		t.Errorf("max_tokens = %d, want 500", gotBody.MaxTokens)
	}
}

func TestChat_ToolChoiceAutoWhenToolsPresent(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Tools: []tool.ToolDef{{
			Name:        "search",
			Description: "Search",
			Parameters:  tool.ParameterSchema{Type: "object"},
		}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	tc, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("expected tool_choice in request when tools present")
	}
	if tc != "auto" {
		t.Errorf("tool_choice = %v, want auto", tc)
	}
}

func TestChat_ToolChoiceOverrideFromOptions(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
		Tools: []tool.ToolDef{{
			Name:        "search",
			Description: "Search",
			Parameters:  tool.ParameterSchema{Type: "object"},
		}},
		Options: map[string]any{"tool_choice": "required"},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	tc, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("expected tool_choice in request")
	}
	if tc != "required" {
		t.Errorf("tool_choice = %v, want required", tc)
	}
}

func TestChat_NoToolChoiceWithoutTools(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(chatCompletion{
			Choices: []choice{{Message: responseMessage{Content: "ok"}}},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, "token")
	req := &talk.Request{
		Model:    "gpt-4o",
		Messages: []talk.Message{{Role: "user", Content: "hi"}},
	}

	client.Chat(context.Background(), req, func(resp talk.Response) error { return nil })

	if _, ok := gotBody["tool_choice"]; ok {
		t.Error("tool_choice should not be present when no tools provided")
	}
}
