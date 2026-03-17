package cloudflare

import (
	"context"
	"os"
	"testing"

	loop "github.com/benaskins/axon-loop"
)

func TestLive_BasicInference(t *testing.T) {
	token := os.Getenv("CLOUDFLARE_AXON_GATE_TOKEN")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if token == "" || accountID == "" {
		t.Skip("CLOUDFLARE_AXON_GATE_TOKEN and CLOUDFLARE_ACCOUNT_ID required")
	}

	baseURL := "https://gateway.ai.cloudflare.com/v1/" + accountID + "/axon-gate/workers-ai"
	client := NewClient(baseURL, token)

	think := false
	req := &loop.Request{
		Model:    "@cf/qwen/qwen3-30b-a3b-fp8",
		Messages: []loop.Message{{Role: "user", Content: "What is 2+2? Reply with just the number."}},
		Think:    &think,
	}

	var got loop.Response
	err := client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	t.Logf("content=%q thinking=%q", got.Content, got.Thinking)
	if got.Content == "" {
		t.Error("expected non-empty content")
	}
	if !got.Done {
		t.Error("expected done=true")
	}
}

func TestLive_ToolResultConversation(t *testing.T) {
	token := os.Getenv("CLOUDFLARE_AXON_GATE_TOKEN")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if token == "" || accountID == "" {
		t.Skip("CLOUDFLARE_AXON_GATE_TOKEN and CLOUDFLARE_ACCOUNT_ID required")
	}

	baseURL := "https://gateway.ai.cloudflare.com/v1/" + accountID + "/axon-gate/workers-ai"
	client := NewClient(baseURL, token)

	think := false
	req := &loop.Request{
		Model: "@cf/qwen/qwen3-30b-a3b-fp8",
		Messages: []loop.Message{
			{Role: "system", Content: "You are a helpful assistant. Ask one question at a time."},
			{Role: "user", Content: "What's the weather in Sydney?"},
			{Role: "assistant", ToolCalls: []loop.ToolCall{
				{Name: "get_weather", Arguments: map[string]any{"city": "Sydney"}},
			}},
			{Role: "tool", Content: "Sydney: 22°C, sunny, light breeze"},
		},
		Think:   &think,
		Options: map[string]any{"max_tokens": 200},
	}

	var got loop.Response
	err := client.Chat(context.Background(), req, func(resp loop.Response) error {
		got = resp
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	t.Logf("content=%q (%d chars)", got.Content, len(got.Content))
	if got.Content == "" {
		t.Error("expected non-empty response after tool result")
	}
}
