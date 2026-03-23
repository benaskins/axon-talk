package anthropic

import (
	"context"
	"os"
	"testing"

	talk "github.com/benaskins/axon-talk"
)

func TestLive_StreamingDirect(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY required")
	}

	client := NewClient("https://api.anthropic.com", apiKey)

	req := &talk.Request{
		Model: "claude-sonnet-4-6",
		Messages: []talk.Message{
			{Role: "system", Content: "Reply in exactly 3 words."},
			{Role: "user", Content: "Hello"},
		},
		Stream:  true,
		Options: map[string]any{"max_tokens": 32},
	}

	var content string
	var gotDone bool
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		content += resp.Content
		if resp.Done {
			gotDone = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if !gotDone {
		t.Error("never received done=true")
	}
	if content == "" {
		t.Error("empty response content")
	}
	t.Logf("response: %q", content)
}

func TestLive_StreamingViaGateway(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	gwToken := os.Getenv("CLOUDFLARE_AI_GATEWAY_TOKEN")

	if apiKey == "" || accountID == "" || gwToken == "" {
		t.Skip("ANTHROPIC_API_KEY, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_AI_GATEWAY_TOKEN required")
	}

	baseURL := "https://gateway.ai.cloudflare.com/v1/" + accountID + "/axon-gate/anthropic"
	client := NewClient(baseURL, apiKey, WithGatewayToken(gwToken))

	req := &talk.Request{
		Model: "claude-sonnet-4-6",
		Messages: []talk.Message{
			{Role: "system", Content: "Reply in exactly 3 words."},
			{Role: "user", Content: "Hello"},
		},
		Stream:  true,
		Options: map[string]any{"max_tokens": 32},
	}

	var content string
	var gotDone bool
	err := client.Chat(context.Background(), req, func(resp talk.Response) error {
		content += resp.Content
		if resp.Done {
			gotDone = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if !gotDone {
		t.Error("never received done=true")
	}
	if content == "" {
		t.Error("empty response content")
	}
	t.Logf("response: %q", content)
}
