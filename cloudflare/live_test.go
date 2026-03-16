package cloudflare_test

import (
	"context"
	"os"
	"testing"

	loop "github.com/benaskins/axon-loop"
	"github.com/benaskins/axon-talk/cloudflare"
)

func TestLive_BasicInference(t *testing.T) {
	token := os.Getenv("CLOUDFLARE_AXON_GATE_TOKEN")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if token == "" || accountID == "" {
		t.Skip("CLOUDFLARE_AXON_GATE_TOKEN and CLOUDFLARE_ACCOUNT_ID required")
	}

	baseURL := "https://gateway.ai.cloudflare.com/v1/" + accountID + "/axon-gate/workers-ai"
	client := cloudflare.NewClient(baseURL, token)

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
