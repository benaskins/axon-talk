//go:build ignore

// Example demonstrates wiring an OpenAI-compatible provider into axon-loop.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/openai"
)

func main() {
	ctx := context.Background()

	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1" // llama-server default
	}
	token := os.Getenv("OPENAI_API_KEY")

	client := openai.NewClient(baseURL, token)

	req := &talk.Request{
		Model: "llama3.2",
		Messages: []talk.Message{
			{Role: "user", Content: "Say hello in one sentence."},
		},
		Stream: true,
	}

	err := client.Chat(ctx, req, func(resp talk.Response) error {
		fmt.Print(resp.Content)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
