//go:build ignore

// Example demonstrates wiring an Ollama provider into axon-loop.
package main

import (
	"context"
	"fmt"
	"log"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/ollama"
)

func main() {
	ctx := context.Background()

	// Create an LLM client from OLLAMA_HOST (or default localhost:11434).
	client, err := ollama.NewClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// Build a simple request.
	req := &talk.Request{
		Model: "llama3.2",
		Messages: []talk.Message{
			{Role: "user", Content: "Say hello in one sentence."},
		},
		Stream: true,
	}

	// Stream the response, printing tokens as they arrive.
	err = client.Chat(ctx, req, func(resp talk.Response) error {
		fmt.Print(resp.Content)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
