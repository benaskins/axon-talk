//go:build ignore

// Example demonstrates wiring an Ollama provider into axon-loop.
package main

import (
	"context"
	"fmt"
	"log"

	loop "github.com/benaskins/axon-loop"
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
	req := &loop.Request{
		Model: "llama3.2",
		Messages: []loop.Message{
			{Role: "user", Content: "Say hello in one sentence."},
		},
		Stream: true,
	}

	// Run the conversation loop, printing tokens as they arrive.
	_, err = loop.Run(ctx, loop.RunConfig{
		Client:  client,
		Request: req,
		Callbacks: loop.Callbacks{
			OnToken: func(token string) { fmt.Print(token) },
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
