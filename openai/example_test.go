package openai_test

import (
	"context"
	"fmt"
	"os"

	"github.com/benaskins/axon-talk/openai"
)

// ExampleNewClient_togetherAI shows how to use the openai adapter with
// Together AI. Any OpenAI-compatible provider works the same way — just
// change the base URL and token.
func ExampleNewClient_togetherAI() {
	client := openai.NewClient(
		"https://api.together.xyz",
		os.Getenv("TOGETHER_API_KEY"),
	)

	_ = client // use with axon-loop or call client.Chat directly
	fmt.Println("ok")
	// Output: ok
}

// ExampleNewClient_openRouter shows how to use the openai adapter with
// OpenRouter, including the recommended identity headers.
func ExampleNewClient_openRouter() {
	client := openai.NewClient(
		"https://openrouter.ai/api",
		os.Getenv("OPENROUTER_API_KEY"),
		openai.WithHeaders(map[string]string{
			"X-Title":      "my-service",
			"HTTP-Referer": "https://github.com/benaskins/my-service",
		}),
	)

	_ = client
	fmt.Println("ok")
	// Output: ok
}

// ExampleNewClient_cloudflareGateway shows how to route requests through
// Cloudflare AI Gateway to an upstream provider.
func ExampleNewClient_cloudflareGateway() {
	client := openai.NewClient(
		"https://gateway.ai.cloudflare.com/v1/ACCOUNT/GATEWAY/openai",
		os.Getenv("OPENAI_API_KEY"),
		openai.WithGatewayToken(os.Getenv("CF_AIG_TOKEN")),
	)

	_ = context.Background() // suppress unused import
	_ = client
	fmt.Println("ok")
	// Output: ok
}
