# axon-talk

> Primitives · Part of the [lamina](https://github.com/benaskins/lamina-mono) workspace

LLM provider adapters implementing a provider-agnostic interface.
Each subpackage implements `talk.LLMClient` for a specific backend, translating
axon-talk's provider-agnostic request/response types into native API calls.

## Getting started

```bash
go get github.com/benaskins/axon-talk@latest
```

```go
import (
    "github.com/benaskins/axon-talk"
    "github.com/benaskins/axon-talk/ollama"
)

client, err := ollama.NewClientFromEnvironment()
if err != nil {
    log.Fatal(err)
}

// client implements talk.LLMClient
req := &talk.Request{
    Model: "llama3.2",
    Messages: []talk.Message{
        {Role: talk.RoleUser, Content: "Say hello in one sentence."},
    },
    Stream: true,
}

err = client.Chat(ctx, req, func(resp talk.Response) error {
    fmt.Print(resp.Content) // Print tokens as they arrive
    return nil
})
```

## Providers

| Package      | Backend                        | Constructor                          |
|--------------|--------------------------------|--------------------------------------|
| `ollama`     | [Ollama](https://ollama.com)   | `ollama.NewClientFromEnvironment()`  |
| `anthropic`  | Anthropic Messages API         | `anthropic.NewClient(baseURL, key)`  |
| `openai`     | OpenAI-compatible APIs         | `openai.NewClient(baseURL, token)`   |
| `cloudflare` | Cloudflare Workers AI          | `cloudflare.NewClient(baseURL, token)` |

The `openai` package works with any provider that speaks the OpenAI
`/v1/chat/completions` protocol: OpenAI, Gemini, Grok, Groq, Together,
Fireworks, Azure OpenAI, and others.

## License

MIT