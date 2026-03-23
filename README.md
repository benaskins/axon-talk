# axon-talk

> Primitives · Part of the [lamina](https://github.com/benaskins/lamina-mono) workspace

LLM provider adapters for [axon-loop](https://github.com/benaskins/axon-loop).
Each subpackage implements `loop.LLMClient` for a specific backend, translating
axon-loop's provider-agnostic request/response types into native API calls.

## Getting started

```bash
go get github.com/benaskins/axon-talk@latest
```

```go
import (
    loop "github.com/benaskins/axon-loop"
    "github.com/benaskins/axon-talk/ollama"
)

client, err := ollama.NewClientFromEnvironment()
if err != nil {
    log.Fatal(err)
}

// client implements loop.LLMClient — pass it to loop.Run
result, err := loop.Run(ctx, client, &loop.Request{
    Model:    "llama3.2",
    Messages: messages,
    Stream:   true,
}, nil, nil, loop.Callbacks{
    OnToken: func(token string) { fmt.Print(token) },
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
