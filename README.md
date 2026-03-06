# axon-talk

LLM provider adapters for [axon-loop](https://github.com/benaskins/axon-loop). Part of [lamina](https://github.com/benaskins/lamina) — each axon package can be used independently.

Each subpackage implements `loop.LLMClient` for a specific backend.

## Providers

| Package | Backend |
|---------|---------|
| `ollama` | [Ollama](https://ollama.com) |

## Usage

```go
import (
    loop "github.com/benaskins/axon-loop"
    "github.com/benaskins/axon-talk/ollama"
)

client, err := ollama.NewClientFromEnvironment()
// client implements loop.LLMClient

result, err := loop.Run(ctx, client, &loop.ChatRequest{
    Model:    "llama3.2",
    Messages: messages,
    Stream:   true,
}, nil, nil, loop.Callbacks{
    OnToken: func(token string) { fmt.Print(token) },
})
```

## License

MIT — see [LICENSE](LICENSE).
