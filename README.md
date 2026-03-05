# axon-talk

LLM provider adapters for [axon-loop](https://github.com/benaskins/axon-loop). Each subpackage implements `loop.ChatClient` for a specific backend.

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
// client implements loop.ChatClient

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
