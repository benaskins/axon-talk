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

| Package  | Backend                        | Constructor                          |
|----------|--------------------------------|--------------------------------------|
| `ollama` | [Ollama](https://ollama.com)   | `ollama.NewClientFromEnvironment()`  |

## License

MIT
