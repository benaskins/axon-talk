# axon-talk

LLM provider adapters implementing a provider-agnostic interface. Each subpackage implements `talk.LLMClient` for a specific backend, translating axon-talk's provider-agnostic request/response types into native API calls.

Import: `github.com/benaskins/axon-talk`

## Providers

| Package      | Backend              | Constructor                            |
|--------------|----------------------|----------------------------------------|
| `ollama`     | Ollama               | `ollama.NewClientFromEnvironment()`    |
| `anthropic`  | Anthropic Messages API | `anthropic.NewClient(baseURL, key)`  |
| `openai`     | OpenAI-compatible APIs | `openai.NewClient(baseURL, token)`   |
| `cloudflare` | Cloudflare Workers AI | `cloudflare.NewClient(baseURL, token)` |

The `openai` package works with any provider that speaks the OpenAI `/v1/chat/completions` protocol (OpenAI, Gemini, Grok, Groq, Together, Fireworks, Azure OpenAI, etc.).

## Key files

- `doc.go` — package doc
- `talk.go` — core types and interface definition
- `ollama/ollama.go` — Ollama adapter
- `anthropic/anthropic.go` — Anthropic adapter (with streaming, gateway token option)
- `cloudflare/cloudflare.go` — Cloudflare Workers AI adapter
- `cloudflare/sse.go` — SSE stream parser for Cloudflare
- `openai/openai.go` — OpenAI-compatible adapter
- `openai/sse.go` — SSE stream parser for OpenAI protocol
- `example/main.go` — usage example

## Dependencies

- `axon` — HTTP utilities
- `axon-tool` — tool definitions for function calling
- `ollama/ollama` — Ollama Go client (ollama provider only)

## Build & Test

```bash
go test ./...
go vet ./...
```

Some tests require a running provider (Ollama, Anthropic, etc.) and are skipped when unavailable.