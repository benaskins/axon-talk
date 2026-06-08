---
module: github.com/benaskins/axon-talk
kind: library
---

# axon-talk

LLM provider adapters implementing a provider-agnostic interface. Each subpackage implements `talk.LLMClient` for a specific backend, translating axon-talk's provider-agnostic request/response types into native API calls.

Import: `github.com/benaskins/axon-talk`

## Providers

| Package      | Backend                | Constructor                          |
|--------------|------------------------|--------------------------------------|
| `anthropic`  | Anthropic Messages API | `anthropic.NewClient(baseURL, key)`  |
| `bedrock`    | Claude via Amazon Bedrock | `bedrock.NewClient(modelID, invoke)` |
| `openai`     | OpenAI-compatible APIs | `openai.NewClient(baseURL, token)`   |

The `openai` package works with any provider that speaks the OpenAI `/v1/chat/completions` protocol (OpenAI, OpenRouter, Gemini, Grok, Groq, Together, Fireworks, Azure OpenAI, etc.).

The `bedrock` package reuses the `anthropic` translation and diverts its HTTP onto a caller-supplied `InvokeFunc`, so axon-talk takes no AWS SDK dependency — the consumer wires `bedrockruntime.InvokeModel` (and owns credentials/region/SigV4). Buffered only.

## Key files

- `doc.go` -- package doc
- `talk.go` -- core types and interface definition
- `anthropic/anthropic.go` -- Anthropic adapter (with streaming, gateway token option)
- `bedrock/bedrock.go` -- Claude-via-Bedrock adapter (reuses anthropic; AWS SDK injected via InvokeFunc)
- `openai/openai.go` -- OpenAI-compatible adapter
- `openai/sse.go` -- SSE stream parser for OpenAI protocol
- `example/main.go` -- usage example

## Dependencies

- `axon-tape` -- token stream filtering
- `axon-tool` -- tool definitions for function calling

## Build & Test

```bash
go test ./...
go vet ./...
```

Some tests require a running provider (Anthropic, etc.) and are skipped when unavailable.
