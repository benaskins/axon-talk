@AGENTS.md

## Conventions
- Each provider lives in its own sub-package (`ollama/`, `anthropic/`, `openai/`, `cloudflare/`)
- All adapters implement `talk.LLMClient` from the root package
- Request/Response types in root package are provider-agnostic — providers translate internally
- SSE parsers are per-provider (`openai/sse.go`, `cloudflare/sse.go`) since protocols differ

## Constraints
- Never add provider-specific types to the root `talk` package
- New providers must be separate sub-packages — no conditionals in existing adapters
- Depends on axon (HTTP client) and axon-tool (tool types); do not add other axon-* deps
- The `openai` package must stay compatible with any OpenAI-protocol API, not just OpenAI

## Testing
- `go test ./...` — unit tests run without providers
- Integration tests require running providers and are skipped when unavailable
- `go vet ./...` for lint
