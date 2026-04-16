@AGENTS.md

## Conventions
- Each provider lives in its own sub-package (`anthropic/`, `openai/`)
- All adapters implement `talk.LLMClient` from the root package
- Request/Response types in root package are provider-agnostic, providers translate internally
- SSE parser in `openai/sse.go` handles the OpenAI streaming protocol

## Constraints
- Never add provider-specific types to the root `talk` package
- New providers must be separate sub-packages — no conditionals in existing adapters
- Depends on axon-tape and axon-tool; do not add other axon-* deps
- The `openai` package must stay compatible with any OpenAI-protocol API, not just OpenAI

## Testing
- `go test ./...` — unit tests run without providers
- Integration tests require running providers and are skipped when unavailable
- `go vet ./...` for lint
