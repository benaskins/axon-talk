package talk

import (
	"context"
	"fmt"

	tool "github.com/benaskins/axon-tool"
)

// Role represents the sender of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a single message in a conversation.
type Message struct {
	Role       Role
	Content    string
	Thinking   string
	ToolCalls  []ToolCall
	ToolCallID string // Correlates a tool result with its originating ToolCall.
}

// ToolCall represents an LLM's decision to invoke a tool.
type ToolCall struct {
	ID        string // Provider-assigned correlation ID (e.g. OpenAI tool_call_id).
	Name      string
	Arguments map[string]any
}

// Request is a provider-agnostic request to an LLM.
type Request struct {
	Model         string
	Messages      []Message
	Tools         []tool.ToolDef
	Stream        bool
	Think         *bool
	Options       map[string]any
	MaxIterations int // Maximum tool-call loop iterations. Defaults to 20 if 0.
	MaxTokens     int // Maximum estimated token budget for messages. 0 means no limit.
}

// Usage reports token consumption for a single LLM call.
type Usage struct {
	InputTokens  int
	OutputTokens int

	// Cache stats (zero when the provider does not support caching).
	CacheCreationInputTokens int
	CacheReadInputTokens     int
}

// TotalTokens returns InputTokens + OutputTokens.
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// Response is a provider-agnostic streamed response chunk from an LLM.
type Response struct {
	Content   string
	Thinking  string
	Done      bool
	ToolCalls []ToolCall
	Usage     *Usage // non-nil on the final chunk (Done=true) when the provider reports it
}

// LLMClient abstracts communication with an LLM backend.
// Implementations translate to/from provider-specific APIs
// (e.g. Ollama, OpenAI, Anthropic).
type LLMClient interface {
	Chat(ctx context.Context, req *Request, fn func(Response) error) error
}

// StatusError is returned by adapters when the provider API responds with
// a non-OK HTTP status. It carries the status code so callers (e.g. retry
// middleware) can make decisions based on it.
type StatusError struct {
	StatusCode int
	Body       string
	Provider   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s: status %d: %s", e.Provider, e.StatusCode, e.Body)
}

// RequestOption configures a Request.
type RequestOption func(*Request)

// WithStructuredOutput returns a RequestOption that requests the LLM to
// respond with JSON conforming to the given schema. Each adapter translates
// this to its provider-specific wire format.
func WithStructuredOutput(schema map[string]any) RequestOption {
	return func(r *Request) {
		if r.Options == nil {
			r.Options = map[string]any{}
		}
		r.Options["structured_output"] = schema
	}
}

// NewRequest creates a Request with the given model, messages, and options.
func NewRequest(model string, messages []Message, opts ...RequestOption) *Request {
	r := &Request{
		Model:    model,
		Messages: messages,
		Options:  map[string]any{},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}
