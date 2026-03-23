package talk

import (
	"context"

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

// Response is a provider-agnostic streamed response chunk from an LLM.
type Response struct {
	Content   string
	Thinking  string
	Done      bool
	ToolCalls []ToolCall
}

// LLMClient abstracts communication with an LLM backend.
// Implementations translate to/from provider-specific APIs
// (e.g. Ollama, OpenAI, Anthropic).
type LLMClient interface {
	Chat(ctx context.Context, req *Request, fn func(Response) error) error
}

// RequestOption configures a Request.
type RequestOption func(*Request)

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
