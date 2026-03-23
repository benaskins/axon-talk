// Package openai provides a talk.LLMClient implementation for any
// OpenAI-compatible chat completions API. It works with OpenAI, Azure
// OpenAI, Gemini, Grok, Groq, Together, Fireworks, and any other
// provider that speaks the /v1/chat/completions protocol.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/benaskins/axon/stream"
	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
)

// Client implements talk.LLMClient for OpenAI-compatible APIs.
type Client struct {
	baseURL      string
	token        string
	gatewayToken string // optional gateway auth token (e.g. Cloudflare AI Gateway)
	httpClient   *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets a custom http.Client (useful for testing).
func WithHTTPClient(c *http.Client) Option {
	return func(cl *Client) { cl.httpClient = c }
}

// WithGatewayToken sets a gateway authentication token.
// When set, requests include the cf-aig-authorization header
// for use with Cloudflare AI Gateway or similar proxies.
func WithGatewayToken(token string) Option {
	return func(cl *Client) { cl.gatewayToken = token }
}

// NewClient creates a Client that talks to an OpenAI-compatible API.
//
// baseURL is the API root, e.g. "https://api.openai.com" or
// "https://generativelanguage.googleapis.com/v1beta/openai".
// token is the bearer token for authentication.
func NewClient(baseURL, token string, opts ...Option) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		token:      token,
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Chat sends a request to the chat completions endpoint and delivers
// the response through fn. When req.Stream is true, the response is
// streamed via SSE with incremental token delivery.
func (c *Client) Chat(ctx context.Context, req *talk.Request, fn func(talk.Response) error) error {
	body := buildRequest(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("openai: marshal request: %w", err)
	}

	url := c.baseURL + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("openai: create request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Content-Type", "application/json")
	if c.gatewayToken != "" {
		httpReq.Header.Set("cf-aig-authorization", "Bearer "+c.gatewayToken)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("openai: do request: %w", err)
	}
	defer httpResp.Body.Close()

	if req.Stream {
		return c.handleStream(httpResp, req.Tools, fn)
	}
	return c.handleFull(httpResp, req.Tools, fn)
}

func (c *Client) handleFull(httpResp *http.Response, tools []tool.ToolDef, fn func(talk.Response) error) error {
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return fmt.Errorf("openai: read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return &talk.StatusError{StatusCode: httpResp.StatusCode, Body: string(respBody), Provider: "openai"}
	}

	var completion chatCompletion
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}

	resp := fromResponse(completion)
	normalizeToolCallArgs(&resp, tools)
	return fn(resp)
}

func (c *Client) handleStream(httpResp *http.Response, tools []tool.ToolDef, fn func(talk.Response) error) error {
	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return &talk.StatusError{StatusCode: httpResp.StatusCode, Body: string(respBody), Provider: "openai"}
	}

	// Accumulate structured tool calls by index.
	type pendingToolCall struct {
		name string
		args strings.Builder
	}
	pending := make(map[int]*pendingToolCall)

	err := parseSSE(httpResp.Body, func(ev sseEvent) error {
		if ev.Done {
			var toolCalls []talk.ToolCall
			for i := 0; i < len(pending); i++ {
				tc := pending[i]
				var args map[string]any
				if tc.args.Len() > 0 {
					json.Unmarshal([]byte(tc.args.String()), &args)
				}
				toolCalls = append(toolCalls, talk.ToolCall{
					Name:      tc.name,
					Arguments: args,
				})
			}

			resp := talk.Response{Done: true, ToolCalls: toolCalls}
			normalizeToolCallArgs(&resp, tools)
			return fn(resp)
		}

		// Accumulate structured tool call deltas.
		for _, tc := range ev.Delta.ToolCalls {
			p, ok := pending[tc.Index]
			if !ok {
				p = &pendingToolCall{}
				pending[tc.Index] = p
			}
			if tc.Function.Name != "" {
				p.name = tc.Function.Name
			}
			p.args.WriteString(tc.Function.Arguments)
		}

		// Emit content tokens directly.
		if ev.Delta.Content != "" {
			if err := fn(talk.Response{Content: ev.Delta.Content}); err != nil {
				return err
			}
		}

		// Emit thinking tokens directly.
		if ev.Delta.ReasoningContent != "" {
			if err := fn(talk.Response{Thinking: ev.Delta.ReasoningContent}); err != nil {
				return err
			}
		}

		return nil
	})

	return err
}

func buildRequest(req *talk.Request) chatRequest {
	cr := chatRequest{
		Model:    req.Model,
		Messages: toMessages(req.Messages),
	}

	if v, ok := req.Options["max_tokens"]; ok {
		if mt, ok := v.(int); ok {
			cr.MaxTokens = mt
		}
	}

	if v, ok := req.Options["temperature"]; ok {
		if t, ok := v.(float64); ok {
			cr.Temperature = &t
		}
	}

	if len(req.Tools) > 0 {
		cr.Tools = toTools(req.Tools)
	}

	if req.Stream {
		s := true
		cr.Stream = &s
	}

	return cr
}

func toMessages(msgs []talk.Message) []message {
	out := make([]message, 0, len(msgs))
	// Track pending tool call IDs so we can match tool results.
	var pendingCallIDs []string

	for _, m := range msgs {
		msg := message{
			Role:    string(m.Role),
			Content: m.Content,
		}

		if len(m.ToolCalls) > 0 {
			msg.ToolCalls = toRequestToolCalls(m.ToolCalls)
			pendingCallIDs = nil
			for _, tc := range msg.ToolCalls {
				pendingCallIDs = append(pendingCallIDs, tc.ID)
			}
		}

		// Match tool result messages with their call IDs.
		if m.Role == "tool" && len(pendingCallIDs) > 0 {
			msg.ToolCallID = pendingCallIDs[0]
			pendingCallIDs = pendingCallIDs[1:]
		}

		out = append(out, msg)
	}
	return out
}

func toRequestToolCalls(calls []talk.ToolCall) []requestToolCall {
	out := make([]requestToolCall, len(calls))
	for i, tc := range calls {
		args, _ := json.Marshal(tc.Arguments)
		out[i] = requestToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: requestToolCallFunction{
				Name:      tc.Name,
				Arguments: string(args),
			},
		}
	}
	return out
}

func toTools(defs []tool.ToolDef) []toolDef {
	out := make([]toolDef, len(defs))
	for i, d := range defs {
		out[i] = toolDef{
			Type: "function",
			Function: functionDef{
				Name:        d.Name,
				Description: d.Description,
				Parameters: parametersDef{
					Type:       d.Parameters.Type,
					Required:   d.Parameters.Required,
					Properties: toPropertyDefs(d.Parameters.Properties),
				},
			},
		}
	}
	return out
}

func toPropertyDefs(props map[string]tool.PropertySchema) map[string]propertyDef {
	if len(props) == 0 {
		return nil
	}
	out := make(map[string]propertyDef, len(props))
	for name, prop := range props {
		out[name] = toPropertyDef(prop)
	}
	return out
}

func toPropertyDef(prop tool.PropertySchema) propertyDef {
	pd := propertyDef{
		Type:        prop.Type,
		Description: prop.Description,
		Enum:        prop.Enum,
		Default:     prop.Default,
		Required:    prop.Required,
		Properties:  toPropertyDefs(prop.Properties),
	}
	if prop.Items != nil {
		items := toPropertyDef(*prop.Items)
		pd.Items = &items
	}
	return pd
}

func fromResponse(completion chatCompletion) talk.Response {
	if len(completion.Choices) == 0 {
		return talk.Response{Done: true}
	}

	choice := completion.Choices[0]
	resp := talk.Response{
		Content:  strings.TrimLeft(choice.Message.Content, "\n"),
		Thinking: strings.TrimSpace(choice.Message.ReasoningContent),
		Done:     true,
	}

	if len(choice.Message.ToolCalls) > 0 {
		resp.ToolCalls = make([]talk.ToolCall, len(choice.Message.ToolCalls))
		for i, tc := range choice.Message.ToolCalls {
			var args map[string]any
			json.Unmarshal([]byte(tc.Function.Arguments), &args)
			resp.ToolCalls[i] = talk.ToolCall{
				Name:      tc.Function.Name,
				Arguments: args,
			}
		}
	}

	return resp
}

// normalizeToolCallArgs coerces tool call argument types to match the
// JSON Schema types declared in each tool's parameter schema.
func normalizeToolCallArgs(resp *talk.Response, tools []tool.ToolDef) {
	if len(resp.ToolCalls) == 0 || len(tools) == 0 {
		return
	}

	toolTypes := make(map[string]map[string]string, len(tools))
	for _, td := range tools {
		types := make(map[string]string, len(td.Parameters.Properties))
		for name, prop := range td.Parameters.Properties {
			types[name] = prop.Type
		}
		toolTypes[td.Name] = types
	}

	for i, tc := range resp.ToolCalls {
		if types, ok := toolTypes[tc.Name]; ok {
			resp.ToolCalls[i].Arguments = stream.NormalizeArguments(tc.Arguments, types)
		}
	}
}

// Wire types for the OpenAI chat completions API.

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Tools       []toolDef `json:"tools,omitempty"`
	Stream      *bool     `json:"stream,omitempty"`
}

type message struct {
	Role       string            `json:"role"`
	Content    string            `json:"content"`
	ToolCalls  []requestToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

type requestToolCall struct {
	ID       string                  `json:"id"`
	Type     string                  `json:"type"`
	Function requestToolCallFunction `json:"function"`
}

type requestToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolDef struct {
	Type     string      `json:"type"`
	Function functionDef `json:"function"`
}

type functionDef struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Parameters  parametersDef `json:"parameters"`
}

type parametersDef struct {
	Type       string                 `json:"type"`
	Required   []string               `json:"required,omitempty"`
	Properties map[string]propertyDef `json:"properties,omitempty"`
}

type propertyDef struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description,omitempty"`
	Enum        []any                  `json:"enum,omitempty"`
	Default     any                    `json:"default,omitempty"`
	Items       *propertyDef           `json:"items,omitempty"`
	Properties  map[string]propertyDef `json:"properties,omitempty"`
	Required    []string               `json:"required,omitempty"`
}

type chatCompletion struct {
	Choices []choice `json:"choices"`
}

type choice struct {
	Message responseMessage `json:"message"`
}

type responseMessage struct {
	Role             string             `json:"role"`
	Content          string             `json:"content"`
	ReasoningContent string             `json:"reasoning_content"`
	ToolCalls        []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	Function responseToolCallFunction `json:"function"`
}

type responseToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
