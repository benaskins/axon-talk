// Package cloudflare provides a talk.LLMClient implementation for
// Cloudflare Workers AI via the AI Gateway. It speaks the OpenAI-compatible
// chat completions API that Workers AI exposes. No external dependencies
// beyond net/http and encoding/json.
package cloudflare

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

// Client implements talk.LLMClient for Cloudflare Workers AI.
type Client struct {
	baseURL      string
	token        string
	gatewayToken string // optional Cloudflare AI Gateway auth token
	httpClient   *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets a custom http.Client (useful for testing).
func WithHTTPClient(c *http.Client) Option {
	return func(cl *Client) { cl.httpClient = c }
}

// WithGatewayToken sets a Cloudflare AI Gateway authentication token.
// When set, requests include the cf-aig-authorization header.
func WithGatewayToken(token string) Option {
	return func(cl *Client) { cl.gatewayToken = token }
}

// NewClient creates a Client that talks to Cloudflare Workers AI
// through the AI Gateway.
//
// baseURL is the gateway endpoint, e.g.:
//
//	https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/workers-ai
//
// token is the Cloudflare API token with Workers AI permissions.
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

// Chat sends a request to Workers AI and delivers the response through fn.
// When req.Stream is true, the response is streamed via SSE with incremental
// token delivery. Otherwise, the full response is returned in one call.
func (c *Client) Chat(ctx context.Context, req *talk.Request, fn func(talk.Response) error) error {
	body := buildRequest(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("cloudflare: marshal request: %w", err)
	}

	url := c.baseURL + "/" + req.Model
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("cloudflare: create request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Content-Type", "application/json")
	if c.gatewayToken != "" {
		httpReq.Header.Set("cf-aig-authorization", "Bearer "+c.gatewayToken)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("cloudflare: do request: %w", err)
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
		return fmt.Errorf("cloudflare: read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return fmt.Errorf("cloudflare: status %d: %s", httpResp.StatusCode, respBody)
	}

	var apiResp apiResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return fmt.Errorf("cloudflare: decode response: %w", err)
	}

	if !apiResp.Success {
		return fmt.Errorf("cloudflare: api error: %s", respBody)
	}

	resp := fromResponse(apiResp.Result)
	normalizeToolCallArgs(&resp, tools)
	return fn(resp)
}

func (c *Client) handleStream(httpResp *http.Response, tools []tool.ToolDef, fn func(talk.Response) error) error {
	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return fmt.Errorf("cloudflare: status %d: %s", httpResp.StatusCode, respBody)
	}

	// Accumulate structured tool calls by index.
	type pendingToolCall struct {
		name string
		args strings.Builder
	}
	pending := make(map[int]*pendingToolCall)

	// StreamFilter detects tool calls that models dump as JSON in content
	// text instead of using the structured tool_calls field. Content tokens
	// flow through immediately unless ToolCallMatcher signals PartialMatch
	// (JSON might be forming), in which case they're held in the buffer.
	var filterErr error
	filter := stream.NewStreamFilter(
		func(token string) {
			if filterErr != nil {
				return
			}
			filterErr = fn(talk.Response{Content: token})
		},
		[]stream.Matcher{stream.NewToolCallMatcher()},
		stream.DefaultMaxBuffer,
	)

	err := parseSSE(httpResp.Body, func(ev sseEvent) error {
		if ev.Done {
			// Flush the filter — emits remaining buffered content or
			// extracts a tool call if the buffer matches.
			action := filter.Flush()
			if filterErr != nil {
				return filterErr
			}
			toolCalls := collectFilterToolCalls(action)

			// Assemble accumulated structured tool calls.
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

		// Feed content tokens through the StreamFilter.
		if ev.Delta.Content != "" {
			action := filter.Write(ev.Delta.Content)
			if filterErr != nil {
				return filterErr
			}
			// If the filter detected a complete tool call in content,
			// emit it immediately.
			if toolCalls := collectFilterToolCalls(action); len(toolCalls) > 0 {
				if err := fn(talk.Response{ToolCalls: toolCalls}); err != nil {
					return err
				}
			}
		}

		// Emit thinking tokens directly (no filtering needed).
		if ev.Delta.ReasoningContent != "" {
			if err := fn(talk.Response{Thinking: ev.Delta.ReasoningContent}); err != nil {
				return err
			}
		}

		return nil
	})

	if err != nil {
		return err
	}
	return filterErr
}

// collectFilterToolCalls converts a StreamFilter action into tool calls.
func collectFilterToolCalls(action stream.FilterAction) []talk.ToolCall {
	tca, ok := action.(stream.ToolCallAction)
	if !ok || len(tca.Calls) == 0 {
		return nil
	}
	calls := make([]talk.ToolCall, len(tca.Calls))
	for i, tc := range tca.Calls {
		calls[i] = talk.ToolCall{
			Name:      tc.Name,
			Arguments: tc.Arguments,
		}
	}
	return calls
}

func buildRequest(req *talk.Request) chatRequest {
	cr := chatRequest{
		Messages: toMessages(req.Messages, req.Think),
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
		t := true
		cr.ParallelToolCalls = &t
	}

	if req.Stream {
		s := true
		cr.Stream = &s
	}

	return cr
}

func toMessages(msgs []talk.Message, think *bool) []message {
	out := make([]message, 0, len(msgs))
	// Track pending tool call IDs so we can match tool results.
	var pendingCallIDs []string

	for i, m := range msgs {
		content := m.Content
		if think != nil && !*think && m.Role == "user" && i == len(msgs)-1 {
			content = "/no_think " + content
		}

		msg := message{
			Role:    string(m.Role),
			Content: content,
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

func fromResponse(result chatCompletion) talk.Response {
	// OpenAI-compatible format: choices[].message with nested tool_calls.
	if len(result.Choices) > 0 {
		return fromOpenAIResponse(result.Choices[0])
	}

	// Native Workers AI format: top-level response + tool_calls.
	if result.Response != "" || len(result.ToolCalls) > 0 {
		return fromNativeResponse(result)
	}

	return talk.Response{Done: true}
}

func fromOpenAIResponse(choice choice) talk.Response {
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

	// Fallback: if no structured tool calls, check if the model dumped
	// a tool call as JSON in the content text. The ToolCallMatcher from
	// axon/stream handles fenced code blocks, bare JSON objects/arrays.
	if len(resp.ToolCalls) == 0 && resp.Content != "" {
		resp = tryMatchContentToolCalls(resp)
	}

	return resp
}

func fromNativeResponse(result chatCompletion) talk.Response {
	resp := talk.Response{
		Content: strings.TrimLeft(result.Response, "\n"),
		Done:    true,
	}

	if len(result.ToolCalls) > 0 {
		resp.ToolCalls = make([]talk.ToolCall, len(result.ToolCalls))
		for i, tc := range result.ToolCalls {
			resp.ToolCalls[i] = talk.ToolCall{
				Name:      tc.Name,
				Arguments: tc.Arguments,
			}
		}
	}

	if len(resp.ToolCalls) == 0 && resp.Content != "" {
		resp = tryMatchContentToolCalls(resp)
	}

	return resp
}

func tryMatchContentToolCalls(resp talk.Response) talk.Response {
	matcher := stream.NewToolCallMatcher()
	if matcher.Scan([]byte(resp.Content), "") == stream.FullMatch {
		if action := matcher.Extract([]byte(resp.Content)); action != (stream.ContinueAction{}) {
			if tca, ok := action.(stream.ToolCallAction); ok {
				resp.ToolCalls = make([]talk.ToolCall, len(tca.Calls))
				for i, tc := range tca.Calls {
					resp.ToolCalls[i] = talk.ToolCall{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					}
				}
				resp.Content = ""
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

// Wire types for the OpenAI-compatible Workers AI API.

type chatRequest struct {
	Messages          []message `json:"messages"`
	MaxTokens         int       `json:"max_tokens,omitempty"`
	Temperature       *float64  `json:"temperature,omitempty"`
	Tools             []toolDef `json:"tools,omitempty"`
	ParallelToolCalls *bool     `json:"parallel_tool_calls,omitempty"`
	Stream            *bool     `json:"stream,omitempty"`
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
	Type       string                  `json:"type"`
	Required   []string                `json:"required,omitempty"`
	Properties map[string]propertyDef  `json:"properties,omitempty"`
}

type propertyDef struct {
	Type        string                  `json:"type"`
	Description string                  `json:"description,omitempty"`
	Enum        []any                   `json:"enum,omitempty"`
	Default     any                     `json:"default,omitempty"`
	Items       *propertyDef            `json:"items,omitempty"`
	Properties  map[string]propertyDef  `json:"properties,omitempty"`
	Required    []string                `json:"required,omitempty"`
}

type apiResponse struct {
	Success bool           `json:"success"`
	Result  chatCompletion `json:"result"`
}

type chatCompletion struct {
	// OpenAI-compatible format: result.choices[].message.tool_calls
	Choices []choice `json:"choices"`
	// Native Workers AI format: result.response + result.tool_calls
	Response  string           `json:"response"`
	ToolCalls []nativeToolCall `json:"tool_calls"`
}

type nativeToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
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
