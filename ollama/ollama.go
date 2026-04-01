// Package ollama provides a talk.LLMClient implementation for Ollama.
//
// Deprecated: Use the openai package with a llama-server endpoint instead.
// The ollama package will be removed in a future release.
package ollama

import (
	"context"
	"encoding/json"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
	ollamaapi "github.com/ollama/ollama/api"
)

// Client implements talk.LLMClient by translating to/from the Ollama API.
type Client struct {
	api *ollamaapi.Client
}

// NewClient creates a Client from an Ollama API client.
func NewClient(api *ollamaapi.Client) *Client {
	return &Client{api: api}
}

// NewClientFromEnvironment creates a Client using OLLAMA_HOST or the default.
func NewClientFromEnvironment() (*Client, error) {
	api, err := ollamaapi.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}
	return &Client{api: api}, nil
}

// Chat sends a request to Ollama and streams responses back through fn.
func (c *Client) Chat(ctx context.Context, req *talk.Request, fn func(talk.Response) error) error {
	ollamaReq := &ollamaapi.ChatRequest{
		Model:    req.Model,
		Messages: toMessages(req.Messages),
	}

	if req.Stream {
		stream := true
		ollamaReq.Stream = &stream
	}

	if req.Options != nil {
		ollamaReq.Options = filterOllamaOptions(req.Options)
	}

	if schema, ok := req.Options["structured_output"].(map[string]any); ok {
		b, _ := json.Marshal(schema)
		ollamaReq.Format = json.RawMessage(b)
	}

	if req.Think != nil {
		ollamaReq.Think = &ollamaapi.ThinkValue{Value: *req.Think}
	}

	if len(req.Tools) > 0 {
		ollamaReq.Tools = toTools(req.Tools)
	}

	keepAlive := ollamaapi.Duration{Duration: -1}
	ollamaReq.KeepAlive = &keepAlive

	return c.api.Chat(ctx, ollamaReq, func(resp ollamaapi.ChatResponse) error {
		r := fromResponse(resp)
		normalizeToolCallArgs(&r, req.Tools)
		return fn(r)
	})
}

func toMessages(msgs []talk.Message) []ollamaapi.Message {
	out := make([]ollamaapi.Message, len(msgs))
	for i, m := range msgs {
		out[i] = ollamaapi.Message{
			Role:     string(m.Role),
			Content:  m.Content,
			Thinking: m.Thinking,
		}
		if len(m.ToolCalls) > 0 {
			out[i].ToolCalls = toToolCalls(m.ToolCalls)
		}
	}
	return out
}

func toToolCalls(calls []talk.ToolCall) []ollamaapi.ToolCall {
	out := make([]ollamaapi.ToolCall, len(calls))
	for i, tc := range calls {
		args := ollamaapi.NewToolCallFunctionArguments()
		for k, v := range tc.Arguments {
			args.Set(k, v)
		}
		out[i] = ollamaapi.ToolCall{
			Function: ollamaapi.ToolCallFunction{
				Name:      tc.Name,
				Arguments: args,
			},
		}
	}
	return out
}

func toTools(defs []tool.ToolDef) ollamaapi.Tools {
	out := make(ollamaapi.Tools, len(defs))
	for i, d := range defs {
		props := ollamaapi.NewToolPropertiesMap()
		for name, prop := range d.Parameters.Properties {
			props.Set(name, ollamaapi.ToolProperty{
				Type:        ollamaapi.PropertyType{prop.Type},
				Description: prop.Description,
			})
		}
		out[i] = ollamaapi.Tool{
			Type: "function",
			Function: ollamaapi.ToolFunction{
				Name:        d.Name,
				Description: d.Description,
				Parameters: ollamaapi.ToolFunctionParameters{
					Type:       d.Parameters.Type,
					Required:   d.Parameters.Required,
					Properties: props,
				},
			},
		}
	}
	return out
}

// filterOllamaOptions returns a copy of opts without axon-talk internal keys.
func filterOllamaOptions(opts map[string]any) map[string]any {
	out := make(map[string]any, len(opts))
	for k, v := range opts {
		switch k {
		case "structured_output", "anthropic_prompt_caching":
			continue
		}
		out[k] = v
	}
	return out
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
			resp.ToolCalls[i].Arguments = tool.NormalizeArguments(tc.Arguments, types)
		}
	}
}

func fromResponse(resp ollamaapi.ChatResponse) talk.Response {
	r := talk.Response{
		Content:  resp.Message.Content,
		Thinking: resp.Message.Thinking,
		Done:     resp.Done,
	}
	if len(resp.Message.ToolCalls) > 0 {
		r.ToolCalls = make([]talk.ToolCall, len(resp.Message.ToolCalls))
		for i, tc := range resp.Message.ToolCalls {
			r.ToolCalls[i] = talk.ToolCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments.ToMap(),
			}
		}
	}
	return r
}
