// Package ollama provides a loop.LLMClient implementation for Ollama.
package ollama

import (
	"context"

	loop "github.com/benaskins/axon-loop"
	tool "github.com/benaskins/axon-tool"
	ollamaapi "github.com/ollama/ollama/api"
)

// Client implements loop.LLMClient by translating to/from the Ollama API.
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
func (c *Client) Chat(ctx context.Context, req *loop.Request, fn func(loop.Response) error) error {
	ollamaReq := &ollamaapi.ChatRequest{
		Model:    req.Model,
		Messages: toMessages(req.Messages),
	}

	if req.Stream {
		stream := true
		ollamaReq.Stream = &stream
	}

	if req.Options != nil {
		ollamaReq.Options = req.Options
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
		return fn(fromResponse(resp))
	})
}

func toMessages(msgs []loop.Message) []ollamaapi.Message {
	out := make([]ollamaapi.Message, len(msgs))
	for i, m := range msgs {
		out[i] = ollamaapi.Message{
			Role:     m.Role,
			Content:  m.Content,
			Thinking: m.Thinking,
		}
		if len(m.ToolCalls) > 0 {
			out[i].ToolCalls = toToolCalls(m.ToolCalls)
		}
	}
	return out
}

func toToolCalls(calls []loop.ToolCall) []ollamaapi.ToolCall {
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

func fromResponse(resp ollamaapi.ChatResponse) loop.Response {
	r := loop.Response{
		Content:  resp.Message.Content,
		Thinking: resp.Message.Thinking,
		Done:     resp.Done,
	}
	if len(resp.Message.ToolCalls) > 0 {
		r.ToolCalls = make([]loop.ToolCall, len(resp.Message.ToolCalls))
		for i, tc := range resp.Message.ToolCalls {
			r.ToolCalls[i] = loop.ToolCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments.ToMap(),
			}
		}
	}
	return r
}
