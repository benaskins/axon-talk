// Package anthropic provides a talk.LLMClient implementation for the
// Anthropic Messages API. It supports model selection (Opus, Sonnet, Haiku)
// via the standard req.Model field. Works directly against api.anthropic.com
// or through a gateway such as Cloudflare AI Gateway.
package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	talk "github.com/benaskins/axon-talk"
	tool "github.com/benaskins/axon-tool"
)

// Client implements talk.LLMClient for the Anthropic Messages API.
type Client struct {
	baseURL      string
	apiKey       string
	gatewayToken string // optional Cloudflare AI Gateway auth token
	httpClient   *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets a custom http.Client.
func WithHTTPClient(c *http.Client) Option {
	return func(cl *Client) { cl.httpClient = c }
}

// WithGatewayToken sets a Cloudflare AI Gateway authentication token.
// When set, requests include the cf-aig-authorization header.
func WithGatewayToken(token string) Option {
	return func(cl *Client) { cl.gatewayToken = token }
}

// WithPromptCaching returns a RequestOption that enables Anthropic prompt
// caching. When set, cache_control breakpoints are added to the last system
// block and the last tool definition, allowing the API to cache these
// across requests.
func WithPromptCaching() talk.RequestOption {
	return func(r *talk.Request) {
		if r.Options == nil {
			r.Options = map[string]any{}
		}
		r.Options["anthropic_prompt_caching"] = true
	}
}

// NewClient creates a Client that talks to the Anthropic Messages API.
//
// baseURL is the API root, e.g. "https://api.anthropic.com" for direct access,
// or a gateway URL like:
//
//	https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/anthropic
//
// apiKey is the Anthropic API key. Pass "" if a gateway injects the
// key server-side (e.g. Cloudflare AI Gateway with stored credentials).
func NewClient(baseURL, apiKey string, opts ...Option) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Chat sends a request to the Anthropic Messages API and delivers the
// response through fn. When req.Stream is true, the response is streamed
// via SSE with incremental token delivery.
func (c *Client) Chat(ctx context.Context, req *talk.Request, fn func(talk.Response) error) error {
	body := c.buildRequest(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("anthropic: marshal request: %w", err)
	}

	url := c.baseURL + "/v1/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("anthropic: create request: %w", err)
	}
	if c.apiKey != "" {
		httpReq.Header.Set("x-api-key", c.apiKey)
	}
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Content-Type", "application/json")
	if c.gatewayToken != "" {
		httpReq.Header.Set("cf-aig-authorization", "Bearer "+c.gatewayToken)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("anthropic: do request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return &talk.StatusError{StatusCode: httpResp.StatusCode, Body: string(respBody), Provider: "anthropic"}
	}

	if req.Stream {
		return c.handleStream(httpResp.Body, fn)
	}
	return c.handleFull(httpResp.Body, fn)
}

func (c *Client) handleFull(body io.Reader, fn func(talk.Response) error) error {
	respBody, err := io.ReadAll(body)
	if err != nil {
		return fmt.Errorf("anthropic: read response: %w", err)
	}

	var apiResp messagesResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return fmt.Errorf("anthropic: decode response: %w", err)
	}

	return fn(fromResponse(apiResp))
}

func (c *Client) handleStream(body io.Reader, fn func(talk.Response) error) error {
	scanner := bufio.NewScanner(body)

	// Accumulate tool use blocks being built across events.
	type pendingToolUse struct {
		id   string
		name string
		args strings.Builder
	}
	var pending []*pendingToolUse

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			eventType := strings.TrimPrefix(line, "event: ")

			// Read the data line.
			if !scanner.Scan() {
				break
			}
			dataLine := scanner.Text()
			if !strings.HasPrefix(dataLine, "data: ") {
				continue
			}
			data := strings.TrimPrefix(dataLine, "data: ")

			switch eventType {
			case "content_block_start":
				var ev streamContentBlockStart
				if err := json.Unmarshal([]byte(data), &ev); err != nil {
					continue
				}
				if ev.ContentBlock.Type == "tool_use" {
					pending = append(pending, &pendingToolUse{
						id:   ev.ContentBlock.ID,
						name: ev.ContentBlock.Name,
					})
				}

			case "content_block_delta":
				var ev streamContentBlockDelta
				if err := json.Unmarshal([]byte(data), &ev); err != nil {
					continue
				}
				switch ev.Delta.Type {
				case "text_delta":
					if err := fn(talk.Response{Content: ev.Delta.Text}); err != nil {
						return err
					}
				case "thinking_delta":
					if err := fn(talk.Response{Thinking: ev.Delta.Thinking}); err != nil {
						return err
					}
				case "input_json_delta":
					if len(pending) > 0 {
						pending[len(pending)-1].args.WriteString(ev.Delta.PartialJSON)
					}
				}

			case "message_delta":
				// Final event — assemble tool calls and signal done.
				var toolCalls []talk.ToolCall
				for _, p := range pending {
					var args map[string]any
					if p.args.Len() > 0 {
						json.Unmarshal([]byte(p.args.String()), &args)
					}
					toolCalls = append(toolCalls, talk.ToolCall{
						Name:      p.name,
						Arguments: args,
					})
				}
				if err := fn(talk.Response{Done: true, ToolCalls: toolCalls}); err != nil {
					return err
				}

			case "message_stop":
				// Stream complete.

			case "error":
				return fmt.Errorf("anthropic: stream error: %s", data)
			}
		}
	}

	return scanner.Err()
}

func (c *Client) buildRequest(req *talk.Request) messagesRequest {
	msgs, system := toMessages(req.Messages)

	// Anthropic API requires at least one message in the messages array.
	// When only a system prompt is provided (e.g. to get the LLM to
	// initiate conversation), add a minimal user message.
	if len(msgs) == 0 {
		msgs = []message{{
			Role:    "user",
			Content: []contentBlock{{Type: "text", Text: "Begin."}},
		}}
	}

	mr := messagesRequest{
		Model:    req.Model,
		Messages: msgs,
		System:   system,
	}

	// max_tokens is required by the Anthropic API.
	mr.MaxTokens = 4096
	if v, ok := req.Options["max_tokens"]; ok {
		switch mt := v.(type) {
		case int:
			mr.MaxTokens = mt
		case float64:
			mr.MaxTokens = int(mt)
		}
	}

	if v, ok := req.Options["temperature"]; ok {
		if t, ok := v.(float64); ok {
			mr.Temperature = &t
		}
	}

	if req.Think != nil && *req.Think {
		budget := 10000
		if v, ok := req.Options["thinking_budget"]; ok {
			switch b := v.(type) {
			case int:
				budget = b
			case float64:
				budget = int(b)
			}
		}
		mr.Thinking = &thinkingParam{
			Type:         "enabled",
			BudgetTokens: budget,
		}
		// Anthropic requires temperature=1 (or unset) when thinking is enabled.
		mr.Temperature = nil
	}

	if len(req.Tools) > 0 {
		mr.Tools = toTools(req.Tools)
	}

	if req.Stream {
		mr.Stream = true
	}

	// Apply structured output via constrained tool use.
	if schema, ok := req.Options["structured_output"].(map[string]any); ok {
		mr.Tools = append(mr.Tools, toolDef{
			Name:        "structured_response",
			Description: "Respond with structured data matching the schema.",
			InputSchema: inputSchema{
				Type:       "object",
				Properties: toPropertyDefsFromSchema(schema),
				Required:   requiredFromSchema(schema),
			},
		})
		mr.ToolChoice = &toolChoice{
			Type: "tool",
			Name: "structured_response",
		}
	}

	// Apply prompt caching breakpoints.
	if _, ok := req.Options["anthropic_prompt_caching"]; ok {
		ephemeral := &cacheControl{Type: "ephemeral"}
		if len(mr.System) > 0 {
			mr.System[len(mr.System)-1].CacheControl = ephemeral
		}
		if len(mr.Tools) > 0 {
			mr.Tools[len(mr.Tools)-1].CacheControl = ephemeral
		}
	}

	return mr
}

// toMessages converts talk.Messages to Anthropic message format,
// extracting system messages into the separate system parameter.
func toMessages(msgs []talk.Message) ([]message, []systemBlock) {
	var out []message
	var system []systemBlock

	// Track tool call IDs: when an assistant message has tool calls,
	// generate IDs that subsequent tool result messages can reference.
	var pendingToolIDs []string

	for _, m := range msgs {
		if m.Role == "system" {
			system = append(system, systemBlock{
				Type: "text",
				Text: m.Content,
			})
			continue
		}

		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			// Assistant message with tool use — convert to content blocks.
			var blocks []contentBlock
			if m.Content != "" {
				blocks = append(blocks, contentBlock{
					Type: "text",
					Text: m.Content,
				})
			}
			pendingToolIDs = nil
			for i, tc := range m.ToolCalls {
				id := fmt.Sprintf("toolu_%d", i)
				pendingToolIDs = append(pendingToolIDs, id)
				input := tc.Arguments
				if input == nil {
					input = map[string]any{}
				}
				blocks = append(blocks, contentBlock{
					Type:  "tool_use",
					ID:    id,
					Name:  tc.Name,
					Input: input,
				})
			}
			out = append(out, message{
				Role:    "assistant",
				Content: blocks,
			})
			continue
		}

		if m.Role == "tool" {
			// Tool result — becomes a user message with tool_result content block.
			// If there are pending tool IDs, consume the first one.
			toolID := ""
			if len(pendingToolIDs) > 0 {
				toolID = pendingToolIDs[0]
				pendingToolIDs = pendingToolIDs[1:]
			}
			block := contentBlock{
				Type:      "tool_result",
				ToolUseID: toolID,
				Content:   m.Content,
			}

			// Merge consecutive tool results into a single user message.
			if len(out) > 0 && out[len(out)-1].Role == "user" && len(out[len(out)-1].Content) > 0 && out[len(out)-1].Content[0].Type == "tool_result" {
				out[len(out)-1].Content = append(out[len(out)-1].Content, block)
			} else {
				out = append(out, message{
					Role:    "user",
					Content: []contentBlock{block},
				})
			}
			continue
		}

		// Regular user or assistant message.
		var blocks []contentBlock
		if m.Thinking != "" {
			blocks = append(blocks, contentBlock{
				Type:     "thinking",
				Thinking: m.Thinking,
			})
		}
		blocks = append(blocks, contentBlock{
			Type: "text",
			Text: m.Content,
		})
		out = append(out, message{
			Role:    string(m.Role),
			Content: blocks,
		})
	}

	return out, system
}

func toTools(defs []tool.ToolDef) []toolDef {
	out := make([]toolDef, len(defs))
	for i, d := range defs {
		out[i] = toolDef{
			Name:        d.Name,
			Description: d.Description,
			InputSchema: toInputSchema(d.Parameters),
		}
	}
	return out
}

func toInputSchema(ps tool.ParameterSchema) inputSchema {
	return inputSchema{
		Type:       ps.Type,
		Required:   ps.Required,
		Properties: toPropertyDefs(ps.Properties),
	}
}

func toPropertyDefs(props map[string]tool.PropertySchema) map[string]propertyDef {
	if len(props) == 0 {
		return nil
	}
	out := make(map[string]propertyDef, len(props))
	for name, prop := range props {
		pd := propertyDef{
			Type:        prop.Type,
			Description: prop.Description,
			Enum:        prop.Enum,
		}
		if prop.Items != nil {
			items := toPropertyDef(*prop.Items)
			pd.Items = &items
		}
		if len(prop.Properties) > 0 {
			pd.Properties = toPropertyDefs(prop.Properties)
			pd.Required = prop.Required
		}
		out[name] = pd
	}
	return out
}

func toPropertyDef(prop tool.PropertySchema) propertyDef {
	pd := propertyDef{
		Type:        prop.Type,
		Description: prop.Description,
		Enum:        prop.Enum,
	}
	if prop.Items != nil {
		items := toPropertyDef(*prop.Items)
		pd.Items = &items
	}
	if len(prop.Properties) > 0 {
		pd.Properties = toPropertyDefs(prop.Properties)
		pd.Required = prop.Required
	}
	return pd
}

// toPropertyDefsFromSchema converts a raw JSON schema's "properties" map to
// anthropic propertyDef format.
func toPropertyDefsFromSchema(schema map[string]any) map[string]propertyDef {
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return nil
	}
	out := make(map[string]propertyDef, len(props))
	for name, v := range props {
		p, ok := v.(map[string]any)
		if !ok {
			continue
		}
		pd := propertyDef{}
		if t, ok := p["type"].(string); ok {
			pd.Type = t
		}
		if d, ok := p["description"].(string); ok {
			pd.Description = d
		}
		out[name] = pd
	}
	return out
}

// requiredFromSchema extracts the "required" array from a JSON schema.
func requiredFromSchema(schema map[string]any) []string {
	req, ok := schema["required"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(req))
	for _, v := range req {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func fromResponse(resp messagesResponse) talk.Response {
	var content strings.Builder
	var thinking strings.Builder
	var toolCalls []talk.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "thinking":
			thinking.WriteString(block.Thinking)
		case "tool_use":
			toolCalls = append(toolCalls, talk.ToolCall{
				Name:      block.Name,
				Arguments: block.Input,
			})
		}
	}

	r := talk.Response{
		Content:   content.String(),
		Thinking:  thinking.String(),
		ToolCalls: toolCalls,
		Done:      true,
	}
	if resp.Usage != nil {
		r.Usage = &talk.Usage{
			InputTokens:             resp.Usage.InputTokens,
			OutputTokens:            resp.Usage.OutputTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:    resp.Usage.CacheReadInputTokens,
		}
	}
	return r
}

// Wire types for the Anthropic Messages API.

type messagesRequest struct {
	Model       string         `json:"model"`
	Messages    []message      `json:"messages"`
	System      []systemBlock  `json:"system,omitempty"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature *float64       `json:"temperature,omitempty"`
	Thinking    *thinkingParam `json:"thinking,omitempty"`
	Tools       []toolDef      `json:"tools,omitempty"`
	ToolChoice  *toolChoice    `json:"tool_choice,omitempty"`
	Stream      bool           `json:"stream,omitempty"`
}

type thinkingParam struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

type toolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type systemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type message struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   string         `json:"content,omitempty"`
}

// MarshalJSON handles the contentBlock's dual use of the "content" field.
// For tool_result blocks, "content" is the tool output string.
// For text blocks, "text" is the content. We need custom marshaling
// because the struct has both Content and Text fields.
func (cb contentBlock) MarshalJSON() ([]byte, error) {
	switch cb.Type {
	case "thinking":
		return json.Marshal(struct {
			Type     string `json:"type"`
			Thinking string `json:"thinking"`
		}{
			Type:     cb.Type,
			Thinking: cb.Thinking,
		})
	case "tool_result":
		return json.Marshal(struct {
			Type      string `json:"type"`
			ToolUseID string `json:"tool_use_id"`
			Content   string `json:"content"`
		}{
			Type:      cb.Type,
			ToolUseID: cb.ToolUseID,
			Content:   cb.Content,
		})
	case "tool_use":
		return json.Marshal(struct {
			Type  string         `json:"type"`
			ID    string         `json:"id"`
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		}{
			Type:  cb.Type,
			ID:    cb.ID,
			Name:  cb.Name,
			Input: cb.Input,
		})
	default:
		return json.Marshal(struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}{
			Type: cb.Type,
			Text: cb.Text,
		})
	}
}

type messagesResponse struct {
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      *apiUsage      `json:"usage,omitempty"`
}

type apiUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

type toolDef struct {
	Name         string        `json:"name"`
	Description  string        `json:"description,omitempty"`
	InputSchema  inputSchema   `json:"input_schema"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type inputSchema struct {
	Type       string                 `json:"type"`
	Required   []string               `json:"required,omitempty"`
	Properties map[string]propertyDef `json:"properties,omitempty"`
}

type propertyDef struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description,omitempty"`
	Enum        []any                  `json:"enum,omitempty"`
	Items       *propertyDef           `json:"items,omitempty"`
	Properties  map[string]propertyDef `json:"properties,omitempty"`
	Required    []string               `json:"required,omitempty"`
}

// Streaming event types.

type streamContentBlockStart struct {
	ContentBlock contentBlock `json:"content_block"`
}

type streamContentBlockDelta struct {
	Delta streamDelta `json:"delta"`
}

type streamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}
