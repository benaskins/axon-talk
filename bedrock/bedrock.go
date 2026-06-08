// Package bedrock provides a talk.LLMClient implementation for Anthropic
// Claude models served through Amazon Bedrock's InvokeModel API.
//
// Bedrock speaks the same Anthropic Messages JSON as api.anthropic.com — the
// only differences on the wire are that the request omits the "model" field
// (the model is addressed by the InvokeModel ModelId instead), omits "stream",
// and carries an "anthropic_version" of "bedrock-2023-05-31". So rather than
// reimplement Claude's request/response (and tool-call) translation, this
// adapter reuses the anthropic adapter wholesale and diverts its HTTP onto a
// caller-supplied InvokeFunc.
//
// axon-talk stays free of an AWS SDK dependency: the caller wires InvokeFunc to
// bedrockruntime.Client (or any signer) in a few lines, keeping control of
// credentials, region, and SigV4 with the consuming service.
//
// Buffered only. Bedrock streaming uses AWS EventStream binary frames, which
// the anthropic adapter's SSE parser cannot read; streaming is a separate path.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/anthropic"
)

// bedrockAnthropicVersion is the value Bedrock requires in the request body's
// "anthropic_version" field for Claude models.
const bedrockAnthropicVersion = "bedrock-2023-05-31"

// sentinelURL is a placeholder base URL handed to the anthropic adapter. No
// real HTTP call is ever made to it — the transport intercepts every request
// and dispatches to InvokeFunc.
const sentinelURL = "https://bedrock.invalid"

// InvokeFunc performs one Bedrock InvokeModel call. Given the model ID (e.g.
// "au.anthropic.claude-sonnet-4-6") and the request body — Anthropic Messages
// JSON with "anthropic_version" already set and "model"/"stream" removed — it
// returns the raw response body, which is itself Anthropic Messages JSON.
//
// The caller implements this against the AWS SDK, for example:
//
//	func(ctx context.Context, modelID string, body []byte) ([]byte, error) {
//		out, err := runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
//			ModelId:     aws.String(modelID),
//			ContentType: aws.String("application/json"),
//			Accept:      aws.String("application/json"),
//			Body:        body,
//		})
//		if err != nil {
//			return nil, err
//		}
//		return out.Body, nil
//	}
type InvokeFunc func(ctx context.Context, modelID string, body []byte) ([]byte, error)

// NewClient returns a talk.LLMClient that serves modelID via Amazon Bedrock,
// reusing the Anthropic Messages translation. invoke must be non-nil.
func NewClient(modelID string, invoke InvokeFunc) talk.LLMClient {
	httpClient := &http.Client{Transport: &transport{modelID: modelID, invoke: invoke}}
	return &client{inner: anthropic.NewClient(sentinelURL, "", anthropic.WithHTTPClient(httpClient))}
}

// client wraps the anthropic adapter to force buffered handling. The anthropic
// adapter picks SSE vs buffered response parsing from req.Stream, but Bedrock
// InvokeModel is buffered-only — its body is a single JSON document, not an SSE
// stream. So we clear Stream before delegating (on a copy, to avoid mutating
// the caller's request), guaranteeing the response is parsed correctly even if
// the caller asked to stream.
type client struct {
	inner talk.LLMClient
}

func (c *client) Chat(ctx context.Context, req *talk.Request, fn func(talk.Response) error) error {
	if req != nil && req.Stream {
		cp := *req
		cp.Stream = false
		req = &cp
	}
	return c.inner.Chat(ctx, req, fn)
}

// transport adapts the anthropic adapter's outbound HTTP request onto a Bedrock
// InvokeModel call: it rewrites the body for Bedrock, invokes, and wraps the
// raw response so the anthropic adapter can parse it unchanged.
type transport struct {
	modelID string
	invoke  InvokeFunc
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.invoke == nil {
		return nil, errors.New("bedrock: nil InvokeFunc")
	}

	raw, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		return nil, err
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	// Bedrock addresses the model by InvokeModel ModelId, not a body field, and
	// requires anthropic_version; stream is a separate (non-Invoke) API.
	delete(body, "model")
	delete(body, "stream")
	body["anthropic_version"] = bedrockAnthropicVersion

	adjusted, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	out, err := t.invoke(req.Context(), t.modelID, adjusted)
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, errors.New("bedrock: empty response body")
	}

	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Body:       io.NopCloser(bytes.NewReader(out)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Request:    req,
	}, nil
}
