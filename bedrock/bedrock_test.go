package bedrock_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/bedrock"
)

// recordingInvoke captures what the adapter sends to Bedrock and replays a
// canned Anthropic Messages response body (or an error).
type recordingInvoke struct {
	gotModelID string
	gotBody    []byte
	reply      []byte
	err        error
}

func (r *recordingInvoke) fn(_ context.Context, modelID string, body []byte) ([]byte, error) {
	r.gotModelID = modelID
	r.gotBody = body
	if r.err != nil {
		return nil, r.err
	}
	return r.reply, nil
}

func collect(t *testing.T, client talk.LLMClient, req *talk.Request) (talk.Response, error) {
	t.Helper()
	var last talk.Response
	err := client.Chat(context.Background(), req, func(r talk.Response) error {
		last = r
		return nil
	})
	return last, err
}

func TestNewClient_TranslatesThroughBedrock(t *testing.T) {
	rec := &recordingInvoke{reply: []byte(`{"content":[{"type":"text","text":"g'day from Sydney"}]}`)}
	client := bedrock.NewClient("au.anthropic.claude-sonnet-4-6", rec.fn)

	resp, err := collect(t, client, &talk.Request{
		Model:    "au.anthropic.claude-sonnet-4-6",
		Stream:   true, // must be stripped for Bedrock InvokeModel
		Messages: []talk.Message{{Role: talk.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "g'day from Sydney" {
		t.Fatalf("Content = %q, want the Bedrock reply", resp.Content)
	}
	if !resp.Done {
		t.Error("buffered response should be Done")
	}

	// The model ID must reach InvokeModel.
	if rec.gotModelID != "au.anthropic.claude-sonnet-4-6" {
		t.Errorf("modelID = %q", rec.gotModelID)
	}

	// The body must drop model/stream and add anthropic_version.
	var sent map[string]any
	if err := json.Unmarshal(rec.gotBody, &sent); err != nil {
		t.Fatalf("unmarshal sent body: %v", err)
	}
	if _, ok := sent["model"]; ok {
		t.Error(`"model" must be stripped from the Bedrock body`)
	}
	if _, ok := sent["stream"]; ok {
		t.Error(`"stream" must be stripped from the Bedrock body`)
	}
	if sent["anthropic_version"] != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version = %v, want bedrock-2023-05-31", sent["anthropic_version"])
	}
	if _, ok := sent["messages"]; !ok {
		t.Error("messages must be preserved in the Bedrock body")
	}
}

func TestNewClient_PropagatesInvokeError(t *testing.T) {
	rec := &recordingInvoke{err: errors.New("throttled")}
	client := bedrock.NewClient("au.anthropic.claude-sonnet-4-6", rec.fn)

	_, err := collect(t, client, &talk.Request{Messages: []talk.Message{{Role: talk.RoleUser, Content: "hi"}}})
	if err == nil {
		t.Fatal("expected the InvokeFunc error to propagate")
	}
}

func TestNewClient_EmptyResponseIsError(t *testing.T) {
	rec := &recordingInvoke{reply: nil}
	client := bedrock.NewClient("au.anthropic.claude-sonnet-4-6", rec.fn)

	_, err := collect(t, client, &talk.Request{Messages: []talk.Message{{Role: talk.RoleUser, Content: "hi"}}})
	if err == nil {
		t.Fatal("expected an error for an empty Bedrock response body")
	}
}

func TestNewClient_NilInvokeIsError(t *testing.T) {
	client := bedrock.NewClient("au.anthropic.claude-sonnet-4-6", nil)
	_, err := collect(t, client, &talk.Request{Messages: []talk.Message{{Role: talk.RoleUser, Content: "hi"}}})
	if err == nil {
		t.Fatal("expected an error when InvokeFunc is nil")
	}
}
