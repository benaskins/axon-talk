package openai

import (
	"encoding/json"
	"strings"
	"testing"

	talk "github.com/benaskins/axon-talk"
)

func TestToMessages_UserImageBecomesContentParts(t *testing.T) {
	msgs := []talk.Message{{
		Role:    talk.RoleUser,
		Content: "what is this asset?",
		Images:  []talk.ImageContent{{MediaType: "image/jpeg", Data: "QUJD"}},
	}}
	out := toMessages(msgs)
	parts, ok := out[0].Content.([]contentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("want 2 content parts (text, image_url), got %#v", out[0].Content)
	}
	if parts[0].Type != "text" || parts[0].Text != "what is this asset?" {
		t.Fatalf("first part should be the text, got %#v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("second part should be an image_url, got %#v", parts[1])
	}
	if parts[1].ImageURL.URL != "data:image/jpeg;base64,QUJD" {
		t.Fatalf("data URI wrong: %s", parts[1].ImageURL.URL)
	}

	raw, err := json.Marshal(out[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"image_url":{"url":"data:image/jpeg;base64,QUJD"}`) {
		t.Fatalf("image not in wire JSON: %s", raw)
	}
}

func TestToMessages_ImageOnlyMessageHasNoTextPart(t *testing.T) {
	msgs := []talk.Message{{
		Role:   talk.RoleUser,
		Images: []talk.ImageContent{{MediaType: "image/png", Data: "QUJD"}},
	}}
	out := toMessages(msgs)
	parts, ok := out[0].Content.([]contentPart)
	if !ok || len(parts) != 1 || parts[0].Type != "image_url" {
		t.Fatalf("want a single image_url part, got %#v", out[0].Content)
	}
}

func TestToMessages_AssistantImagesIgnored(t *testing.T) {
	msgs := []talk.Message{{
		Role:    talk.RoleAssistant,
		Content: "here is the result",
		Images:  []talk.ImageContent{{MediaType: "image/jpeg", Data: "QUJD"}},
	}}
	out := toMessages(msgs)
	if _, isParts := out[0].Content.([]contentPart); isParts {
		t.Fatalf("assistant images should be ignored, got parts: %#v", out[0].Content)
	}
	if out[0].Content != "here is the result" {
		t.Fatalf("assistant content should stay a plain string, got %#v", out[0].Content)
	}
}

func TestToMessages_TextMessageStaysString(t *testing.T) {
	msgs := []talk.Message{{Role: talk.RoleUser, Content: "hello"}}
	out := toMessages(msgs)
	if out[0].Content != "hello" {
		t.Fatalf("text-only message should stay a plain string, got %#v", out[0].Content)
	}
}
