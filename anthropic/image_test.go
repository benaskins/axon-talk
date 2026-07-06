package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	talk "github.com/benaskins/axon-talk"
)

func TestContentBlock_ImageMarshalJSON(t *testing.T) {
	cb := contentBlock{Type: "image", Source: &imageSource{Type: "base64", MediaType: "image/jpeg", Data: "QUJD"}}
	got, err := json.Marshal(cb)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"QUJD"}}`
	if string(got) != want {
		t.Fatalf("image block JSON:\n got %s\nwant %s", got, want)
	}
}

func TestToMessages_UserImagePrecedesText(t *testing.T) {
	msgs := []talk.Message{{
		Role:    talk.RoleUser,
		Content: "what is this asset?",
		Images:  []talk.ImageContent{{MediaType: "image/jpeg", Data: "QUJD"}},
	}}
	out, _ := toMessages(msgs)
	if len(out) != 1 || len(out[0].Content) != 2 {
		t.Fatalf("want 1 message with 2 blocks (image, text), got %+v", out)
	}
	if out[0].Content[0].Type != "image" || out[0].Content[1].Type != "text" {
		t.Fatalf("blocks out of order: %s then %s", out[0].Content[0].Type, out[0].Content[1].Type)
	}
	raw, err := json.Marshal(out[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"media_type":"image/jpeg"`) || !strings.Contains(string(raw), `"data":"QUJD"`) {
		t.Fatalf("image not in wire JSON: %s", raw)
	}
}

func TestToMessages_ImageOnlyMessageHasNoEmptyTextBlock(t *testing.T) {
	msgs := []talk.Message{{
		Role:   talk.RoleUser,
		Images: []talk.ImageContent{{MediaType: "image/png", Data: "QUJD"}},
	}}
	out, _ := toMessages(msgs)
	if len(out[0].Content) != 1 || out[0].Content[0].Type != "image" {
		t.Fatalf("want a single image block, got %+v", out[0].Content)
	}
}

func TestToMessages_TextMessageUnchanged(t *testing.T) {
	msgs := []talk.Message{{Role: talk.RoleUser, Content: "hello"}}
	out, _ := toMessages(msgs)
	if len(out[0].Content) != 1 || out[0].Content[0].Type != "text" || out[0].Content[0].Text != "hello" {
		t.Fatalf("text-only message changed: %+v", out[0].Content)
	}
}
