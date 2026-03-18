package cloudflare

import (
	"strings"
	"testing"
)

func TestParseSSE_ContentDeltas(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]

`
	r := strings.NewReader(input)
	var tokens []string
	err := parseSSE(r, func(ev sseEvent) error {
		if ev.Done {
			return nil
		}
		tokens = append(tokens, ev.Delta.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}
	if got := strings.Join(tokens, ""); got != "Hello world" {
		t.Errorf("content = %q, want %q", got, "Hello world")
	}
}

func TestParseSSE_ThinkingDeltas(t *testing.T) {
	input := `data: {"choices":[{"delta":{"reasoning_content":"Let me think"}}]}

data: {"choices":[{"delta":{"content":"Answer"}}]}

data: [DONE]

`
	r := strings.NewReader(input)
	var thinking, content string
	err := parseSSE(r, func(ev sseEvent) error {
		if ev.Done {
			return nil
		}
		thinking += ev.Delta.ReasoningContent
		content += ev.Delta.Content
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}
	if thinking != "Let me think" {
		t.Errorf("thinking = %q, want %q", thinking, "Let me think")
	}
	if content != "Answer" {
		t.Errorf("content = %q, want %q", content, "Answer")
	}
}

func TestParseSSE_ToolCallDeltas(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Sydney\"}"}}]}}]}

data: [DONE]

`
	r := strings.NewReader(input)
	var events []sseEvent
	err := parseSSE(r, func(ev sseEvent) error {
		events = append(events, ev)
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}

	// Should have 4 delta events + 1 done
	if len(events) != 5 {
		t.Fatalf("got %d events, want 5", len(events))
	}

	// First delta has name
	if events[0].Delta.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", events[0].Delta.ToolCalls[0].Function.Name)
	}

	// Last event is done
	if !events[4].Done {
		t.Error("last event should be done")
	}
}

func TestParseSSE_MultipleToolCalls(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"search","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"go\"}"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"q\":\"rust\"}"}}]}}]}

data: [DONE]

`
	r := strings.NewReader(input)
	var events []sseEvent
	err := parseSSE(r, func(ev sseEvent) error {
		events = append(events, ev)
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}

	// Check index values are preserved
	if events[2].Delta.ToolCalls[0].Index != 1 {
		t.Errorf("second tool call index = %d, want 1", events[2].Delta.ToolCalls[0].Index)
	}
}

func TestParseSSE_EmptyLines(t *testing.T) {
	// SSE spec allows multiple blank lines between events
	input := `data: {"choices":[{"delta":{"content":"ok"}}]}


data: [DONE]

`
	r := strings.NewReader(input)
	var count int
	err := parseSSE(r, func(ev sseEvent) error {
		count++
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}
	if count != 2 {
		t.Errorf("got %d events, want 2", count)
	}
}

func TestParseSSE_DoneSignal(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

`
	r := strings.NewReader(input)
	var gotDone bool
	err := parseSSE(r, func(ev sseEvent) error {
		if ev.Done {
			gotDone = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}
	if !gotDone {
		t.Error("expected done event")
	}
}
