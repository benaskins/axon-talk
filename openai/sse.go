package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// sseEvent is a parsed SSE event from a streaming chat completion response.
type sseEvent struct {
	Delta streamDelta
	Done  bool
	Usage *completionUsage
}

// streamDelta is the delta object from a streaming chunk.
type streamDelta struct {
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content"`
	ToolCalls        []streamToolCall `json:"tool_calls"`
}

// streamToolCall is a tool call delta in a streaming chunk.
type streamToolCall struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function streamToolCallFunction `json:"function"`
}

// streamToolCallFunction holds the incremental function name and arguments.
type streamToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// streamChunk is the wire format for a single SSE data line.
type streamChunk struct {
	Choices []streamChoice   `json:"choices"`
	Usage   *completionUsage `json:"usage,omitempty"`
}

// streamChoice wraps the delta in the OpenAI-compatible format.
type streamChoice struct {
	Delta streamDelta `json:"delta"`
}

// parseSSE reads SSE data lines from r and calls fn for each parsed event.
// It handles the [DONE] sentinel and skips non-data lines.
func parseSSE(r io.Reader, fn func(sseEvent) error) error {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			return fn(sseEvent{Done: true})
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("openai: parse SSE chunk: %w", err)
		}

		if len(chunk.Choices) == 0 {
			// Usage-only chunk (final chunk with stream_options.include_usage)
			if chunk.Usage != nil {
				if err := fn(sseEvent{Usage: chunk.Usage}); err != nil {
					return err
				}
			}
			continue
		}

		ev := sseEvent{Delta: chunk.Choices[0].Delta, Usage: chunk.Usage}
		if err := fn(ev); err != nil {
			return err
		}
	}

	return scanner.Err()
}
