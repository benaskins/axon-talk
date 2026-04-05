package openai_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/openai"
	tool "github.com/benaskins/axon-tool"
)

// TestLive_OpenRouterStreamingToolCall reproduces the code-lead stall:
// streaming request to Sonnet via OpenRouter with tool definitions.
// Run with: OPENROUTER_API_KEY=sk-or-... go test -run TestLive_OpenRouterStreamingToolCall -v -timeout 60s
func TestLive_OpenRouterStreamingToolCall(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY required")
	}

	model := os.Getenv("TEST_MODEL")
	if model == "" {
		model = "anthropic/claude-sonnet-4"
	}

	client := openai.NewClient("https://openrouter.ai/api", apiKey)

	tools := []tool.ToolDef{
		{
			Name:        "greet",
			Description: "Greet someone by name.",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"name"},
				Properties: map[string]tool.PropertySchema{
					"name": {Type: "string", Description: "The person's name."},
				},
			},
		},
	}

	req := &talk.Request{
		Model: model,
		Messages: []talk.Message{
			{Role: "system", Content: "You are a helpful assistant. When asked to greet someone, use the greet tool."},
			{Role: "user", Content: "Please greet Alice."},
		},
		Tools:  tools,
		Stream: true,
		Options: map[string]any{
			"temperature": float64(0),
		},
	}

	start := time.Now()
	var gotContent string
	var gotToolCalls []talk.ToolCall
	var gotDone bool

	t.Logf("sending streaming request to %s via OpenRouter...", model)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := client.Chat(ctx, req, func(resp talk.Response) error {
		elapsed := time.Since(start).Round(time.Millisecond)
		if resp.Content != "" {
			gotContent += resp.Content
			t.Logf("[%s] content: %q", elapsed, resp.Content)
		}
		if resp.Thinking != "" {
			t.Logf("[%s] thinking: %q", elapsed, resp.Thinking)
		}
		if len(resp.ToolCalls) > 0 {
			gotToolCalls = resp.ToolCalls
			t.Logf("[%s] tool calls: %v", elapsed, resp.ToolCalls)
		}
		if resp.Done {
			gotDone = true
			t.Logf("[%s] done", elapsed)
		}
		return nil
	})

	elapsed := time.Since(start)
	t.Logf("total elapsed: %s", elapsed)

	if err != nil {
		t.Fatalf("Chat error after %s: %v", elapsed, err)
	}

	if !gotDone {
		t.Error("never received done signal")
	}

	if len(gotToolCalls) == 0 && gotContent == "" {
		t.Error("got neither tool calls nor content")
	}

	if len(gotToolCalls) > 0 {
		tc := gotToolCalls[0]
		t.Logf("tool call: %s(%v)", tc.Name, tc.Arguments)
		if tc.Name != "greet" {
			t.Errorf("expected greet tool call, got %q", tc.Name)
		}
		name, _ := tc.Arguments["name"].(string)
		if name != "Alice" {
			t.Errorf("expected name=Alice, got %q", name)
		}
	}

	fmt.Fprintf(os.Stderr, "\nRESULT: streaming=%v, tool_calls=%d, content_len=%d, elapsed=%s\n",
		req.Stream, len(gotToolCalls), len(gotContent), elapsed)
}

// TestLive_OpenRouterRealisticOrchestration simulates code-lead's actual payload:
// large system prompt, 4 tool definitions, multi-step plan in user message.
// This is the scenario that was stalling in production.
func TestLive_OpenRouterRealisticOrchestration(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY required")
	}

	model := os.Getenv("TEST_MODEL")
	if model == "" {
		model = "anthropic/claude-sonnet-4"
	}

	client := openai.NewClient("https://openrouter.ai/api", apiKey)

	// code-lead's actual system prompt
	sysPrompt := `You are code-lead, a build orchestrator on the factory floor.

Your job is to execute a build plan step by step. For each step:

1. Call run_code_hand with the step title, description, and project directory.
   This invokes the coding agent to implement the step.
2. Call verify_build to check that the project compiles and tests pass.
   If verification fails, call run_code_hand again with feedback describing
   what went wrong. You get up to 3 attempts per step.
3. If verification passes, call read_diff to review the changes.
4. Call commit_step with the project directory and the commit message from
   the plan step.
5. Move to the next step.

Be methodical. Execute steps in order. Report what you're doing at each stage.
If a step fails after 3 attempts, report the failure and stop.

Do not skip steps. Do not modify the plan. Execute what is given to you.`

	// code-lead's actual 4 tools
	tools := []tool.ToolDef{
		{
			Name:        "run_code_hand",
			Description: "Invoke code-hand to implement a plan step. Provide step title, description, project directory, and optional feedback from a previous attempt.",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"step_title":       {Type: "string", Description: "The step title"},
					"step_description": {Type: "string", Description: "Full step description with implementation details"},
					"project_dir":      {Type: "string", Description: "Absolute path to the project directory"},
					"feedback":         {Type: "string", Description: "Optional feedback from a previous failed attempt"},
				},
				Required: []string{"step_title", "step_description", "project_dir"},
			},
		},
		{
			Name:        "verify_build",
			Description: "Run go build and go test in the project directory. Returns pass/fail with output.",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"project_dir": {Type: "string", Description: "Absolute path to the project directory"},
				},
				Required: []string{"project_dir"},
			},
		},
		{
			Name:        "read_diff",
			Description: "Read the git diff of uncommitted changes in the project directory.",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"project_dir": {Type: "string", Description: "Absolute path to the project directory"},
				},
				Required: []string{"project_dir"},
			},
		},
		{
			Name:        "commit_step",
			Description: "Stage all changes and commit with the given message. Returns the commit hash.",
			Parameters: tool.ParameterSchema{
				Type: "object",
				Properties: map[string]tool.PropertySchema{
					"project_dir": {Type: "string", Description: "Absolute path to the project directory"},
					"message":     {Type: "string", Description: "Commit message"},
				},
				Required: []string{"project_dir", "message"},
			},
		},
	}

	// Realistic multi-step plan (abbreviated but representative)
	userMsg := `Project directory: /Users/dev/bookmarks-app

Execute the following plan steps in order:

## Step 3: User authentication setup

Integrate axon-auth with email/password authentication. Create login and registration endpoints. Add session middleware for protected routes. Create user repository interface and implementation using axon-base. Test user registration, login, and session persistence.

Commit message: ` + "`feat: implement user authentication with axon-auth`" + `

## Step 4: Bookmark data layer

Create bookmark repository interface with methods for Create, GetByUserID, GetByID, Update, Delete, and Search. Implement repository using axon-base with proper SQL queries including full-text search using PostgreSQL's to_tsvector. Create bookmark domain model with validation. Test all repository methods.

Commit message: ` + "`feat: implement bookmark repository with full-text search`"

	think := false
	req := &talk.Request{
		Model: model,
		Messages: []talk.Message{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: userMsg},
		},
		Tools:  tools,
		Stream: true,
		Think:  &think,
		Options: map[string]any{
			"temperature": float64(0),
		},
	}

	start := time.Now()
	var gotContent string
	var gotToolCalls []talk.ToolCall
	var gotDone bool
	firstTokenAt := time.Duration(0)

	t.Logf("sending realistic orchestration request to %s (streaming=%v)...", model, req.Stream)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	err := client.Chat(ctx, req, func(resp talk.Response) error {
		elapsed := time.Since(start).Round(time.Millisecond)
		if firstTokenAt == 0 && (resp.Content != "" || len(resp.ToolCalls) > 0) {
			firstTokenAt = elapsed
		}
		if resp.Content != "" {
			gotContent += resp.Content
			t.Logf("[%s] content: %q", elapsed, resp.Content)
		}
		if len(resp.ToolCalls) > 0 {
			gotToolCalls = resp.ToolCalls
			for _, tc := range resp.ToolCalls {
				t.Logf("[%s] tool call: %s(%v)", elapsed, tc.Name, tc.Arguments)
			}
		}
		if resp.Done {
			gotDone = true
			t.Logf("[%s] done", elapsed)
		}
		return nil
	})

	elapsed := time.Since(start)
	t.Logf("total elapsed: %s, first token: %s", elapsed, firstTokenAt)

	if err != nil {
		t.Fatalf("Chat error after %s: %v", elapsed, err)
	}

	if !gotDone {
		t.Error("never received done signal")
	}

	if len(gotToolCalls) == 0 {
		t.Errorf("expected tool calls (run_code_hand), got none. content=%q", gotContent)
	} else {
		tc := gotToolCalls[0]
		if tc.Name != "run_code_hand" {
			t.Errorf("expected run_code_hand, got %q", tc.Name)
		}
	}

	fmt.Fprintf(os.Stderr, "\nRESULT: model=%s streaming=%v tool_calls=%d first_token=%s total=%s\n",
		model, req.Stream, len(gotToolCalls), firstTokenAt, elapsed)
}

// TestLive_OpenRouterNonStreamingToolCall is the same test without streaming,
// to compare behavior.
func TestLive_OpenRouterNonStreamingToolCall(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY required")
	}

	model := os.Getenv("TEST_MODEL")
	if model == "" {
		model = "anthropic/claude-sonnet-4"
	}

	client := openai.NewClient("https://openrouter.ai/api", apiKey)

	tools := []tool.ToolDef{
		{
			Name:        "greet",
			Description: "Greet someone by name.",
			Parameters: tool.ParameterSchema{
				Type:     "object",
				Required: []string{"name"},
				Properties: map[string]tool.PropertySchema{
					"name": {Type: "string", Description: "The person's name."},
				},
			},
		},
	}

	req := &talk.Request{
		Model: model,
		Messages: []talk.Message{
			{Role: "system", Content: "You are a helpful assistant. When asked to greet someone, use the greet tool."},
			{Role: "user", Content: "Please greet Alice."},
		},
		Tools:  tools,
		Stream: false,
		Options: map[string]any{
			"temperature": float64(0),
		},
	}

	start := time.Now()
	var gotToolCalls []talk.ToolCall

	t.Logf("sending non-streaming request to %s via OpenRouter...", model)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := client.Chat(ctx, req, func(resp talk.Response) error {
		elapsed := time.Since(start).Round(time.Millisecond)
		t.Logf("[%s] response: content=%q tool_calls=%d done=%v", elapsed, resp.Content, len(resp.ToolCalls), resp.Done)
		if len(resp.ToolCalls) > 0 {
			gotToolCalls = resp.ToolCalls
		}
		return nil
	})

	elapsed := time.Since(start)
	t.Logf("total elapsed: %s", elapsed)

	if err != nil {
		t.Fatalf("Chat error after %s: %v", elapsed, err)
	}

	if len(gotToolCalls) == 0 {
		t.Error("expected tool call, got none")
	}

	fmt.Fprintf(os.Stderr, "\nRESULT: streaming=%v, tool_calls=%d, elapsed=%s\n",
		req.Stream, len(gotToolCalls), elapsed)
}
