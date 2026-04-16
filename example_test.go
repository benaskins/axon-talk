package talk_test

import (
	talk "github.com/benaskins/axon-talk"
)

func ExampleNewRequest() {
	req := talk.NewRequest("claude-sonnet-4-20250514", []talk.Message{
		{Role: talk.RoleSystem, Content: "You are a helpful assistant."},
		{Role: talk.RoleUser, Content: "What is the capital of France?"},
	})
	_ = req
}

func ExampleMessage() {
	user := talk.Message{
		Role:    talk.RoleUser,
		Content: "Explain goroutines in one sentence.",
	}

	assistant := talk.Message{
		Role:    talk.RoleAssistant,
		Content: "Goroutines are lightweight, cooperatively scheduled threads managed by the Go runtime.",
	}

	_ = user
	_ = assistant
}
