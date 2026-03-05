package ollama_test

import (
	"testing"

	loop "github.com/benaskins/axon-loop"
	"github.com/benaskins/axon-talk/ollama"
	ollamaapi "github.com/ollama/ollama/api"
)

func TestClientImplementsChatClient(t *testing.T) {
	api, _ := ollamaapi.ClientFromEnvironment()
	var _ loop.ChatClient = ollama.NewClient(api)
}
