package ollama_test

import (
	"testing"

	loop "github.com/benaskins/axon-loop"
	"github.com/benaskins/axon-talk/ollama"
	ollamaapi "github.com/ollama/ollama/api"
)

func TestClientImplementsLLMClient(t *testing.T) {
	api, _ := ollamaapi.ClientFromEnvironment()
	var _ loop.LLMClient = ollama.NewClient(api)
}
