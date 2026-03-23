package ollama_test

import (
	"testing"

	talk "github.com/benaskins/axon-talk"
	"github.com/benaskins/axon-talk/ollama"
	ollamaapi "github.com/ollama/ollama/api"
)

func TestClientImplementsLLMClient(t *testing.T) {
	api, _ := ollamaapi.ClientFromEnvironment()
	var _ talk.LLMClient = ollama.NewClient(api)
}
