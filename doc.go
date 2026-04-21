// Package talk defines the protocol types for LLM communication and
// provides provider adapters in sub-packages. Each sub-package implements
// talk.LLMClient for a specific backend, translating provider-agnostic
// types into native API calls.
//
// Class: primitive
// UseWhen: Always required with axon-loop. For CLI agents, use axon-hand instead which provides axon-talk automatically.
package talk
