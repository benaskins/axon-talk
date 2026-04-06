package openai

import (
	"context"
	"io"
)

// contextReader wraps an io.Reader so that Read returns the context error
// when the context is cancelled, even if the underlying Read is blocked.
type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func newContextReader(ctx context.Context, r io.Reader) *contextReader {
	return &contextReader{ctx: ctx, r: r}
}

type readResult struct {
	n   int
	err error
}

func (cr *contextReader) Read(p []byte) (int, error) {
	if err := cr.ctx.Err(); err != nil {
		return 0, err
	}

	ch := make(chan readResult, 1)
	go func() {
		n, err := cr.r.Read(p)
		ch <- readResult{n, err}
	}()

	select {
	case <-cr.ctx.Done():
		return 0, cr.ctx.Err()
	case res := <-ch:
		return res.n, res.err
	}
}
