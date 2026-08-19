//go:build !windows

package iovs

// Non-Windows builds should link libiovs via cgo. Windows uses syscall in iovs_windows.go.

func Version() string { return "" }

func Gemm(a, b []float32, m, n, k int) ([]float32, error) {
	return nil, errUnimplemented
}

var errUnimplemented = errString("iovs: cgo path not built")

type errString string

func (e errString) Error() string { return string(e) }
