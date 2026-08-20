//go:build !windows

package iovs

// Non-Windows builds should link libiovs via cgo. Windows uses syscall in iovs_windows.go.

func Version() string { return "" }

func Gemm(a, b []float32, m, n, k int) ([]float32, error) {
	return nil, errUnimplemented
}

func BruteSearch(data []float32, n, dim int, query []float32, k int) ([]int64, []float32, error) {
	return nil, nil, errUnimplemented
}

func KMeansPredict(data []float32, n, dim, nclusters int) ([]int64, []float32, error) {
	return nil, nil, errUnimplemented
}

var errUnimplemented = errString("iovs: cgo path not built")

type errString string

func (e errString) Error() string { return string(e) }
