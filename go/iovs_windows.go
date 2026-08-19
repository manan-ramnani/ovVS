//go:build windows

package iovs

import (
	"errors"
	"syscall"
	"unsafe"
)

var (
	mod      = syscall.NewLazyDLL("iovs.dll")
	procVer  = mod.NewProc("iovsGetVersion")
	procCre  = mod.NewProc("iovsResourcesCreate")
	procDes  = mod.NewProc("iovsResourcesDestroy")
	procGemm = mod.NewProc("iovsGemm")
)

func windowsCString(p uintptr) string {
	if p == 0 {
		return ""
	}
	b := make([]byte, 0, 32)
	for {
		ch := *(*byte)(unsafe.Pointer(p + uintptr(len(b))))
		if ch == 0 {
			return string(b)
		}
		b = append(b, ch)
	}
}

func Version() string {
	r, _, _ := procVer.Call()
	if r == 0 {
		return ""
	}
	return windowsCString(r)
}

func Gemm(a, b []float32, m, n, k int) ([]float32, error) {
	var res uintptr
	st, _, _ := procCre.Call(uintptr(unsafe.Pointer(&res)))
	if st != 0 {
		return nil, errors.New("resources")
	}
	defer procDes.Call(res)
	out := make([]float32, m*n)
	st, _, _ = procGemm.Call(
		res,
		uintptr(unsafe.Pointer(&a[0])),
		uintptr(unsafe.Pointer(&b[0])),
		uintptr(unsafe.Pointer(&out[0])),
		uintptr(m), uintptr(n), uintptr(k), 1,
	)
	if st != 0 {
		return nil, errors.New("gemm")
	}
	return out, nil
}
