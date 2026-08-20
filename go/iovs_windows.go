//go:build windows

package iovs

import (
	"errors"
	"syscall"
	"unsafe"
)

var (
	mod         = syscall.NewLazyDLL("iovs.dll")
	procVer     = mod.NewProc("iovsGetVersion")
	procCre     = mod.NewProc("iovsResourcesCreate")
	procDes     = mod.NewProc("iovsResourcesDestroy")
	procGemm    = mod.NewProc("iovsGemm")
	procBfBuild = mod.NewProc("iovsBruteForceBuild")
	procBfSearch = mod.NewProc("iovsBruteForceSearch")
	procBfDes   = mod.NewProc("iovsBruteForceDestroy")
	procKmFit   = mod.NewProc("iovsKMeansFit")
	procKmPred  = mod.NewProc("iovsKMeansPredict")
	procKmDes   = mod.NewProc("iovsKMeansDestroy")
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

func BruteSearch(data []float32, n, dim int, query []float32, k int) ([]int64, []float32, error) {
	var res uintptr
	st, _, _ := procCre.Call(uintptr(unsafe.Pointer(&res)))
	if st != 0 {
		return nil, nil, errors.New("resources")
	}
	defer procDes.Call(res)
	var ix uintptr
	metric := 0
	st, _, _ = procBfBuild.Call(
		res,
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(n), uintptr(dim), uintptr(metric),
		uintptr(unsafe.Pointer(&ix)),
	)
	if st != 0 {
		return nil, nil, errors.New("build")
	}
	defer procBfDes.Call(ix)
	nb := make([]int64, k)
	ds := make([]float32, k)
	st, _, _ = procBfSearch.Call(
		res,
		ix,
		uintptr(unsafe.Pointer(&query[0])),
		uintptr(1), uintptr(k), 0,
		uintptr(unsafe.Pointer(&nb[0])),
		uintptr(unsafe.Pointer(&ds[0])),
	)
	if st != 0 {
		return nil, nil, errors.New("search")
	}
	return nb, ds, nil
}

func KMeansPredict(data []float32, n, dim, nclusters int) ([]int64, []float32, error) {
	var res uintptr
	st, _, _ := procCre.Call(uintptr(unsafe.Pointer(&res)))
	if st != 0 {
		return nil, nil, errors.New("resources")
	}
	defer procDes.Call(res)
	var model uintptr
	st, _, _ = procKmFit.Call(
		res,
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(n), uintptr(dim), uintptr(nclusters), uintptr(8),
		uintptr(unsafe.Pointer(&model)),
	)
	if st != 0 {
		return nil, nil, errors.New("kmeans fit")
	}
	defer procKmDes.Call(model)
	labels := make([]int64, n)
	dist := make([]float32, n)
	st, _, _ = procKmPred.Call(
		res, model,
		uintptr(unsafe.Pointer(&data[0])),
		uintptr(n),
		uintptr(unsafe.Pointer(&labels[0])),
		uintptr(unsafe.Pointer(&dist[0])),
	)
	if st != 0 {
		return nil, nil, errors.New("kmeans predict")
	}
	return labels, dist, nil
}
