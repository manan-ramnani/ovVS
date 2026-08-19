package iovs

// Thin cgo wrapper around libiovs. Requires the shared library on the linker path.

/*
#cgo CFLAGS: -I../include
#cgo LDFLAGS: -liovs
#include "iovs/iovs.h"
*/
import "C"

func Version() string {
	return C.GoString(C.iovsGetVersion())
}
