"""ioVS Python bindings — ctypes over the stable C ABI."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int32, c_int64, c_void_p
from pathlib import Path


def _load():
    env = os.environ.get("IOVS_LIBRARY")
    names = []
    if env:
        names.append(Path(env))
    here = Path(__file__).resolve().parent
    roots = [
        here,
        here.parent,
        here.parent.parent / "build" / "bin",
        here.parent.parent / "build" / "Release",
        here.parent.parent / "build" / "Debug",
    ]
    if sys.platform.startswith("win"):
        fnames = ["iovs.dll"]
    elif sys.platform == "darwin":
        fnames = ["libiovs.dylib"]
    else:
        fnames = ["libiovs.so"]
    for root in roots:
        for fn in fnames:
            p = root / fn
            if p.exists():
                return ctypes.CDLL(str(p))
    raise FileNotFoundError("libiovs not found; set IOVS_LIBRARY or build the C++ library")


_lib = _load()
_lib.iovsGetVersion.restype = c_char_p
_lib.iovsResourcesCreate.argtypes = [POINTER(c_void_p)]
_lib.iovsResourcesDestroy.argtypes = [c_void_p]
_lib.iovsBruteForceBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    POINTER(c_void_p),
]
_lib.iovsBruteForceSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.iovsBruteForceDestroy.argtypes = [c_void_p]
_lib.iovsGemm.argtypes = [
    c_void_p,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int64,
    c_int32,
]
_lib.iovsTopk.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int64,
    POINTER(c_int64),
    POINTER(c_float),
    c_int32,
]
_lib.iovsGatherRows.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    POINTER(c_int64),
    c_int64,
    POINTER(c_float),
]
_lib.iovsResourcesSetPolicy.argtypes = [c_void_p, c_int32]
_lib.iovsResourcesLastDevice.argtypes = [c_void_p, POINTER(c_int32)]


def version() -> str:
    return _lib.iovsGetVersion().decode()


class Resources:
    def __init__(self):
        self._h = c_void_p()
        if _lib.iovsResourcesCreate(ctypes.byref(self._h)) != 0:
            raise RuntimeError("iovsResourcesCreate failed")

    def close(self):
        if self._h:
            _lib.iovsResourcesDestroy(self._h)
            self._h = None

    def set_policy(self, policy: int) -> None:
        if _lib.iovsResourcesSetPolicy(self._h, c_int32(policy)) != 0:
            raise RuntimeError("set_policy failed")

    def last_device(self) -> int:
        d = c_int32()
        if _lib.iovsResourcesLastDevice(self._h, ctypes.byref(d)) != 0:
            raise RuntimeError("last_device failed")
        return int(d.value)

    def gemm(self, a, b, m, n, k, trans_b=1):
        cbuf = (c_float * (m * n))()
        ap = (c_float * len(a))(*list(a))
        bp = (c_float * len(b))(*list(b))
        rc = _lib.iovsGemm(self._h, ap, bp, cbuf, c_int64(m), c_int64(n), c_int64(k), c_int32(trans_b))
        if rc != 0:
            raise RuntimeError(f"gemm failed rc={rc}")
        return [cbuf[i] for i in range(m * n)]

    def topk(self, scores, rows, cols, k, largest=0):
        idx = (c_int64 * (rows * k))()
        val = (c_float * (rows * k))()
        sp = (c_float * len(scores))(*list(scores))
        rc = _lib.iovsTopk(self._h, sp, c_int64(rows), c_int64(cols), c_int64(k), idx, val, c_int32(largest))
        if rc != 0:
            raise RuntimeError(f"topk failed rc={rc}")
        return [idx[i] for i in range(rows * k)], [val[i] for i in range(rows * k)]

    def gather_rows(self, src, src_rows, dim, indices):
        nidx = len(indices)
        out = (c_float * (nidx * dim))()
        sp = (c_float * len(src))(*list(src))
        ip = (c_int64 * nidx)(*list(indices))
        rc = _lib.iovsGatherRows(self._h, sp, c_int64(src_rows), c_int64(dim), ip, c_int64(nidx), out)
        if rc != 0:
            raise RuntimeError(f"gather failed rc={rc}")
        return [out[i] for i in range(nidx * dim)]

    def __del__(self):
        self.close()


def _f32_ptr(arr):
    import array

    if isinstance(arr, array.array):
        return arr.buffer_info()[0]
    if hasattr(arr, "ctypes"):
        return arr.ctypes.data_as(POINTER(c_float))
    buf = (c_float * len(arr))(*arr)
    return buf


class neighbors:
    class brute_force:
        @staticmethod
        def build(dataset, dim=None, metric=0, resources=None):
            import array

            res = resources or Resources()
            n = len(dataset) // (dim or 1) if dim else 0
            if hasattr(dataset, "shape"):
                n, dim = int(dataset.shape[0]), int(dataset.shape[1])
                data = dataset.astype("float32").ravel()
            else:
                data = array.array("f", dataset)
            ix = c_void_p()
            ptr = (c_float * len(data))(*list(data))
            rc = _lib.iovsBruteForceBuild(res._h, ptr, c_int64(n), c_int64(dim), c_int32(metric), ctypes.byref(ix))
            if rc != 0:
                raise RuntimeError("brute_force.build failed")
            return BruteIndex(res, ix, dim, own_res=resources is None)

    build = brute_force.build


class BruteIndex:
    def __init__(self, res, handle, dim, own_res):
        self._res = res
        self._h = handle
        self.dim = dim
        self._own = own_res

    def search(self, queries, k=10):
        import array

        if hasattr(queries, "shape"):
            nq, dim = int(queries.shape[0]), int(queries.shape[1])
            q = queries.astype("float32").ravel()
        else:
            q = array.array("f", queries)
            nq = len(q) // self.dim
            dim = self.dim
        qptr = (c_float * len(q))(*list(q))
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.iovsBruteForceSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), None, nb, ds
        )
        if rc != 0:
            raise RuntimeError("search failed")
        return [nb[i] for i in range(nq * k)], [ds[i] for i in range(nq * k)]

    def close(self):
        if self._h:
            _lib.iovsBruteForceDestroy(self._h)
            self._h = None
        if self._own and self._res:
            self._res.close()
            self._res = None
