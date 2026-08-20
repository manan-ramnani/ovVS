"""ioVS Python bindings — ctypes over the stable C ABI. numpy arrays preferred."""

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
_lib.iovsCagraBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.iovsCagraSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.iovsCagraDestroy.argtypes = [c_void_p]
_lib.iovsIvfFlatBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.iovsIvfFlatSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.iovsIvfFlatDestroy.argtypes = [c_void_p]
_lib.iovsIvfPqBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.iovsIvfPqSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.iovsIvfPqDestroy.argtypes = [c_void_p]
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


def _as_f32(arr):
    try:
        import numpy as np

        if isinstance(arr, np.ndarray):
            a = np.ascontiguousarray(arr, dtype=np.float32)
            return a, a.ctypes.data_as(POINTER(c_float))
    except ImportError:
        pass
    buf = (c_float * len(arr))(*list(arr))
    return buf, buf


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
        _, ap = _as_f32(a)
        _, bp = _as_f32(b)
        rc = _lib.iovsGemm(self._h, ap, bp, cbuf, c_int64(m), c_int64(n), c_int64(k), c_int32(trans_b))
        if rc != 0:
            raise RuntimeError(f"gemm failed rc={rc}")
        return [cbuf[i] for i in range(m * n)]

    def topk(self, scores, rows, cols, k, largest=0):
        idx = (c_int64 * (rows * k))()
        val = (c_float * (rows * k))()
        _, sp = _as_f32(scores)
        rc = _lib.iovsTopk(self._h, sp, c_int64(rows), c_int64(cols), c_int64(k), idx, val, c_int32(largest))
        if rc != 0:
            raise RuntimeError(f"topk failed rc={rc}")
        return [idx[i] for i in range(rows * k)], [val[i] for i in range(rows * k)]

    def gather_rows(self, src, src_rows, dim, indices):
        nidx = len(indices)
        out = (c_float * (nidx * dim))()
        _, sp = _as_f32(src)
        ip = (c_int64 * nidx)(*list(indices))
        rc = _lib.iovsGatherRows(self._h, sp, c_int64(src_rows), c_int64(dim), ip, c_int64(nidx), out)
        if rc != 0:
            raise RuntimeError(f"gather failed rc={rc}")
        return [out[i] for i in range(nidx * dim)]

    def __del__(self):
        self.close()


class _Index:
    def __init__(self, res, handle, dim, own_res, destroy):
        self._res = res
        self._h = handle
        self.dim = dim
        self._own = own_res
        self._destroy = destroy

    def close(self):
        if self._h:
            self._destroy(self._h)
            self._h = None
        if self._own and self._res:
            self._res.close()
            self._res = None

    def __del__(self):
        self.close()


class BruteIndex(_Index):
    def search(self, queries, k=10):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.iovsBruteForceSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), None, nb, ds
        )
        if rc != 0:
            raise RuntimeError("search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class CagraIndex(_Index):
    def search(self, queries, k=10, itopk_size=32, search_width=2):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.iovsCagraSearch(
            self._res._h,
            self._h,
            qptr,
            c_int64(nq),
            c_int64(k),
            c_int32(itopk_size),
            c_int32(search_width),
            None,
            nb,
            ds,
        )
        if rc != 0:
            raise RuntimeError("cagra search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class IvfFlatIndex(_Index):
    def search(self, queries, k=10, nprobe=8):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.iovsIvfFlatSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), c_int32(nprobe), None, nb, ds
        )
        if rc != 0:
            raise RuntimeError("ivf search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class IvfPqIndex(_Index):
    def search(self, queries, k=10, nprobe=8, krefine=32):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.iovsIvfPqSearch(
            self._res._h,
            self._h,
            qptr,
            c_int64(nq),
            c_int64(k),
            c_int32(nprobe),
            c_int32(krefine),
            None,
            nb,
            ds,
        )
        if rc != 0:
            raise RuntimeError("ivf-pq search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


def _query_ptr(queries, dim):
    try:
        import numpy as np

        if hasattr(queries, "shape"):
            a = np.ascontiguousarray(queries, dtype=np.float32)
            nq = int(a.shape[0]) if a.ndim == 2 else 1
            return a, a.ctypes.data_as(POINTER(c_float)), nq
    except ImportError:
        pass
    import array

    q = array.array("f", queries)
    nq = len(q) // dim
    buf = (c_float * len(q))(*list(q))
    return buf, buf, nq


def _maybe_np(nb, nq, k):
    try:
        import numpy as np

        return np.frombuffer((c_int64 * (nq * k))(*nb), dtype=np.int64).reshape(nq, k).copy()
    except Exception:
        return [nb[i] for i in range(nq * k)]


def _maybe_np_f(ds, nq, k):
    try:
        import numpy as np

        return np.frombuffer((c_float * (nq * k))(*ds), dtype=np.float32).reshape(nq, k).copy()
    except Exception:
        return [ds[i] for i in range(nq * k)]


def _dataset_ptr(dataset, dim):
    try:
        import numpy as np

        if hasattr(dataset, "shape"):
            a = np.ascontiguousarray(dataset, dtype=np.float32)
            n, d = int(a.shape[0]), int(a.shape[1])
            return a, a.ctypes.data_as(POINTER(c_float)), n, d
    except ImportError:
        pass
    import array

    data = array.array("f", dataset)
    n = len(data) // (dim or 1)
    buf = (c_float * len(data))(*list(data))
    return buf, buf, n, dim


class neighbors:
    class brute_force:
        @staticmethod
        def build(dataset, dim=None, metric=0, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.iovsBruteForceBuild(res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), ctypes.byref(ix))
            if rc != 0:
                raise RuntimeError("brute_force.build failed")
            return BruteIndex(res, ix, d, own_res=resources is None, destroy=_lib.iovsBruteForceDestroy)

    class cagra:
        @staticmethod
        def build(dataset, dim=None, metric=0, graph_degree=16, intermediate_degree=32, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.iovsCagraBuild(
                res._h,
                ptr,
                c_int64(n),
                c_int64(d),
                c_int32(metric),
                c_int32(graph_degree),
                c_int32(intermediate_degree),
                ctypes.byref(ix),
            )
            if rc != 0:
                raise RuntimeError("cagra.build failed")
            return CagraIndex(res, ix, d, own_res=resources is None, destroy=_lib.iovsCagraDestroy)

    class ivf_flat:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.iovsIvfFlatBuild(
                res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), c_int32(nlist), ctypes.byref(ix)
            )
            if rc != 0:
                raise RuntimeError("ivf_flat.build failed")
            return IvfFlatIndex(res, ix, d, own_res=resources is None, destroy=_lib.iovsIvfFlatDestroy)

    class ivf_pq:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, pq_m=8, pq_nbits=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.iovsIvfPqBuild(
                res._h,
                ptr,
                c_int64(n),
                c_int64(d),
                c_int32(metric),
                c_int32(nlist),
                c_int32(pq_m),
                c_int32(pq_nbits),
                ctypes.byref(ix),
            )
            if rc != 0:
                raise RuntimeError("ivf_pq.build failed")
            return IvfPqIndex(res, ix, d, own_res=resources is None, destroy=_lib.iovsIvfPqDestroy)

    build = brute_force.build
