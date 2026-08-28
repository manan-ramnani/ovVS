"""ovVS Python bindings — ctypes over the stable C ABI. numpy arrays preferred."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int32, c_int64, c_uint8, c_void_p
from pathlib import Path


def _load():
    here = Path(__file__).resolve().parent
    if sys.platform.startswith("win"):
        fnames = ["ovvs.dll"]
    elif sys.platform == "darwin":
        fnames = ["libovvs.dylib"]
    else:
        fnames = ["libovvs.so"]
    candidates = []
    env = os.environ.get("OVVS_LIBRARY")
    if env:
        candidates.append(Path(env))
    roots = [
        here,
        here.parent,
        here.parent.parent / "build-icpx" / "bin",
        here.parent.parent / "build-sycl" / "bin",
        here.parent.parent / "build" / "bin",
        here.parent.parent / "build" / "Release",
        here.parent.parent / "build" / "Debug",
    ]
    for root in roots:
        for fn in fnames:
            candidates.append(root / fn)
    for p in candidates:
        if p.exists():
            p = p.resolve()
            if sys.platform.startswith("win"):
                os.add_dll_directory(str(p.parent))
            return ctypes.CDLL(str(p))
    raise FileNotFoundError("libovvs not found; set OVVS_LIBRARY or build the C++ library")


_lib = _load()
_lib.ovvsGetVersion.restype = c_char_p
_lib.ovvsStatusString.argtypes = [c_int32]
_lib.ovvsStatusString.restype = c_char_p
_lib.ovvsResourcesCreate.argtypes = [POINTER(c_void_p)]
_lib.ovvsResourcesDestroy.argtypes = [c_void_p]
_lib.ovvsBruteForceBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    POINTER(c_void_p),
]
_lib.ovvsBruteForceSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.ovvsBruteForceDestroy.argtypes = [c_void_p]
_lib.ovvsCagraBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.ovvsCagraSearch.argtypes = [
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
_lib.ovvsCagraDestroy.argtypes = [c_void_p]
_lib.ovvsIvfFlatBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.ovvsIvfFlatSearch.argtypes = [
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
_lib.ovvsIvfFlatDestroy.argtypes = [c_void_p]
_lib.ovvsIvfPqBuild.argtypes = [
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
_lib.ovvsIvfPqSearch.argtypes = [
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
_lib.ovvsIvfPqDestroy.argtypes = [c_void_p]
_lib.ovvsGemm.argtypes = [
    c_void_p,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int64,
    c_int32,
]
_lib.ovvsTopk.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int64,
    POINTER(c_int64),
    POINTER(c_float),
    c_int32,
]
_lib.ovvsGatherRows.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    POINTER(c_int64),
    c_int64,
    POINTER(c_float),
]
_lib.ovvsResourcesSetPolicy.argtypes = [c_void_p, c_int32]
_lib.ovvsResourcesLastDevice.argtypes = [c_void_p, POINTER(c_int32)]
_lib.ovvsResourcesLastComputeDtype.argtypes = [c_void_p, POINTER(c_int32)]
_lib.ovvsResourcesEnergyUj.argtypes = [c_void_p, POINTER(c_int64)]
_lib.ovvsGemmEx.argtypes = [
    c_void_p,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int64,
    c_int32,
    c_int32,
]
_lib.ovvsBitsetFromAllowList.argtypes = [c_int64, POINTER(c_int64), c_int64, POINTER(c_uint8)]
_lib.ovvsIvfPqSerialize.argtypes = [c_void_p, c_char_p]
_lib.ovvsIvfPqDeserialize.argtypes = [c_void_p, c_char_p, POINTER(c_void_p)]
_lib.ovvsIvfPqExtend.argtypes = [c_void_p, c_void_p, POINTER(c_float), c_int64]
_lib.ovvsIvfRabitqBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.ovvsIvfRabitqSearch.argtypes = [
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
_lib.ovvsIvfRabitqDestroy.argtypes = [c_void_p]
_lib.ovvsIvfRabitqSerialize.argtypes = [c_void_p, c_char_p]
_lib.ovvsIvfRabitqDeserialize.argtypes = [c_void_p, c_char_p, POINTER(c_void_p)]
_lib.ovvsIvfRabitqExtend.argtypes = [c_void_p, c_void_p, POINTER(c_float), c_int64]
_lib.ovvsSyclEnabled.restype = c_int32
_lib.ovvsSyclEnabled.argtypes = []
_lib.ovvsVamanaBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_float,
    POINTER(c_void_p),
]
_lib.ovvsVamanaSearch.argtypes = [
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
_lib.ovvsVamanaDestroy.argtypes = [c_void_p]
_lib.ovvsScannBuild.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    c_int32,
    POINTER(c_void_p),
]
_lib.ovvsScannSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    c_int32,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.ovvsScannDestroy.argtypes = [c_void_p]
_lib.ovvsHnswFromCagra.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
_lib.ovvsHnswSearch.argtypes = [
    c_void_p,
    c_void_p,
    POINTER(c_float),
    c_int64,
    c_int64,
    c_int32,
    POINTER(c_int64),
    POINTER(c_float),
]
_lib.ovvsHnswDestroy.argtypes = [c_void_p]
_lib.ovvsKMeansFit.argtypes = [c_void_p, POINTER(c_float), c_int64, c_int64, c_int32, c_int32, POINTER(c_void_p)]
_lib.ovvsKMeansPredict.argtypes = [c_void_p, c_void_p, POINTER(c_float), c_int64, POINTER(c_int64), POINTER(c_float)]
_lib.ovvsKMeansDestroy.argtypes = [c_void_p]
_lib.ovvsBatcherCreate.argtypes = [c_void_p, c_void_p, c_int32, c_int32, POINTER(c_void_p)]
_lib.ovvsBatcherSearch.argtypes = [c_void_p, POINTER(c_float), c_int64, c_int64, POINTER(c_int64), POINTER(c_float)]
_lib.ovvsBatcherDestroy.argtypes = [c_void_p]

METRIC_L2 = 0
METRIC_L2_SQRT = 1
METRIC_INNER_PRODUCT = 2
METRIC_COSINE = 3
METRIC_HAMMING = 4
METRIC_LP = 5


class OvvsError(RuntimeError):
    def __init__(self, operation: str, status: int):
        self.operation = operation
        self.status = int(status)
        raw = _lib.ovvsStatusString(c_int32(status))
        detail = raw.decode(errors="replace") if raw else "unknown"
        super().__init__(f"{operation} failed: {detail} (status={status})")


def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise OvvsError(operation, status)


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
    return _lib.ovvsGetVersion().decode()


def sycl_enabled() -> bool:
    return bool(_lib.ovvsSyclEnabled())


class Resources:
    def __init__(self):
        self._h = c_void_p()
        if _lib.ovvsResourcesCreate(ctypes.byref(self._h)) != 0:
            raise RuntimeError("ovvsResourcesCreate failed")

    def close(self):
        if self._h:
            _lib.ovvsResourcesDestroy(self._h)
            self._h = None

    def set_policy(self, policy: int) -> None:
        if _lib.ovvsResourcesSetPolicy(self._h, c_int32(policy)) != 0:
            raise RuntimeError("set_policy failed")

    def last_device(self) -> int:
        d = c_int32()
        if _lib.ovvsResourcesLastDevice(self._h, ctypes.byref(d)) != 0:
            raise RuntimeError("last_device failed")
        return int(d.value)

    def last_compute_dtype(self) -> int:
        d = c_int32()
        if _lib.ovvsResourcesLastComputeDtype(self._h, ctypes.byref(d)) != 0:
            raise RuntimeError("last_compute_dtype failed")
        return int(d.value)

    def energy_uj(self):
        uj = c_int64()
        rc = _lib.ovvsResourcesEnergyUj(self._h, ctypes.byref(uj))
        if rc != 0:
            return None
        return int(uj.value)

    def gemm(self, a, b, m, n, k, trans_b=1, compute_dtype=0):
        cbuf = (c_float * (m * n))()
        _, ap = _as_f32(a)
        _, bp = _as_f32(b)
        if compute_dtype:
            rc = _lib.ovvsGemmEx(
                self._h, ap, bp, cbuf, c_int64(m), c_int64(n), c_int64(k), c_int32(trans_b),
                c_int32(compute_dtype),
            )
        else:
            rc = _lib.ovvsGemm(self._h, ap, bp, cbuf, c_int64(m), c_int64(n), c_int64(k), c_int32(trans_b))
        if rc != 0:
            raise RuntimeError(f"gemm failed rc={rc}")
        return [cbuf[i] for i in range(m * n)]

    def topk(self, scores, rows, cols, k, largest=0):
        idx = (c_int64 * (rows * k))()
        val = (c_float * (rows * k))()
        _, sp = _as_f32(scores)
        rc = _lib.ovvsTopk(self._h, sp, c_int64(rows), c_int64(cols), c_int64(k), idx, val, c_int32(largest))
        if rc != 0:
            raise RuntimeError(f"topk failed rc={rc}")
        return [idx[i] for i in range(rows * k)], [val[i] for i in range(rows * k)]

    def gather_rows(self, src, src_rows, dim, indices):
        nidx = len(indices)
        out = (c_float * (nidx * dim))()
        _, sp = _as_f32(src)
        ip = (c_int64 * nidx)(*list(indices))
        rc = _lib.ovvsGatherRows(self._h, sp, c_int64(src_rows), c_int64(dim), ip, c_int64(nidx), out)
        if rc != 0:
            raise RuntimeError(f"gather failed rc={rc}")
        return [out[i] for i in range(nidx * dim)]

    def __del__(self):
        self.close()


def bitset_from_allow_list(n, ids):
    n = int(n)
    seq = list(ids)
    nids = len(seq)
    nbytes = (n + 7) // 8
    buf = (c_uint8 * max(nbytes, 1))()
    ip = (c_int64 * max(nids, 1))(*([int(x) for x in seq] if nids else []))
    rc = _lib.ovvsBitsetFromAllowList(
        c_int64(n), ip if nids else None, c_int64(nids), buf
    )
    if rc != 0:
        raise RuntimeError("ovvsBitsetFromAllowList failed")
    return buf


def _bitset_ptr(n, bitset=None, allow_list=None):
    if allow_list is not None:
        buf = bitset_from_allow_list(n, allow_list)
        return buf, ctypes.cast(buf, c_void_p)
    if bitset is None:
        return None, None
    try:
        import numpy as np

        if isinstance(bitset, np.ndarray):
            a = np.ascontiguousarray(bitset, dtype=np.uint8)
            return a, a.ctypes.data_as(c_void_p)
    except ImportError:
        pass
    if isinstance(bitset, (bytes, bytearray)):
        buf = (c_uint8 * len(bitset)).from_buffer_copy(bitset)
        return buf, ctypes.cast(buf, c_void_p)
    return bitset, ctypes.cast(bitset, c_void_p)


class _Index:
    def __init__(self, res, handle, dim, own_res, destroy, n=0):
        self._res = res
        self._h = handle
        self.dim = dim
        self.n = n
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
    def search(self, queries, k=10, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsBruteForceSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), bsp, nb, ds
        )
        _check_status(rc, "brute-force search")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class CagraIndex(_Index):
    def search(self, queries, k=10, itopk_size=32, search_width=2, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsCagraSearch(
            self._res._h,
            self._h,
            qptr,
            c_int64(nq),
            c_int64(k),
            c_int32(itopk_size),
            c_int32(search_width),
            bsp,
            nb,
            ds,
        )
        _check_status(rc, "CAGRA search")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class IvfFlatIndex(_Index):
    def search(self, queries, k=10, nprobe=8, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsIvfFlatSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), c_int32(nprobe), bsp, nb, ds
        )
        _check_status(rc, "IVF-Flat search")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class IvfPqIndex(_Index):
    def search(self, queries, k=10, nprobe=8, krefine=32, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsIvfPqSearch(
            self._res._h,
            self._h,
            qptr,
            c_int64(nq),
            c_int64(k),
            c_int32(nprobe),
            c_int32(krefine),
            bsp,
            nb,
            ds,
        )
        _check_status(rc, "IVF-PQ search")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)

    def serialize(self, path):
        rc = _lib.ovvsIvfPqSerialize(self._h, str(path).encode())
        if rc != 0:
            raise RuntimeError("ivf-pq serialize failed")

    def extend(self, extra):
        keep, ptr, n, _d = _dataset_ptr(extra, self.dim)
        rc = _lib.ovvsIvfPqExtend(self._res._h, self._h, ptr, c_int64(n))
        if rc != 0:
            raise RuntimeError("ivf-pq extend failed")
        self.n += n
        return keep


class IvfRabitqIndex(_Index):
    def search(self, queries, k=10, nprobe=8, krefine=32, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsIvfRabitqSearch(
            self._res._h,
            self._h,
            qptr,
            c_int64(nq),
            c_int64(k),
            c_int32(nprobe),
            c_int32(krefine),
            bsp,
            nb,
            ds,
        )
        _check_status(rc, "IVF-RaBitQ search")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)

    def serialize(self, path):
        rc = _lib.ovvsIvfRabitqSerialize(self._h, str(path).encode())
        if rc != 0:
            raise RuntimeError("ivf-rabitq serialize failed")

    def extend(self, extra):
        keep, ptr, n, _d = _dataset_ptr(extra, self.dim)
        rc = _lib.ovvsIvfRabitqExtend(self._res._h, self._h, ptr, c_int64(n))
        if rc != 0:
            raise RuntimeError("ivf-rabitq extend failed")
        self.n += n
        return keep


def _from_dlpack(obj):
    """Return a C-contiguous float32 ndarray from a DLPack exporter, or None."""
    try:
        import numpy as np

        if hasattr(obj, "__dlpack__"):
            return np.ascontiguousarray(np.from_dlpack(obj), dtype=np.float32)
    except Exception:
        return None
    return None


def _query_ptr(queries, dim):
    try:
        import numpy as np

        dl = _from_dlpack(queries)
        if dl is not None:
            queries = dl
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

        dl = _from_dlpack(dataset)
        if dl is not None:
            dataset = dl
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
            rc = _lib.ovvsBruteForceBuild(res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), ctypes.byref(ix))
            _check_status(rc, "brute-force build")
            return BruteIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsBruteForceDestroy, n=n)

    class cagra:
        @staticmethod
        def build(dataset, dim=None, metric=0, graph_degree=16, intermediate_degree=32, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsCagraBuild(
                res._h,
                ptr,
                c_int64(n),
                c_int64(d),
                c_int32(metric),
                c_int32(graph_degree),
                c_int32(intermediate_degree),
                ctypes.byref(ix),
            )
            _check_status(rc, "CAGRA build")
            return CagraIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsCagraDestroy, n=n)

    class ivf_flat:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsIvfFlatBuild(
                res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), c_int32(nlist), ctypes.byref(ix)
            )
            _check_status(rc, "IVF-Flat build")
            return IvfFlatIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsIvfFlatDestroy, n=n)

    class ivf_pq:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, pq_m=8, pq_nbits=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsIvfPqBuild(
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
            _check_status(rc, "IVF-PQ build")
            return IvfPqIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsIvfPqDestroy, n=n)

    class ivf_rabitq:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsIvfRabitqBuild(
                res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), c_int32(nlist), ctypes.byref(ix)
            )
            if rc != 0:
                raise RuntimeError("ivf_rabitq.build failed")
            return IvfRabitqIndex(
                res, ix, d, own_res=resources is None, destroy=_lib.ovvsIvfRabitqDestroy, n=n
            )

    class vamana:
        @staticmethod
        def build(dataset, dim=None, metric=0, graph_degree=16, alpha=1.2, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsVamanaBuild(
                res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), c_int32(graph_degree),
                c_float(alpha), ctypes.byref(ix)
            )
            if rc != 0:
                raise RuntimeError("vamana.build failed")
            return VamanaIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsVamanaDestroy, n=n)

    class scann:
        @staticmethod
        def build(dataset, dim=None, metric=0, nlist=8, pq_m=8, resources=None):
            res = resources or Resources()
            keep, ptr, n, d = _dataset_ptr(dataset, dim)
            ix = c_void_p()
            rc = _lib.ovvsScannBuild(
                res._h, ptr, c_int64(n), c_int64(d), c_int32(metric), c_int32(nlist), c_int32(pq_m),
                ctypes.byref(ix)
            )
            if rc != 0:
                raise RuntimeError("scann.build failed")
            return ScannIndex(res, ix, d, own_res=resources is None, destroy=_lib.ovvsScannDestroy, n=n)

    class hnsw:
        @staticmethod
        def from_cagra(cagra_index, resources=None):
            res = resources or cagra_index._res
            ix = c_void_p()
            rc = _lib.ovvsHnswFromCagra(res._h, cagra_index._h, ctypes.byref(ix))
            if rc != 0:
                raise RuntimeError("hnsw.from_cagra failed")
            return HnswIndex(res, ix, cagra_index.dim, own_res=False, destroy=_lib.ovvsHnswDestroy, n=cagra_index.n)

    build = brute_force.build


class VamanaIndex(_Index):
    def search(self, queries, k=10, beam=32, bitset=None, allow_list=None):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        bskeep, bsp = _bitset_ptr(self.n, bitset, allow_list)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsVamanaSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), c_int32(beam), bsp, nb, ds
        )
        if rc != 0:
            raise RuntimeError("vamana search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class ScannIndex(_Index):
    def search(self, queries, k=10, nprobe=8, krefine=32):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsScannSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), c_int32(nprobe), c_int32(krefine), nb, ds
        )
        if rc != 0:
            raise RuntimeError("scann search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class HnswIndex(_Index):
    def search(self, queries, k=10, ef=32):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsHnswSearch(
            self._res._h, self._h, qptr, c_int64(nq), c_int64(k), c_int32(ef), nb, ds
        )
        if rc != 0:
            raise RuntimeError("hnsw search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)


class KMeans:
    def __init__(self, dataset, nclusters=2, iters=12, dim=None, resources=None):
        self._res = resources or Resources()
        self._own = resources is None
        keep, ptr, n, d = _dataset_ptr(dataset, dim)
        self._h = c_void_p()
        rc = _lib.ovvsKMeansFit(
            self._res._h, ptr, c_int64(n), c_int64(d), c_int32(nclusters), c_int32(iters), ctypes.byref(self._h)
        )
        if rc != 0:
            raise RuntimeError("kmeans fit failed")
        self.dim = d
        self._keep = keep

    def predict(self, x):
        keep, ptr, n, _d = _dataset_ptr(x, self.dim)
        labs = (c_int64 * n)()
        dist = (c_float * n)()
        rc = _lib.ovvsKMeansPredict(self._res._h, self._h, ptr, c_int64(n), labs, dist)
        if rc != 0:
            raise RuntimeError("kmeans predict failed")
        return [labs[i] for i in range(n)], [dist[i] for i in range(n)]

    def close(self):
        if self._h:
            _lib.ovvsKMeansDestroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


class Batcher:
    def __init__(self, brute_index, max_batch=8, max_wait_ms=0):
        self._b = c_void_p()
        rc = _lib.ovvsBatcherCreate(
            brute_index._res._h, brute_index._h, c_int32(max_batch), c_int32(max_wait_ms), ctypes.byref(self._b)
        )
        if rc != 0:
            raise RuntimeError("batcher create failed")
        self.dim = brute_index.dim

    def search(self, queries, k=10):
        qkeep, qptr, nq = _query_ptr(queries, self.dim)
        nb = (c_int64 * (nq * k))()
        ds = (c_float * (nq * k))()
        rc = _lib.ovvsBatcherSearch(self._b, qptr, c_int64(nq), c_int64(k), nb, ds)
        if rc != 0:
            raise RuntimeError("batcher search failed")
        return _maybe_np(nb, nq, k), _maybe_np_f(ds, nq, k)

    def close(self):
        if self._b:
            _lib.ovvsBatcherDestroy(self._b)
            self._b = None

    def __del__(self):
        self.close()
