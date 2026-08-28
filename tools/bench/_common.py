"""Shared contracts and pure helpers for the ovVS benchmark harness."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "out" / "bench"
SCHEMA_VERSION = "ovvs.benchmark.v1"
SIFT_SHA256 = "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984"
WORKER_PREFIX = "__OVVS_BENCH_RESULT__="
LANE_STATUSES = {"success", "unavailable", "failed", "timeout", "skipped"}

ALGORITHM_ORDER = ("brute", "ivf-flat", "ivf-pq", "cagra")
POLICY_ORDER = ("auto", "cpu", "npu", "gpu", "hetero")
POLICY_VALUES = {"auto": 0, "hetero": 3, "npu": 4, "gpu": 5, "cpu": 6}
POLICY_LABELS = {
    "auto": "AUTO",
    "cpu": "FORCE_CPU",
    "npu": "FORCE_NPU",
    "gpu": "FORCE_GPU",
    "hetero": "HETERO",
}
DEVICE_LABELS = {0: "AUTO", 1: "CPU", 2: "NPU", 3: "GPU", 4: "HETERO"}


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "description": "Bounded smoke: SIFT prefix when readable, otherwise an explicit synthetic fallback.",
        "expected_n": 2_000,
        "expected_dim": 128,
        "nq": 32,
        "k": 10,
        "query_batch_size": 32,
        "warmups": 1,
        "repeats": 3,
        "timeout_seconds": 30.0,
        "energy_min_seconds": 0.15,
        "energy_max_passes": 8,
        "nlist": 32,
        "nprobes": [2, 8],
        "pq_m": 8,
        "pq_nbits": 8,
        "krefine": 32,
        "graph_degree": 16,
        "intermediate_degree": 32,
        "cagra_points": [
            {"itopk_size": 32, "search_width": 1, "query_batch_size": 32},
            {"itopk_size": 64, "search_width": 2, "query_batch_size": 32},
            {"itopk_size": 64, "search_width": 2, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
        "hnsw_ef_construction": 100,
        "hnsw_points": [
            {"ef": 32, "query_batch_size": 32},
            {"ef": 64, "query_batch_size": 32},
            {"ef": 64, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
    },
    "sift1m": {
        "description": "Full one-million-vector SIFT base with a 1,000-query measured prefix.",
        "expected_n": 1_000_000,
        "expected_dim": 128,
        "nq": 1_000,
        "k": 10,
        "query_batch_size": 32,
        "warmups": 1,
        "repeats": 5,
        "timeout_seconds": 3_600.0,
        "energy_min_seconds": 0.5,
        "energy_max_passes": 3,
        "nlist": 1_024,
        "nprobes": [8, 16, 32, 64],
        "pq_m": 16,
        "pq_nbits": 8,
        "krefine": 64,
        "graph_degree": 16,
        "intermediate_degree": 32,
        "cagra_points": [
            {"itopk_size": 32, "search_width": 1, "query_batch_size": 32},
            {"itopk_size": 64, "search_width": 2, "query_batch_size": 32},
            {"itopk_size": 128, "search_width": 4, "query_batch_size": 32},
            {"itopk_size": 128, "search_width": 4, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
        "hnsw_ef_construction": 200,
        "hnsw_points": [
            {"ef": 32, "query_batch_size": 32},
            {"ef": 64, "query_batch_size": 32},
            {"ef": 128, "query_batch_size": 32},
            {"ef": 256, "query_batch_size": 32},
            {"ef": 256, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
    },
    "embedding-100k": {
        "description": "100,000 x 768 deterministic synthetic or user-supplied embedding workload.",
        "expected_n": 100_000,
        "expected_dim": 768,
        "nq": 32,
        "k": 10,
        "query_batch_size": 32,
        "warmups": 1,
        "repeats": 5,
        "timeout_seconds": 1_800.0,
        "energy_min_seconds": 0.5,
        "energy_max_passes": 3,
        "nlist": 1_024,
        "nprobes": [8, 16, 32, 64],
        "pq_m": 32,
        "pq_nbits": 8,
        "krefine": 64,
        "graph_degree": 16,
        "intermediate_degree": 32,
        "cagra_points": [
            {"itopk_size": 32, "search_width": 1, "query_batch_size": 32},
            {"itopk_size": 64, "search_width": 2, "query_batch_size": 32},
            {"itopk_size": 128, "search_width": 4, "query_batch_size": 32},
            {"itopk_size": 128, "search_width": 4, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
        "hnsw_ef_construction": 200,
        "hnsw_points": [
            {"ef": 32, "query_batch_size": 32},
            {"ef": 64, "query_batch_size": 32},
            {"ef": 128, "query_batch_size": 32},
            {"ef": 256, "query_batch_size": 32},
            {"ef": 256, "query_batch_size": 1, "purpose": "b1_latency"},
        ],
    },
}


class UnavailableError(RuntimeError):
    """A required package, library, dataset, or device is absent."""


@dataclass(frozen=True)
class Lane:
    id: str
    implementation: str
    algorithm: str
    policy: str | None = None
    mandatory: bool = False
    skip_reason: str | None = None
    expected_skip: bool = False
    blocking_skip: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "implementation": self.implementation,
            "algorithm": self.algorithm,
            "policy": POLICY_LABELS.get(self.policy, self.policy),
            "policy_key": self.policy,
            "mandatory": self.mandatory,
            "skip_reason": self.skip_reason,
            "expected_skip": self.expected_skip,
            "blocking_skip": self.blocking_skip,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize_samples(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def duplicate_safe_recall(got: Any, truth: Any, k: int) -> float:
    """Mean recall@k using set intersection, so duplicate results never add hits."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(got) != len(truth) or not got:
        raise ValueError("got and truth need the same non-zero query count")
    recalls: list[float] = []
    for got_row, truth_row in zip(got, truth):
        expected = {int(v) for v in list(truth_row)[:k] if int(v) >= 0}
        returned = {int(v) for v in list(got_row)[:k] if int(v) >= 0}
        recalls.append(len(returned & expected) / float(len(expected)) if expected else 0.0)
    return statistics.fmean(recalls)


def validate_neighbors(neighbors: Any, distances: Any, nq: int, k: int, n: int) -> dict[str, Any]:
    import numpy as np

    ids, dists = np.asarray(neighbors), np.asarray(distances)
    shape = (int(nq), int(k))
    issues: list[str] = []
    if tuple(ids.shape) != shape:
        issues.append(f"neighbor shape {tuple(ids.shape)} != {shape}")
    if tuple(dists.shape) != shape:
        issues.append(f"distance shape {tuple(dists.shape)} != {shape}")
    bad_ids = duplicate_rows = nonfinite = 0
    if tuple(ids.shape) == shape:
        bad_ids = int(np.count_nonzero((ids < 0) | (ids >= n)))
        duplicate_rows = sum(len(set(int(v) for v in row)) != k for row in ids)
        if bad_ids:
            issues.append(f"{bad_ids} neighbor IDs outside [0, {n})")
        if duplicate_rows:
            issues.append(f"duplicate neighbor IDs in {duplicate_rows} rows")
    if tuple(dists.shape) == shape:
        nonfinite = int(np.count_nonzero(~np.isfinite(dists)))
        if nonfinite:
            issues.append(f"{nonfinite} non-finite distances")
    return {
        "status": "success" if not issues else "failed",
        "issues": issues,
        "invalid_id_count": bad_ids,
        "duplicate_row_count": duplicate_rows,
        "nonfinite_distance_count": nonfinite,
    }


def parse_selection(raw: str | None, allowed: Sequence[str], label: str) -> list[str]:
    if raw is None or raw.strip().lower() == "all":
        return list(allowed)
    values = [part.strip().lower().replace("_", "-") for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"at least one {label} must be selected")
    aliases = {"force-cpu": "cpu", "force-npu": "npu", "force-gpu": "gpu"}
    values = [aliases.get(value, value) for value in values]
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    selected = set(values)
    return [value for value in allowed if value in selected]


def resolved_profile(name: str, warmups: int | None = None, repeats: int | None = None,
                     timeout_seconds: float | None = None) -> dict[str, Any]:
    profile = json.loads(json.dumps(PROFILE_DEFAULTS[name]))
    if warmups is not None:
        profile["warmups"] = warmups
    if repeats is not None:
        profile["repeats"] = repeats
    if timeout_seconds is not None:
        profile["timeout_seconds"] = timeout_seconds
    if profile["warmups"] < 0:
        raise ValueError("warmups must be non-negative")
    if profile["repeats"] < 2:
        raise ValueError("repeats must be at least 2 for stable statistics")
    if profile["timeout_seconds"] <= 0:
        raise ValueError("timeout must be positive")
    return profile


def enumerate_lanes(algorithms: Sequence[str], policies: Sequence[str], n: int,
                    full_profile: bool, allow_unscalable_cagra: bool) -> list[Lane]:
    lanes: list[Lane] = []
    for algorithm in algorithms:
        for policy in policies:
            reason = None
            expected_skip = False
            blocking_skip = False
            if algorithm == "cagra" and policy == "npu":
                reason = (
                    "CAGRA has iGPU and CPU walks but no NPU walk; FORCE_NPU would execute the host walk "
                    "and cannot be reported as an NPU lane."
                )
                expected_skip = True
            elif algorithm == "cagra" and n > 4_096 and not allow_unscalable_cagra:
                reason = (
                    "Known B5 scalability gate: CAGRA initialization above n=4096 uses the host NN-Descent "
                    "join. Pass --allow-unscalable-cagra to opt in under the lane timeout."
                )
                expected_skip = True
                blocking_skip = True
            lanes.append(
                Lane(
                    f"ovvs.{algorithm}.{policy}", "ovvs", algorithm, policy, full_profile,
                    reason, expected_skip, blocking_skip,
                )
            )
    comparators = {
        "brute": ("faiss-cpu.brute", "faiss-cpu", "brute"),
        "ivf-flat": ("faiss-cpu.ivf-flat", "faiss-cpu", "ivf-flat"),
        "ivf-pq": ("faiss-cpu.ivf-pq", "faiss-cpu", "ivf-pq"),
        "cagra": ("hnswlib.hnsw", "hnswlib", "hnsw"),
    }
    lanes.extend(Lane(*comparators[algorithm], mandatory=full_profile) for algorithm in algorithms)
    return lanes


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_ovvs(explicit_library: str | None = None):
    if explicit_library:
        path = Path(explicit_library).expanduser().resolve()
        if not path.exists():
            raise UnavailableError(f"ovVS library does not exist: {path}")
        os.environ["OVVS_LIBRARY"] = str(path)
    elif "OVVS_LIBRARY" not in os.environ:
        names = ("ovvs.dll",) if sys.platform.startswith("win") else ("libovvs.so", "libovvs.dylib")
        for directory in (
            ROOT / "build-icpx" / "bin",
            ROOT / "build-sycl" / "bin",
            ROOT / "build" / "bin",
            ROOT / "build" / "Release",
        ):
            for name in names:
                candidate = directory / name
                if candidate.exists():
                    os.environ["OVVS_LIBRARY"] = str(candidate)
                    break
            if "OVVS_LIBRARY" in os.environ:
                break
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import ovvs
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise UnavailableError(str(exc)) from exc
    return ovvs


def _resource_scalar(module: Any, resources: Any, symbol: str) -> int | None:
    try:
        fn = getattr(module._lib, symbol)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        value = ctypes.c_int32()
        return int(value.value) if fn(resources._h, ctypes.byref(value)) == 0 else None
    except Exception:
        return None


def ovvs_resource_metadata(module: Any, resources: Any) -> dict[str, Any]:
    metadata = {
        "npu_available": _resource_scalar(module, resources, "ovvsResourcesNpuAvailable"),
        "gpu_available": _resource_scalar(module, resources, "ovvsResourcesGpuAvailable"),
        "npu_compile_fails": _resource_scalar(module, resources, "ovvsResourcesNpuCompileFails"),
        "npu_fallbacks": _resource_scalar(module, resources, "ovvsResourcesNpuFallbacks"),
        "sycl_enabled": bool(module.sycl_enabled()),
    }
    try:
        fn = module._lib.ovvsResourcesSku
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
        buf = ctypes.create_string_buffer(256)
        metadata["sku"] = buf.value.decode(errors="replace") if fn(resources._h, buf, len(buf)) == 0 else None
    except Exception:
        metadata["sku"] = None
    return metadata


def _synthetic(directory: Path, n: int, dim: int, nq: int, seed: int, label: str) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"numpy is required: {exc}"}
    started = time.perf_counter()
    base_path, query_path = directory / f"{label}-base.npy", directory / f"{label}-queries.npy"
    rng = np.random.default_rng(seed)
    base = np.lib.format.open_memmap(base_path, mode="w+", dtype=np.float32, shape=(n, dim))
    chunk = max(1, min(n, (32 * 1024 * 1024) // (dim * 4)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        base[start:stop] = rng.standard_normal((stop - start, dim), dtype=np.float32)
    base.flush()
    del base
    queries = np.lib.format.open_memmap(query_path, mode="w+", dtype=np.float32, shape=(nq, dim))
    queries[:] = rng.standard_normal((nq, dim), dtype=np.float32)
    queries.flush()
    del queries
    generation_spec = {
        "generator": "numpy.default_rng.standard_normal",
        "numpy_version": np.__version__,
        "seed": seed,
        "base_shape": [n, dim],
        "query_shape": [nq, dim],
        "dtype": "float32",
    }
    generation_fingerprint = hashlib.sha256(
        json.dumps(generation_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "success",
        "kind": "synthetic",
        "loader": "npy",
        "base_path": str(base_path),
        "query_path": str(query_path),
        "n": n,
        "source_n": n,
        "dim": dim,
        "nq": nq,
        "source_nq": nq,
        "dtype": "float32",
        "metric": "squared_l2",
        "seed": seed,
        "generation": "numpy.default_rng.standard_normal",
        "source": {
            "type": "ephemeral_generated",
            "generation_spec": generation_spec,
            "generation_spec_sha256": generation_fingerprint,
            "base_size_bytes": base_path.stat().st_size,
            "query_size_bytes": query_path.stat().st_size,
        },
        "preparation_ms": (time.perf_counter() - started) * 1_000,
        "ground_truth_hint": {"method": "faiss_index_flat_l2", "hdf5_neighbors_eligible": False},
    }


def _hdf5(path: Path, profile_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise UnavailableError(f"h5py is required to read {path}: {exc}") from exc
    if not path.exists():
        raise UnavailableError(f"SIFT HDF5 is absent: {path}; run tools/data/fetch.py")
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != SIFT_SHA256:
        raise UnavailableError(
            f"SIFT HDF5 SHA-256 mismatch: expected {SIFT_SHA256}, found {actual_sha256}"
        )
    with h5py.File(path, "r") as handle:
        if any(key not in handle for key in ("train", "test")):
            raise UnavailableError("HDF5 must contain train and test datasets")
        train, test = tuple(handle["train"].shape), tuple(handle["test"].shape)
        if len(train) != 2 or len(test) != 2 or train[1] != test[1]:
            raise UnavailableError(f"unsupported train/test shapes: {train}, {test}")
        if profile_name == "sift1m" and train != (profile["expected_n"], profile["expected_dim"]):
            raise UnavailableError(f"sift1m requires {(profile['expected_n'], profile['expected_dim'])}; found {train}")
        n = int(train[0] if profile_name == "sift1m" else min(profile["expected_n"], train[0]))
        nq, k = int(min(profile["nq"], test[0])), profile["k"]
        neighbor_shape = tuple(handle["neighbors"].shape) if "neighbors" in handle else None
        distance_attr = handle.attrs.get("distance")
        if isinstance(distance_attr, bytes):
            distance_attr = distance_attr.decode(errors="replace")
        distance_name = str(distance_attr).lower() if distance_attr is not None else None
        metric_matches = distance_name in ("euclidean", "l2", "sqeuclidean", "squared_l2")
        eligible = bool(
            n == train[0]
            and metric_matches
            and neighbor_shape
            and neighbor_shape[0] >= nq
            and neighbor_shape[1] >= k
        )
    return {
        "status": "success",
        "kind": "sift1m" if n == train[0] else "sift_prefix",
        "loader": "hdf5",
        "path": str(path.resolve()),
        "base_key": "train",
        "query_key": "test",
        "neighbors_key": "neighbors" if neighbor_shape else None,
        "n": n,
        "source_n": int(train[0]),
        "dim": int(train[1]),
        "nq": nq,
        "source_nq": int(test[0]),
        "dtype": "float32_normalized_on_load",
        "metric": "squared_l2",
        "selection": {"base": f"first {n}", "queries": f"first {nq}"},
        "source": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha256,
            "expected_sha256": SIFT_SHA256,
            "checksum_valid": True,
        },
        "ground_truth_hint": {
            "method": "hdf5_neighbors" if eligible else "faiss_index_flat_l2",
            "hdf5_neighbors_eligible": eligible,
            "neighbors_shape": neighbor_shape,
            "hdf5_distance_attribute": distance_name,
            "reason": (
                "official neighbors apply only to the complete base with an explicit Euclidean/L2 attribute"
                if not eligible
                else "complete base, query prefix, neighbor depth, and Euclidean/L2 attribute validated"
            ),
        },
    }


def _custom(base_path: Path, query_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise UnavailableError(f"numpy is required: {exc}") from exc
    for path in (base_path, query_path):
        if not path.exists() or path.suffix.lower() != ".npy":
            raise UnavailableError(f"custom input must be an existing .npy file: {path}")
    try:
        base = np.load(base_path, mmap_mode="r", allow_pickle=False)
        queries = np.load(query_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise UnavailableError(
            f"could not read custom NumPy inputs: {type(exc).__name__}: {exc}"
        ) from exc
    n, dim, nq = profile["expected_n"], profile["expected_dim"], profile["nq"]
    if base.ndim != 2 or queries.ndim != 2 or base.shape[0] < n or base.shape[1] != dim or queries.shape[0] < nq or queries.shape[1] != dim:
        raise UnavailableError(f"embedding-100k requires base >=({n},{dim}), queries >=({nq},{dim}); found {base.shape}, {queries.shape}")
    return {
        "status": "success",
        "kind": "custom_embedding",
        "loader": "npy",
        "base_path": str(base_path.resolve()),
        "query_path": str(query_path.resolve()),
        "n": n,
        "source_n": int(base.shape[0]),
        "dim": dim,
        "nq": nq,
        "source_nq": int(queries.shape[0]),
        "dtype": "float32_normalized_on_load",
        "metric": "squared_l2",
        "selection": {"base": f"first {n}", "queries": f"first {nq}"},
        "source": {
            "base": {
                "path": str(base_path.resolve()),
                "size_bytes": base_path.stat().st_size,
                "sha256": sha256_file(base_path),
            },
            "queries": {
                "path": str(query_path.resolve()),
                "size_bytes": query_path.stat().st_size,
                "sha256": sha256_file(query_path),
            },
        },
        "ground_truth_hint": {"method": "faiss_index_flat_l2", "hdf5_neighbors_eligible": False},
    }


def prepare_dataset(args: Any, profile: dict[str, Any], directory: Path) -> dict[str, Any]:
    if args.profile in ("smoke", "sift1m"):
        try:
            return _hdf5(Path(args.sift).expanduser(), args.profile, profile)
        except (UnavailableError, OSError, ValueError) as exc:
            if args.profile == "sift1m":
                return {"status": "unavailable", "reason": str(exc), "requested_path": args.sift}
            try:
                result = _synthetic(
                    directory, profile["expected_n"], 32, profile["nq"], args.seed, "smoke"
                )
            except (OSError, ValueError, MemoryError) as synth_exc:
                return {
                    "status": "unavailable",
                    "reason": f"could not prepare smoke fallback: {type(synth_exc).__name__}: {synth_exc}",
                }
            result["fallback"] = {
                "status": "explicit",
                "reason": str(exc),
                "note": "Synthetic fallback is permitted only for smoke.",
            }
            return result
    if bool(args.base) != bool(args.queries):
        return {"status": "unavailable", "reason": "--base and --queries must be provided together"}
    if args.base:
        try:
            return _custom(Path(args.base).expanduser(), Path(args.queries).expanduser(), profile)
        except (UnavailableError, OSError, ValueError) as exc:
            return {"status": "unavailable", "reason": str(exc)}
    try:
        return _synthetic(
            directory,
            profile["expected_n"],
            profile["expected_dim"],
            profile["nq"],
            args.seed,
            "embedding-100k",
        )
    except (OSError, ValueError, MemoryError) as exc:
        return {
            "status": "unavailable",
            "reason": f"could not prepare synthetic dataset: {type(exc).__name__}: {exc}",
        }


def load_dataset(dataset: dict[str, Any]):
    import numpy as np

    if dataset["loader"] == "hdf5":
        try:
            import h5py
        except ImportError as exc:
            raise UnavailableError(f"h5py is required: {exc}") from exc
        with h5py.File(dataset["path"], "r") as handle:
            base = np.asarray(handle[dataset["base_key"]][: dataset["n"]], dtype=np.float32, order="C")
            queries = np.asarray(handle[dataset["query_key"]][: dataset["nq"]], dtype=np.float32, order="C")
        return base, queries
    base = np.load(dataset["base_path"], mmap_mode="r", allow_pickle=False)
    queries = np.load(dataset["query_path"], mmap_mode="r", allow_pickle=False)
    return (
        np.asarray(base[: dataset["n"]], dtype=np.float32, order="C"),
        np.asarray(queries[: dataset["nq"]], dtype=np.float32, order="C"),
    )


def query_batches(queries: Any, batch_size: int) -> Iterable[Any]:
    for start in range(0, len(queries), batch_size):
        yield queries[start : start + batch_size]


def measure_search(search_batch: Callable[[Any], tuple[Any, Any]], queries: Any, batch_size: int,
                   warmups: int, repeats: int) -> tuple[dict[str, Any], Any, Any]:
    import numpy as np

    for _ in range(warmups):
        for batch in query_batches(queries, batch_size):
            search_batch(batch)
    pass_ms: list[float] = []
    qps: list[float] = []
    batch_ms: list[float] = []
    per_query_ms: list[float] = []
    final_ids: list[Any] = []
    final_distances: list[Any] = []
    for repeat in range(repeats):
        ids_parts: list[Any] = []
        distance_parts: list[Any] = []
        pass_started = time.perf_counter_ns()
        for batch in query_batches(queries, batch_size):
            batch_started = time.perf_counter_ns()
            ids, distances = search_batch(batch)
            elapsed = (time.perf_counter_ns() - batch_started) / 1_000_000
            batch_ms.append(elapsed)
            per_query_ms.append(elapsed / max(len(batch), 1))
            if repeat == repeats - 1:
                ids_parts.append(np.asarray(ids))
                distance_parts.append(np.asarray(distances))
        elapsed = (time.perf_counter_ns() - pass_started) / 1_000_000
        pass_ms.append(elapsed)
        qps.append(len(queries) * 1_000 / max(elapsed, 1e-12))
        if repeat == repeats - 1:
            final_ids, final_distances = ids_parts, distance_parts
    measurement = {
        "warmup_passes": warmups,
        "measured_passes": repeats,
        "query_count_per_pass": len(queries),
        "query_batch_size": batch_size,
        "pass_latency_ms": {"samples": pass_ms, "summary": summarize_samples(pass_ms)},
        "batch_latency_ms": {"summary": summarize_samples(batch_ms)},
        "amortized_per_query_latency_ms": {"summary": summarize_samples(per_query_ms)},
        "qps": {"samples": qps, "summary": summarize_samples(qps)},
    }
    return measurement, np.concatenate(final_ids), np.concatenate(final_distances)


def measure_energy(reader: Callable[[], int | None] | None, search_once: Callable[[], None], nq: int,
                   min_seconds: float, max_passes: int) -> dict[str, Any]:
    if reader is None:
        return {"status": "unavailable", "reason": "no package energy reader"}
    try:
        e0 = reader()
    except Exception as exc:
        return {"status": "unavailable", "reason": f"energy read failed: {exc}"}
    if e0 is None:
        return {"status": "unavailable", "reason": "package energy telemetry is unsupported"}
    passes, started = 0, time.perf_counter()
    try:
        while passes < max_passes and (passes == 0 or time.perf_counter() - started < min_seconds):
            search_once()
            passes += 1
        e1 = reader()
    except Exception as exc:
        return {"status": "failed", "reason": f"energy measurement failed: {exc}"}
    if e1 is None or e1 <= e0 or not passes:
        return {"status": "unavailable", "reason": "package counter was absent or non-monotonic"}
    delta = int(e1 - e0)
    return {
        "status": "success",
        "scope": "whole_package_cpu_igpu_uncore_not_isolated_device",
        "microjoules_per_query": delta / float(passes * max(nq, 1)),
        "energy_delta_uj": delta,
        "passes": passes,
        "queries": passes * nq,
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
    }


def point_parameters(profile: dict[str, Any], algorithm: str) -> list[dict[str, Any]]:
    if algorithm == "brute":
        return [{"query_batch_size": profile["query_batch_size"]}]
    if algorithm in ("ivf-flat", "ivf-pq"):
        return [{"nprobe": value, "query_batch_size": profile["query_batch_size"]} for value in profile["nprobes"]]
    if algorithm == "cagra":
        return list(profile["cagra_points"])
    if algorithm == "hnsw":
        return list(profile["hnsw_points"])
    raise ValueError(f"unsupported algorithm: {algorithm}")


def point_failure_reason(points: Sequence[dict[str, Any]]) -> str | None:
    failures: list[str] = []
    for point in points:
        if point.get("status") == "success":
            continue
        parameters = ",".join(f"{key}={value}" for key, value in point.get("parameters", {}).items())
        reason = point.get("reason") or point.get("error_type") or point.get("status", "failed")
        failures.append(f"[{parameters or 'default'}] {reason}")
    return "; ".join(failures[:4]) if failures else None


def recall_result(neighbors: Any, truth: Any | None, k: int) -> dict[str, Any]:
    if truth is None:
        return {"status": "unavailable", "reason": "exact ground truth was not produced"}
    return {"status": "success", "metric": f"recall@{k}", "value": duplicate_safe_recall(neighbors, truth, k)}


def policy_contract(algorithm: str, policy: str, devices: Sequence[int]) -> dict[str, Any]:
    labels = sorted({DEVICE_LABELS.get(int(device), f"UNKNOWN_{device}") for device in devices})
    base = {
        "requested": POLICY_LABELS[policy],
        "reported_last_primitive_devices": labels,
        "attribution_scope": "last_primitive_only_not_complete_pipeline",
    }
    if policy == "hetero":
        return {**base, "status": "unwired_equals_auto", "conforming": False, "reason": "Backlog B6: HETERO routes as AUTO."}
    if policy == "auto":
        return {**base, "status": "not_forced", "conforming": None}
    if algorithm == "ivf-pq" and policy == "gpu":
        return {
            **base,
            "status": "known_mixed_or_unverified",
            "conforming": False,
            "reason": "FORCE_GPU may run IVF-PQ ADC on NPU; last_device cannot prove a pure-GPU pipeline.",
        }
    expected = {"cpu": "CPU", "npu": "NPU", "gpu": "GPU"}[policy]
    conforming = bool(labels) and all(label == expected for label in labels)
    return {
        **base,
        "status": "conforming_last_primitive" if conforming else "mismatch_or_unattributed",
        "conforming": conforming,
        "reason": None if conforming else f"last primitive was not consistently {expected}",
    }


def completion_issues(profile_name: str, dataset: dict[str, Any], ground_truth: dict[str, Any],
                      lanes: Sequence[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    full_profile = profile_name != "smoke"
    if dataset.get("status") != "success":
        issues.append(f"dataset: {dataset.get('reason', dataset.get('status'))}")
    if ground_truth.get("status") != "success":
        issues.append(f"exact ground truth: {ground_truth.get('reason', ground_truth.get('status'))}")
    elif ground_truth.get("exact") is not True:
        issues.append("ground truth was not marked exact")
    seen_ids: set[str] = set()
    for lane in lanes:
        lane_id = str(lane.get("id"))
        if lane_id in seen_ids:
            issues.append(f"duplicate lane id: {lane_id}")
        seen_ids.add(lane_id)
        if lane.get("status") not in LANE_STATUSES:
            issues.append(f"{lane_id}: invalid status {lane.get('status')}")
            continue
        if (
            lane.get("status") == "skipped"
            and lane.get("expected_skip")
            and (not full_profile or not lane.get("blocking_skip"))
        ):
            continue
        if lane.get("status") != "success":
            issues.append(f"{lane_id}: {lane.get('status')} - {lane.get('reason', 'no reason')}")
            continue
        if lane.get("build", {}).get("status") != "success":
            issues.append(f"{lane_id}: successful lane has no successful build record")
        points = lane.get("points", [])
        if not points:
            issues.append(f"{lane_id}: successful lane has no curve points")
        for point in points:
            if point.get("status") != "success":
                issues.append(f"{lane_id} curve point: {point.get('reason', point.get('status'))}")
                continue
            validation = point.get("validation", {})
            if validation.get("status") != "success":
                issues.append(f"{lane_id}: successful point failed output validation")
            measurement = point.get("measurement", {})
            repeat_count = measurement.get("pass_latency_ms", {}).get("summary", {}).get("count", 0)
            qps = measurement.get("qps", {}).get("summary", {}).get("median")
            if repeat_count < 2 or not isinstance(qps, (int, float)) or not math.isfinite(qps) or qps <= 0:
                issues.append(f"{lane_id}: successful point has incomplete/invalid repeat statistics")
            if ground_truth.get("status") == "success" and point.get("recall", {}).get("status") != "success":
                issues.append(f"{lane_id}: successful point has no exact-recall result")
            energy = point.get("energy", {})
            if full_profile and energy.get("status") != "success":
                issues.append(
                    f"{lane_id}: full-profile point has no package-energy result - "
                    f"{energy.get('reason', energy.get('status', 'missing'))}"
                )
            contract = point.get("policy_contract", {})
            if contract.get("conforming") is False and lane.get("policy_key") in ("cpu", "npu", "gpu"):
                issues.append(f"{lane_id}: policy contract {contract.get('status')}")
    return issues


def markdown_escape(value: Any) -> str:
    return "—" if value is None else str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return markdown_escape(value)


def render_markdown(artifact: dict[str, Any]) -> str:
    dataset = artifact["dataset"]
    lines = [
        f"# ovVS benchmark — {artifact['profile']['name']}",
        "",
        f"- Started: `{artifact['started_at']}`",
        f"- Completion: **{artifact['completion']['status']}**",
        f"- Dataset: `{dataset.get('kind', 'unavailable')}`; n={dataset.get('n', '—')}, dim={dataset.get('dim', '—')}, nq={dataset.get('nq', '—')}, k={artifact['profile']['settings']['k']}",
        f"- Exact ground truth: `{artifact['ground_truth'].get('status')}` via `{artifact['ground_truth'].get('method', 'unavailable')}`",
        "- Timings exclude build and warmups. Package energy is whole-package, not isolated device energy.",
        "- `last_device` is final-primitive evidence only. HETERO equals AUTO today; IVF-PQ FORCE_GPU may be mixed NPU+GPU.",
        "",
        "| Lane | Status | Build ms | Curve point | Recall | QPS median | Batch p50 ms | Batch p99 ms | µJ/query | Last primitive | Reason |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for lane in artifact["lanes"]:
        build_ms = lane.get("build", {}).get("elapsed_ms")
        points = lane.get("points") or [None]
        for point in points:
            point = point or {}
            measurement = point.get("measurement", {})
            params = ", ".join(f"{key}={value}" for key, value in point.get("parameters", {}).items()) or "—"
            devices = point.get("policy_contract", {}).get("reported_last_primitive_devices")
            cells = [
                lane["id"],
                point.get("status", lane.get("status")),
                format_number(build_ms),
                params,
                format_number(point.get("recall", {}).get("value"), 4),
                format_number(measurement.get("qps", {}).get("summary", {}).get("median"), 1),
                format_number(measurement.get("batch_latency_ms", {}).get("summary", {}).get("p50")),
                format_number(measurement.get("batch_latency_ms", {}).get("summary", {}).get("p99")),
                format_number(point.get("energy", {}).get("microjoules_per_query"), 1),
                ",".join(devices) if devices else None,
                point.get("reason") or lane.get("reason"),
            ]
            lines.append("| " + " | ".join(markdown_escape(cell) for cell in cells) + " |")
    if artifact["completion"]["issues"]:
        lines.extend(["", "## Incomplete evidence", ""])
        lines.extend(f"- {issue}" for issue in artifact["completion"]["issues"])
    lines.append("")
    return "\n".join(lines)


def default_output_path(profile_name: str, started_at: str) -> Path:
    stamp = started_at.replace("-", "").replace(":", "").replace("T", "-").replace("Z", "")
    return RESULTS_DIR / f"bench-{profile_name}-{stamp}.json"


def write_artifacts(artifact: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown = output.with_suffix(".md")
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(artifact), encoding="utf-8")
    return output, markdown
