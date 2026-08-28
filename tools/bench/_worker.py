"""One-process benchmark lane worker. Invoked only by tools/bench/bench.py."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from _common import (
    DEVICE_LABELS,
    POLICY_LABELS,
    POLICY_VALUES,
    UnavailableError,
    load_dataset,
    load_ovvs,
    measure_energy,
    measure_search,
    ovvs_resource_metadata,
    package_version,
    point_failure_reason,
    point_parameters,
    policy_contract,
    query_batches,
    recall_result,
    utc_now,
    validate_neighbors,
)


def exception_point_status(exc: Exception) -> str:
    return "unavailable" if getattr(exc, "status", None) == 7 else "failed"


def compute_ground_truth(spec: dict[str, Any]) -> dict[str, Any]:
    """Write exact IDs; official HDF5 neighbors are used only for the complete base."""
    import numpy as np

    dataset, profile = spec["dataset"], spec["profile"]
    truth_path = Path(spec["truth_path"])
    started = time.perf_counter()
    if dataset["ground_truth_hint"].get("hdf5_neighbors_eligible"):
        try:
            import h5py
        except ImportError as exc:
            raise UnavailableError(f"h5py is required for HDF5 ground truth: {exc}") from exc
        with h5py.File(dataset["path"], "r") as handle:
            truth = np.asarray(
                handle[dataset["neighbors_key"]][: dataset["nq"], : profile["k"]], dtype=np.int64
            )
        bad = int(np.count_nonzero((truth < 0) | (truth >= dataset["n"])))
        if bad:
            raise RuntimeError(f"official HDF5 neighbors contain {bad} out-of-range IDs")
        np.save(truth_path, truth, allow_pickle=False)
        return {
            "status": "success",
            "method": "hdf5_neighbors",
            "exact": True,
            "shape": list(truth.shape),
            "validation": {
                "complete_base_selected": dataset["n"] == dataset["source_n"],
                "query_prefix_selected": True,
                "invalid_id_count": bad,
            },
            "elapsed_ms": (time.perf_counter() - started) * 1_000,
        }
    try:
        import faiss
    except ImportError as exc:
        raise UnavailableError(
            "exact ground truth unavailable: HDF5 neighbors do not apply to this base selection and "
            f"faiss-cpu is not importable ({exc})"
        ) from exc
    base, queries = load_dataset(dataset)
    build_started = time.perf_counter()
    index = faiss.IndexFlatL2(dataset["dim"])
    index.add(base)
    build_ms = (time.perf_counter() - build_started) * 1_000
    labels: list[Any] = []
    search_started = time.perf_counter()
    for batch in query_batches(queries, profile["query_batch_size"]):
        _, ids = index.search(batch, profile["k"])
        labels.append(ids)
    truth = np.concatenate(labels).astype(np.int64, copy=False)
    search_ms = (time.perf_counter() - search_started) * 1_000
    bad = int(np.count_nonzero((truth < 0) | (truth >= dataset["n"])))
    if bad:
        raise RuntimeError(f"FAISS exact ground truth contains {bad} invalid IDs")
    np.save(truth_path, truth, allow_pickle=False)
    return {
        "status": "success",
        "method": "faiss_index_flat_l2",
        "exact": True,
        "faiss_version": package_version("faiss-cpu") or getattr(faiss, "__version__", None),
        "shape": list(truth.shape),
        "build_ms": build_ms,
        "search_ms": search_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
    }


def _optional_energy_reader(library: str | None) -> tuple[Callable[[], int | None] | None, Any]:
    try:
        module = load_ovvs(library)
        resources = module.Resources()
        if resources.energy_uj() is None:
            resources.close()
            return None, None
        return resources.energy_uj, resources
    except Exception:
        return None, None


def _energy(spec: dict[str, Any], reader: Callable[[], int | None] | None,
            search_once: Callable[[], None], nq: int) -> dict[str, Any]:
    if not spec["energy"]:
        return {"status": "skipped", "reason": "disabled by --no-energy"}
    profile = spec["profile"]
    return measure_energy(reader, search_once, nq, profile["energy_min_seconds"], profile["energy_max_passes"])


def _build_ovvs_index(module: Any, resources: Any, algorithm: str, base: Any,
                      dataset: dict[str, Any], profile: dict[str, Any]):
    dim = dataset["dim"]
    if algorithm == "brute":
        return module.neighbors.brute_force.build(base, dim=dim, resources=resources)
    if algorithm == "ivf-flat":
        return module.neighbors.ivf_flat.build(
            base, dim=dim, nlist=min(profile["nlist"], dataset["n"]), resources=resources
        )
    if algorithm == "ivf-pq":
        if dim % profile["pq_m"]:
            raise RuntimeError(f"dimension {dim} is not divisible by pq_m={profile['pq_m']}")
        return module.neighbors.ivf_pq.build(
            base,
            dim=dim,
            nlist=min(profile["nlist"], dataset["n"]),
            pq_m=profile["pq_m"],
            pq_nbits=profile["pq_nbits"],
            resources=resources,
        )
    if algorithm == "cagra":
        return module.neighbors.cagra.build(
            base,
            dim=dim,
            graph_degree=profile["graph_degree"],
            intermediate_degree=profile["intermediate_degree"],
            resources=resources,
        )
    raise ValueError(f"unsupported ovVS algorithm: {algorithm}")


def _last_device_evidence(resources: Any, build_succeeded: bool) -> dict[str, Any]:
    try:
        code = int(resources.last_device())
    except Exception as exc:
        return {
            "status": "unavailable",
            "code": None,
            "label": None,
            "scope": "final_primitive_only" if build_succeeded else "failed_build_snapshot_unavailable",
            "reason": f"last_device unavailable: {type(exc).__name__}: {exc}",
        }
    evidence = {
        "status": "reported" if build_succeeded else "snapshot_unverified_after_failure",
        "code": code,
        "label": DEVICE_LABELS.get(code, f"UNKNOWN({code})"),
        "scope": "final_primitive_only" if build_succeeded else "resource_snapshot_not_verified_build_attribution",
    }
    if not build_succeeded:
        evidence["reason"] = "build did not complete; last_device may be the resource default or a prior primitive"
    return evidence


def _fallback_telemetry(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    telemetry: dict[str, Any] = {}
    for key in ("npu_compile_fails", "npu_fallbacks"):
        before_value, after_value = before.get(key), after.get(key)
        delta = after_value - before_value if isinstance(before_value, int) and isinstance(after_value, int) else None
        telemetry[key] = {"before": before_value, "after": after_value, "delta": delta}
    return telemetry


def _resource_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | None]:
    deltas: dict[str, int | None] = {}
    for key in sorted(set(before) | set(after)):
        before_value, after_value = before.get(key), after.get(key)
        if type(before_value) is int and type(after_value) is int:
            deltas[key] = after_value - before_value
    return deltas


def _cagra_transfer_evidence(before: Any, after: Any) -> dict[str, Any]:
    keys = ("walks", "direct_walks", "index_upload_calls", "index_upload_bytes")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {
            "status": "unavailable",
            "before": before,
            "after": after,
            "reason": "CAGRA transfer counters are unavailable",
        }
    if not all(type(before.get(key)) is int and type(after.get(key)) is int for key in keys):
        return {
            "status": "unavailable",
            "before": before,
            "after": after,
            "reason": "CAGRA transfer counters are incomplete",
        }
    delta = {key: after[key] - before[key] for key in keys}
    if any(value < 0 for value in delta.values()):
        return {
            "status": "unavailable",
            "before": before,
            "after": after,
            "delta": delta,
            "reason": "CAGRA transfer counters moved backwards",
        }
    walks = delta["walks"]
    conforming = (
        walks > 0
        and delta["direct_walks"] == walks
        and delta["index_upload_calls"] == 0
        and delta["index_upload_bytes"] == 0
    )
    return {
        "status": "success",
        "before": before,
        "after": after,
        "delta": delta,
        "contract": {
            "status": "direct_zero_upload" if conforming else "mismatch_or_upload",
            "conforming": conforming,
        },
        "scope": "warmups_timed_repeats_and_energy_passes",
    }


def _cagra_transfer_snapshot(module: Any, resources: Any) -> Any:
    try:
        return ovvs_resource_metadata(module, resources).get("cagra_transfer_stats")
    except Exception:
        return None


def _build_record(status: str, policy: str, elapsed_ms: float, last_device: dict[str, Any],
                  before: dict[str, Any], after: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    reported_devices = (
        [last_device["code"]]
        if status == "success"
        and last_device.get("status") == "reported"
        and isinstance(last_device.get("code"), int)
        else []
    )
    record = {
        "status": status,
        # v1 compatibility alias; new consumers should use requested_policy/key.
        "policy": POLICY_LABELS[policy],
        "requested_policy": POLICY_LABELS[policy],
        "requested_policy_key": policy,
        "elapsed_ms": elapsed_ms,
        "last_device": last_device,
        "policy_contract": policy_contract("build", policy, reported_devices),
        "resource_before": before,
        "resource_after": after,
        "resource_deltas": _resource_deltas(before, after),
        "fallback_telemetry": _fallback_telemetry(before, after),
        "attribution_caveat": (
            "last_device reports the final build primitive, not the complete build pipeline"
            if status == "success"
            else "build failed; last_device is an unverified resource snapshot, not build attribution"
        ),
    }
    if reason:
        record["reason"] = reason
    return record


def _ovvs_search(index: Any, algorithm: str, point: dict[str, Any], k: int):
    if algorithm == "brute":
        return lambda batch: index.search(batch, k=k)
    if algorithm == "ivf-flat":
        return lambda batch: index.search(batch, k=k, nprobe=point["nprobe"])
    if algorithm == "ivf-pq":
        return lambda batch: index.search(
            batch, k=k, nprobe=point["nprobe"], krefine=point["krefine"]
        )
    if algorithm == "cagra":
        return lambda batch: index.search(
            batch, k=k, itopk_size=point["itopk_size"], search_width=point["search_width"]
        )
    raise ValueError(f"unsupported ovVS algorithm: {algorithm}")


def run_ovvs(spec: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    started_at, started = utc_now(), time.perf_counter()
    module = load_ovvs(spec.get("library"))
    resources, index = module.Resources(), None
    try:
        before = ovvs_resource_metadata(module, resources)
        policy = lane["policy_key"]
        if policy == "npu" and before.get("npu_available") == 0:
            raise UnavailableError("FORCE_NPU requested but the resource probe reports no NPU")
        if policy == "gpu" and before.get("gpu_available") == 0:
            raise UnavailableError("FORCE_GPU requested but the resource probe reports no GPU")
        base, queries = load_dataset(spec["dataset"])
        truth = np.load(spec["truth_path"], allow_pickle=False).tolist() if Path(spec["truth_path"]).exists() else None
        # Every search lane uses the same independently requested construction policy.
        build_policy = spec.get("build_policy", "auto")
        resources.set_policy(POLICY_VALUES[build_policy])
        build_before = ovvs_resource_metadata(module, resources)
        build_started = time.perf_counter()
        try:
            index = _build_ovvs_index(
                module, resources, lane["algorithm"], base, spec["dataset"], spec["profile"]
            )
        except Exception as exc:
            build_ms = (time.perf_counter() - build_started) * 1_000
            build_last_device = _last_device_evidence(resources, False)
            build_after = ovvs_resource_metadata(module, resources)
            build_status = exception_point_status(exc)
            reason = f"index build under {POLICY_LABELS[build_policy]}: {exc}"
            return {
                **lane,
                "status": build_status,
                "reason": reason,
                "started_at": started_at,
                "elapsed_ms": (time.perf_counter() - started) * 1_000,
                "build": _build_record(
                    build_status,
                    build_policy,
                    build_ms,
                    build_last_device,
                    build_before,
                    build_after,
                    reason,
                ),
                "points": [],
                "implementation_metadata": {
                    "ovvs_version": module.version(),
                    "library": os.environ.get("OVVS_LIBRARY"),
                    "resource_before": before,
                    "resource_after": build_after,
                    "routing_caveat": "build and search policies are independent; last_device is final-primitive evidence",
                },
            }
        build_ms = (time.perf_counter() - build_started) * 1_000
        build_last_device = _last_device_evidence(resources, True)
        build_after = ovvs_resource_metadata(module, resources)
        build_record = _build_record(
            "success", build_policy, build_ms, build_last_device, build_before, build_after
        )
        resources.set_policy(POLICY_VALUES[policy])
        points: list[dict[str, Any]] = []
        for configured in point_parameters(spec["profile"], lane["algorithm"]):
            point = dict(configured)
            if lane["algorithm"] == "ivf-pq":
                point["krefine"] = spec["profile"]["krefine"]
            raw_search = _ovvs_search(index, lane["algorithm"], point, spec["profile"]["k"])
            observed: list[int] = []
            transfer_before = (
                _cagra_transfer_snapshot(module, resources)
                if lane["algorithm"] == "cagra"
                else None
            )

            def search(batch: Any):
                result = raw_search(batch)
                try:
                    observed.append(resources.last_device())
                except Exception:
                    pass
                return result

            try:
                measured, ids, distances = measure_search(
                    search,
                    queries,
                    point["query_batch_size"],
                    spec["profile"]["warmups"],
                    spec["profile"]["repeats"],
                )
                validation = validate_neighbors(
                    ids, distances, spec["dataset"]["nq"], spec["profile"]["k"], spec["dataset"]["n"]
                )

                def search_once() -> None:
                    for batch in query_batches(queries, point["query_batch_size"]):
                        search(batch)

                contract = policy_contract(lane["algorithm"], policy, observed)
                status = "success" if validation["status"] == "success" else "failed"
                if contract["status"] == "mismatch_or_unattributed" and policy in ("cpu", "npu", "gpu"):
                    status = "failed"
                energy = _energy(spec, resources.energy_uj, search_once, len(queries))
                transfer = (
                    _cagra_transfer_evidence(
                        transfer_before,
                        _cagra_transfer_snapshot(module, resources),
                    )
                    if lane["algorithm"] == "cagra"
                    else None
                )
                point_result = {
                    "status": status,
                    "parameters": point,
                    "measurement": measured,
                    "recall": recall_result(ids.tolist(), truth, spec["profile"]["k"]),
                    "validation": validation,
                    "energy": energy,
                    "policy_contract": contract,
                    "reason": "; ".join(validation["issues"])
                    or (contract.get("reason") if status == "failed" else None),
                }
                if transfer is not None:
                    point_result["cagra_transfer"] = transfer
                points.append(point_result)
            except Exception as exc:
                failed_point = {
                    "status": exception_point_status(exc),
                    "parameters": point,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
                if lane["algorithm"] == "cagra":
                    failed_point["cagra_transfer"] = _cagra_transfer_evidence(
                        transfer_before,
                        _cagra_transfer_snapshot(module, resources),
                    )
                points.append(failed_point)
        if all(point["status"] == "success" for point in points):
            lane_status = "success"
        elif points and all(point["status"] == "unavailable" for point in points):
            lane_status = "unavailable"
        else:
            lane_status = "failed"
        return {
            **lane,
            "status": lane_status,
            "reason": point_failure_reason(points),
            "started_at": started_at,
            "elapsed_ms": (time.perf_counter() - started) * 1_000,
            "build": build_record,
            "points": points,
            "implementation_metadata": {
                "ovvs_version": module.version(),
                "library": os.environ.get("OVVS_LIBRARY"),
                "resource_before": before,
                "resource_after": ovvs_resource_metadata(module, resources),
                "routing_caveat": "build and search policies are independent; last_device is final-primitive evidence",
            },
        }
    finally:
        if index is not None:
            try:
                index.close()
            except Exception:
                pass
        resources.close()


def _faiss_index(faiss: Any, algorithm: str, base: Any, dim: int, profile: dict[str, Any]):
    if algorithm == "brute":
        index = faiss.IndexFlatL2(dim)
    elif algorithm == "ivf-flat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, profile["nlist"])
        index.train(base)
    elif algorithm == "ivf-pq":
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dim), dim, profile["nlist"], profile["pq_m"], profile["pq_nbits"]
        )
        index.train(base)
    else:
        raise ValueError(f"unsupported FAISS algorithm: {algorithm}")
    index.add(base)
    return index


def run_faiss(spec: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    try:
        import faiss
    except ImportError as exc:
        raise UnavailableError(f"faiss-cpu is not importable: {exc}") from exc
    started_at, started = utc_now(), time.perf_counter()
    base, queries = load_dataset(spec["dataset"])
    truth = np.load(spec["truth_path"], allow_pickle=False).tolist() if Path(spec["truth_path"]).exists() else None
    build_started = time.perf_counter()
    index = _faiss_index(faiss, lane["algorithm"], base, spec["dataset"]["dim"], spec["profile"])
    build_ms = (time.perf_counter() - build_started) * 1_000
    reader, energy_resources = _optional_energy_reader(spec.get("library"))
    points: list[dict[str, Any]] = []
    try:
        for point in point_parameters(spec["profile"], lane["algorithm"]):
            point = dict(point)
            if lane["algorithm"] in ("ivf-flat", "ivf-pq"):
                index.nprobe = point["nprobe"]

            def search(batch: Any):
                distances, ids = index.search(batch, spec["profile"]["k"])
                return ids, distances

            try:
                measured, ids, distances = measure_search(
                    search, queries, point["query_batch_size"], spec["profile"]["warmups"], spec["profile"]["repeats"]
                )
                validation = validate_neighbors(
                    ids, distances, spec["dataset"]["nq"], spec["profile"]["k"], spec["dataset"]["n"]
                )

                def search_once() -> None:
                    for batch in query_batches(queries, point["query_batch_size"]):
                        search(batch)

                points.append(
                    {
                        "status": validation["status"],
                        "parameters": point,
                        "measurement": measured,
                        "recall": recall_result(ids.tolist(), truth, spec["profile"]["k"]),
                        "validation": validation,
                        "energy": _energy(spec, reader, search_once, len(queries)),
                        "reason": "; ".join(validation["issues"]) or None,
                    }
                )
            except Exception as exc:
                points.append({"status": "failed", "parameters": point, "reason": str(exc), "error_type": type(exc).__name__})
    finally:
        if energy_resources is not None:
            energy_resources.close()
    lane_status = "success" if all(point["status"] == "success" for point in points) else "failed"
    return {
        **lane,
        "status": lane_status,
        "reason": point_failure_reason(points),
        "started_at": started_at,
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
        "build": {"status": "success", "elapsed_ms": build_ms},
        "points": points,
        "implementation_metadata": {
            "faiss_version": package_version("faiss-cpu") or getattr(faiss, "__version__", None),
            "omp_max_threads": int(faiss.omp_get_max_threads()) if hasattr(faiss, "omp_get_max_threads") else None,
        },
    }


def run_hnsw(spec: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    try:
        import hnswlib
    except ImportError as exc:
        raise UnavailableError(f"hnswlib is not importable: {exc}") from exc
    started_at, started = utc_now(), time.perf_counter()
    base, queries = load_dataset(spec["dataset"])
    truth = np.load(spec["truth_path"], allow_pickle=False).tolist() if Path(spec["truth_path"]).exists() else None
    profile = spec["profile"]
    build_started = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=spec["dataset"]["dim"])
    index.init_index(
        max_elements=spec["dataset"]["n"],
        ef_construction=profile["hnsw_ef_construction"],
        M=profile["graph_degree"],
        random_seed=spec["seed"],
    )
    index.add_items(base, np.arange(spec["dataset"]["n"], dtype=np.int64))
    build_ms = (time.perf_counter() - build_started) * 1_000
    reader, energy_resources = _optional_energy_reader(spec.get("library"))
    points: list[dict[str, Any]] = []
    try:
        for point in point_parameters(profile, "hnsw"):
            point = dict(point)
            index.set_ef(max(point["ef"], profile["k"]))

            def search(batch: Any):
                return index.knn_query(batch, k=profile["k"])

            try:
                measured, ids, distances = measure_search(
                    search, queries, point["query_batch_size"], profile["warmups"], profile["repeats"]
                )
                validation = validate_neighbors(ids, distances, spec["dataset"]["nq"], profile["k"], spec["dataset"]["n"])

                def search_once() -> None:
                    for batch in query_batches(queries, point["query_batch_size"]):
                        search(batch)

                points.append(
                    {
                        "status": validation["status"],
                        "parameters": point,
                        "measurement": measured,
                        "recall": recall_result(ids.tolist(), truth, profile["k"]),
                        "validation": validation,
                        "energy": _energy(spec, reader, search_once, len(queries)),
                        "reason": "; ".join(validation["issues"]) or None,
                    }
                )
            except Exception as exc:
                points.append({"status": "failed", "parameters": point, "reason": str(exc), "error_type": type(exc).__name__})
    finally:
        if energy_resources is not None:
            energy_resources.close()
    lane_status = "success" if all(point["status"] == "success" for point in points) else "failed"
    return {
        **lane,
        "status": lane_status,
        "reason": point_failure_reason(points),
        "started_at": started_at,
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
        "build": {"status": "success", "elapsed_ms": build_ms},
        "points": points,
        "implementation_metadata": {
            "hnswlib_version": package_version("hnswlib"),
            "M": profile["graph_degree"],
            "ef_construction": profile["hnsw_ef_construction"],
            "threading": "hnswlib_default",
        },
    }


def _error(lane: dict[str, Any], status: str, exc: Exception) -> dict[str, Any]:
    return {
        **lane,
        "status": status,
        "reason": str(exc),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        },
    }


def run_worker(spec: dict[str, Any], lane_id: str) -> dict[str, Any]:
    if lane_id == "__ground_truth__":
        try:
            return compute_ground_truth(spec)
        except UnavailableError as exc:
            return {"id": lane_id, "status": "unavailable", "reason": str(exc), "error_type": type(exc).__name__}
        except Exception as exc:
            return _error({"id": lane_id}, "failed", exc)
    lane = next(item for item in spec["lanes"] if item["id"] == lane_id)
    try:
        if lane["implementation"] == "ovvs":
            return run_ovvs(spec, lane)
        if lane["implementation"] == "faiss-cpu":
            return run_faiss(spec, lane)
        if lane["implementation"] == "hnswlib":
            return run_hnsw(spec, lane)
        raise ValueError(f"unknown implementation: {lane['implementation']}")
    except UnavailableError as exc:
        return _error(lane, "unavailable", exc)
    except (ImportError, FileNotFoundError, OSError) as exc:
        return _error(lane, "unavailable", exc)
    except Exception as exc:
        return _error(lane, "failed", exc)
