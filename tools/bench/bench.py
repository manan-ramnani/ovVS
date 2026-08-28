#!/usr/bin/env python3
"""Isolated recall/QPS harness for ovVS, FAISS-CPU, and hnswlib.

No arguments run a bounded smoke profile. Full profiles are strict: absent
datasets, exact truth, comparators, device lanes, and resource-gated lanes stay
visible and make the command non-zero unless ``--allow-partial`` is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from _common import (
    ALGORITHM_ORDER,
    CAGRA_RECALL_GATE,
    CAGRA_RECALL_GATE_POINTS,
    CAGRA_SIFT100K_PREFLIGHT,
    LANE_STATUSES,
    POLICY_LABELS,
    POLICY_ORDER,
    PROFILE_DEFAULTS,
    ROOT,
    SCHEMA_VERSION,
    WORKER_PREFIX,
    cagra_recall_gate_result,
    cagra_sift100k_preflight_result,
    completion_issues,
    default_output_path,
    enumerate_lanes,
    package_version,
    parse_selection,
    prepare_dataset,
    resolved_profile,
    utc_now,
    write_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated ovVS recall/QPS matrix; no arguments run the bounded smoke profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="smoke")
    fixed_mode = parser.add_mutually_exclusive_group()
    fixed_mode.add_argument(
        "--gate-only",
        choices=(CAGRA_RECALL_GATE,),
        help="run one noncanonical SIFT1M quality checkpoint instead of a full curve",
    )
    fixed_mode.add_argument(
        "--preflight-only",
        choices=(CAGRA_SIFT100K_PREFLIGHT,),
        help="run one noncanonical 100K SIFT runtime/memory/quality diagnostic",
    )
    parser.add_argument("--algorithms", default="all", help="comma-separated: brute,ivf-flat,ivf-pq,cagra")
    parser.add_argument("--policies", default="all", help="comma-separated: auto,cpu,npu,gpu,hetero")
    parser.add_argument(
        "--build-policy",
        default="auto",
        help="single ovVS construction policy, independent of the search-policy lanes",
    )
    parser.add_argument("--sift", default=str(ROOT / "data" / "sift-128-euclidean.hdf5"))
    parser.add_argument("--base", help="embedding-100k custom base .npy (at least 100000 x 768)")
    parser.add_argument("--queries", help="embedding-100k custom queries .npy (at least 32 x 768)")
    parser.add_argument("--library", help="explicit libovvs/ovvs.dll path forwarded to every lane")
    parser.add_argument("--output", help="JSON artifact path; a Markdown sibling is also written")
    parser.add_argument("--warmups", type=int, help="override warmup passes")
    parser.add_argument("--repeats", type=int, help="override measured passes (minimum 2)")
    parser.add_argument("--timeout-seconds", type=float, help="per-oracle/per-lane timeout")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--hnsw-threads",
        type=int,
        help="explicit hnswlib build/search threads; fixed modes default to os.cpu_count()",
    )
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument(
        "--allow-unscalable-cagra",
        action="store_true",
        help="opt in to not-yet-scale-qualified CAGRA build above n=4096 under the lane timeout",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return zero for incomplete full-profile evidence; statuses and completion remain unchanged",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_spec", help=argparse.SUPPRESS)
    parser.add_argument("--_lane", help=argparse.SUPPRESS)
    return parser


def normalize_gate_configuration(args: argparse.Namespace) -> None:
    """Apply the fixed effective selection for a reproducible gate-only run."""

    if getattr(args, "hnsw_threads", None) is not None and args.hnsw_threads <= 0:
        raise ValueError("hnsw threads must be positive")
    if getattr(args, "gate_only", None) != CAGRA_RECALL_GATE:
        return
    args.profile = "sift1m"
    args.algorithms = "cagra"
    args.policies = "gpu"
    args.build_policy = "auto"
    args.warmups = 1
    args.repeats = 5
    args.seed = 7
    args.no_energy = True
    args.allow_unscalable_cagra = True
    if args.hnsw_threads is None:
        args.hnsw_threads = max(1, os.cpu_count() or 1)


def normalize_preflight_configuration(args: argparse.Namespace) -> None:
    """Apply the fixed selection for the bounded SIFT-prefix preflight."""

    if getattr(args, "hnsw_threads", None) is not None and args.hnsw_threads <= 0:
        raise ValueError("hnsw threads must be positive")
    if getattr(args, "preflight_only", None) != CAGRA_SIFT100K_PREFLIGHT:
        return
    args.profile = "sift-100k"
    args.algorithms = "cagra"
    args.policies = "gpu"
    args.build_policy = "auto"
    args.warmups = 1
    args.repeats = 5
    args.seed = 7
    args.no_energy = True
    args.allow_unscalable_cagra = True
    if args.hnsw_threads is None:
        args.hnsw_threads = max(1, os.cpu_count() or 1)


def gate_exit_code(artifact: dict[str, Any]) -> int:
    return 0 if artifact.get("quality_gate", {}).get("status") == "pass" else 3


def preflight_exit_code(artifact: dict[str, Any]) -> int:
    return 0 if artifact.get("preflight", {}).get("status") == "complete" else 3


def _tail(value: str | bytes | None, limit: int = 4_000) -> str | None:
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def run_child(spec_path: Path, lane_id: str, timeout_seconds: float) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_spec",
        str(spec_path),
        "--_lane",
        lane_id,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": lane_id,
            "status": "timeout",
            "reason": f"lane exceeded {timeout_seconds:g} seconds and its process was terminated",
            "timeout_seconds": timeout_seconds,
            "elapsed_ms": (time.perf_counter() - started) * 1_000,
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(WORKER_PREFIX):
            try:
                payload = json.loads(line[len(WORKER_PREFIX) :])
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        return {
            "id": lane_id,
            "status": "failed",
            "reason": f"worker exited {completed.returncode} without a result (native crash is possible)",
            "returncode": completed.returncode,
            "elapsed_ms": (time.perf_counter() - started) * 1_000,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    payload["worker_returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr_tail"] = _tail(completed.stderr)
    if payload.get("status") not in LANE_STATUSES:
        payload["status"] = "failed"
        payload["reason"] = "worker emitted an unsupported status"
    return payload


def git_metadata() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("short_status", ["git", "status", "--short"]),
    ):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
            values[key] = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            values[key] = None
    values["dirty"] = bool(values.pop("short_status", ""))
    return values


def _artifact(
    args: argparse.Namespace,
    profile: dict[str, Any],
    algorithms: list[str],
    policies: list[str],
    dataset: dict[str, Any],
    ground_truth: dict[str, Any],
    lanes: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    issues = completion_issues(args.profile, dataset, ground_truth, lanes)
    gate_only = getattr(args, "gate_only", None)
    preflight_only = getattr(args, "preflight_only", None)
    quality_gate = None
    preflight = None
    if gate_only == CAGRA_RECALL_GATE:
        quality_gate = cagra_recall_gate_result(
            dataset, ground_truth, profile, lanes, getattr(args, "hnsw_threads", None)
        )
        issues.append(
            "gate-only cagra-recall is a noncanonical two-point quality checkpoint; it does not "
            "satisfy B1 full recall-QPS curves or package-energy evidence"
        )
    if preflight_only == CAGRA_SIFT100K_PREFLIGHT:
        preflight = cagra_sift100k_preflight_result(
            dataset, ground_truth, profile, lanes, getattr(args, "hnsw_threads", None)
        )
    if args.profile == "embedding-100k" and dataset.get("kind") == "synthetic":
        issues.append(
            "embedding-100k uses the provisional synthetic workload; this exercises the B1 harness but "
            "does not satisfy the real 768-d corpus requirement in B20"
        )
    if args.profile == "sift-100k":
        issues.append(
            "sift-100k is a noncanonical prefix preflight for resource, build, and quality diagnosis; "
            "it does not satisfy the full SIFT1M B1/B2 gate"
        )
    artifact_dataset = json.loads(json.dumps(dataset))
    if artifact_dataset.get("kind") == "synthetic":
        artifact_dataset.pop("base_path", None)
        artifact_dataset.pop("query_path", None)
        artifact_dataset["runtime_storage"] = "ephemeral_files_deleted_after_run"
    counts = {status: sum(lane.get("status") == status for lane in lanes) for status in sorted(LANE_STATUSES)}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": utc_now(),
        "profile": {"name": args.profile, "settings": profile},
        "selection": {
            "gate_only": gate_only,
            "preflight_only": preflight_only,
            "algorithms": algorithms,
            "build_policy": POLICY_LABELS[getattr(args, "build_policy", "auto")],
            "build_policy_key": getattr(args, "build_policy", "auto"),
            "policies": [POLICY_LABELS[value] for value in policies],
            "point_selection": (
                {key: [value] for key, value in CAGRA_RECALL_GATE_POINTS.items()}
                if gate_only == CAGRA_RECALL_GATE or preflight_only == CAGRA_SIFT100K_PREFLIGHT
                else None
            ),
            "hnswlib_threads": getattr(args, "hnsw_threads", None),
            "seed": getattr(args, "seed", 7),
            "energy": not getattr(args, "no_energy", False),
        },
        "dataset": artifact_dataset,
        "ground_truth": ground_truth,
        "lanes": lanes,
        "completion": {
            "status": "complete" if not issues else "partial",
            "issues": issues,
            "lane_counts": counts,
            "full_profile_strict": (
                args.profile not in ("smoke", "sift-100k")
                and not getattr(args, "allow_partial", False)
                and gate_only is None
                and preflight_only is None
            ),
        },
        "metadata": {
            "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "git": git_metadata(),
            "dependencies": {
                "numpy": package_version("numpy"),
                "h5py": package_version("h5py"),
                "faiss-cpu": package_version("faiss-cpu"),
                "hnswlib": package_version("hnswlib"),
            },
            "caveats": [
                "Package energy covers the package, iGPU, and uncore; it is not isolated device energy.",
                "ovVS last_device reports the final primitive, not a per-stage route trace.",
                "ovVS construction policy is independent of each lane's search policy.",
                "HETERO currently equals AUTO (backlog B6).",
                "IVF-PQ FORCE_GPU is a visible expected skip because ADC has no iGPU backend and fails closed.",
                "Synthetic 100k x 768 is provisional and does not close real-corpus backlog B20.",
                "SIFT-100k is a noncanonical prefix preflight and cannot close the SIFT1M quality gate.",
                *(
                    [
                        "Gate-only CAGRA recall reports timing but decides only on matched-point recall; "
                        "it is never full B1 evidence."
                    ]
                    if gate_only == CAGRA_RECALL_GATE
                    else []
                ),
            ],
        },
    }
    if quality_gate is not None:
        artifact["selection"]["canonical"] = False
        artifact["completion"]["canonical_b1_evidence"] = False
        artifact["quality_gate"] = quality_gate
    if args.profile == "sift-100k":
        artifact["selection"]["canonical"] = False
        artifact["completion"]["canonical_b1_evidence"] = False
    if preflight is not None:
        artifact["preflight"] = preflight
    return artifact


def orchestrate(args: argparse.Namespace) -> int:
    try:
        normalize_gate_configuration(args)
        normalize_preflight_configuration(args)
        algorithms = parse_selection(args.algorithms, ALGORITHM_ORDER, "algorithm")
        policies = parse_selection(args.policies, POLICY_ORDER, "policy")
        build_policies = parse_selection(args.build_policy, POLICY_ORDER, "build policy")
        if len(build_policies) != 1:
            raise ValueError("exactly one build policy must be selected")
        build_policy = build_policies[0]
        args.build_policy = build_policy
        profile = resolved_profile(args.profile, args.warmups, args.repeats, args.timeout_seconds)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    started_at = utc_now()
    full_profile = args.profile != "smoke"
    with tempfile.TemporaryDirectory(prefix="ovvs-bench-") as temporary:
        directory = Path(temporary)
        dataset = prepare_dataset(args, profile, directory)
        lanes = enumerate_lanes(
            algorithms,
            policies,
            int(dataset.get("n", profile["expected_n"])),
            full_profile,
            args.allow_unscalable_cagra,
        )
        spec = {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "profile_name": args.profile,
            "dataset": dataset,
            "truth_path": str(directory / "exact-ground-truth.npy"),
            "lanes": [lane.as_dict() for lane in lanes],
            "build_policy": build_policy,
            "library": args.library,
            "energy": not args.no_energy,
            "seed": args.seed,
            "gate_only": getattr(args, "gate_only", None),
            "preflight_only": getattr(args, "preflight_only", None),
            "point_selection": (
                {key: [value] for key, value in CAGRA_RECALL_GATE_POINTS.items()}
                if (
                    getattr(args, "gate_only", None) == CAGRA_RECALL_GATE
                    or getattr(args, "preflight_only", None) == CAGRA_SIFT100K_PREFLIGHT
                )
                else {}
            ),
            "hnsw_threads": getattr(args, "hnsw_threads", None),
        }
        spec_path = directory / "run-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        if dataset.get("status") == "success":
            print("[ground-truth] starting exact oracle", file=sys.stderr, flush=True)
            ground_truth = run_child(spec_path, "__ground_truth__", profile["timeout_seconds"])
        else:
            ground_truth = {
                "status": "unavailable",
                "reason": f"dataset unavailable: {dataset.get('reason', 'unknown reason')}",
            }
        results: list[dict[str, Any]] = []
        for lane in lanes:
            definition = lane.as_dict()
            if dataset.get("status") != "success":
                result = {
                    **definition,
                    "status": "skipped",
                    "reason": f"dataset unavailable: {dataset.get('reason', 'unknown reason')}",
                    "expected_skip": False,
                }
            elif lane.skip_reason:
                result = {**definition, "status": "skipped", "reason": lane.skip_reason}
            else:
                print(f"[{lane.id}] starting", file=sys.stderr, flush=True)
                result = run_child(spec_path, lane.id, profile["timeout_seconds"])
                for key, value in definition.items():
                    result.setdefault(key, value)
            results.append(result)
            print(f"[{lane.id}] {result['status']}", file=sys.stderr, flush=True)
        artifact = _artifact(args, profile, algorithms, policies, dataset, ground_truth, results, started_at)
        output = Path(args.output) if args.output else default_output_path(args.profile, started_at)
        try:
            json_path, markdown_path = write_artifacts(artifact, output)
        except (OSError, ValueError) as exc:
            print(f"failed to write artifacts: {exc}", file=sys.stderr)
            return 2
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Completion: {artifact['completion']['status']} {artifact['completion']['lane_counts']}")
    if artifact["completion"]["issues"]:
        print(f"Incomplete evidence: {len(artifact['completion']['issues'])} issue(s)")
    if getattr(args, "gate_only", None) == CAGRA_RECALL_GATE:
        gate = artifact["quality_gate"]
        print(
            f"Quality gate: {gate['status']} "
            f"(hnswlib-CAGRA recall={gate['recall']['hnswlib_minus_cagra']})"
        )
        return gate_exit_code(artifact)
    if getattr(args, "preflight_only", None) == CAGRA_SIFT100K_PREFLIGHT:
        preflight = artifact["preflight"]
        print(
            f"Preflight: {preflight['status']} "
            f"(hnswlib-CAGRA recall={preflight['recall']['hnswlib_minus_cagra']})"
        )
        return preflight_exit_code(artifact)
    if full_profile and artifact["completion"]["issues"] and not args.allow_partial:
        return 3
    return 0


def worker_main(spec_path: Path, lane_id: str) -> int:
    from _worker import process_peak_rss, run_worker

    result = run_worker(json.loads(spec_path.read_text(encoding="utf-8")), lane_id)
    result["process_memory"] = process_peak_rss()
    print(WORKER_PREFIX + json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("success", "unavailable") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._worker:
        if not args._spec or not args._lane:
            parser.error("worker mode requires --_spec and --_lane")
        return worker_main(Path(args._spec), args._lane)
    if args.list_profiles:
        for name, settings in PROFILE_DEFAULTS.items():
            print(f"{name}: {settings['description']}")
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
