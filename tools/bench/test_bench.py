from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import bench
import _common
import _worker
from _worker import (
    _cagra_build_stats_evidence,
    _cagra_transfer_evidence,
    _ivfpq_search_stats_evidence,
    exception_point_status,
    process_peak_rss,
)
from _common import (
    ALGORITHM_ORDER,
    CAGRA_RECALL_GATE_POINTS,
    CAGRA_SIFT100K_PREFLIGHT,
    POLICY_ORDER,
    ROOT,
    SIFT_SHA256,
    cagra_recall_gate_result,
    cagra_sift100k_preflight_result,
    completion_issues,
    default_output_path,
    duplicate_safe_recall,
    enumerate_lanes,
    measure_energy,
    ovvs_resource_metadata,
    parse_selection,
    percentile,
    point_parameters,
    point_failure_reason,
    policy_contract,
    prepare_dataset,
    resolved_profile,
    sha256_file,
    summarize_samples,
    validate_neighbors,
)


def successful_lane(lane_id: str = "faiss-cpu.brute") -> dict:
    return {
        "id": lane_id,
        "status": "success",
        "mandatory": True,
        "build": {"status": "success", "elapsed_ms": 1.0},
        "points": [
            {
                "status": "success",
                "validation": {"status": "success"},
                "recall": {"status": "success", "value": 1.0},
                "energy": {"status": "success", "microjoules_per_query": 10.0},
                "measurement": {
                    "pass_latency_ms": {"summary": {"count": 3}},
                    "qps": {"summary": {"median": 123.0}},
                },
            }
        ],
    }


def successful_gpu_cagra_lane(transfer: dict | None) -> dict:
    lane = successful_lane("ovvs.cagra.gpu")
    lane.update(implementation="ovvs", algorithm="cagra", policy_key="gpu", policy="FORCE_GPU")
    lane["build"] = {
        "status": "success",
        "policy": "AUTO",
        "requested_policy": "AUTO",
        "requested_policy_key": "auto",
        "build_algo": "nndescent",
        "elapsed_ms": 1.0,
        "last_device": {"status": "reported", "code": 1, "label": "CPU"},
        "policy_contract": {"status": "not_forced", "conforming": None},
        "cagra_build_stats": successful_cagra_build_evidence(1_000_000),
    }
    if transfer is not None:
        lane["points"][0]["cagra_transfer"] = transfer
    return lane


def cagra_build_snapshot(**updates: int) -> dict[str, int]:
    snapshot = {
        "abi_version": 1,
        "struct_size": 256,
        **{key: 0 for key in _worker._CAGRA_BUILD_STATS_COUNTERS},
    }
    snapshot.update(updates)
    return snapshot


def successful_cagra_build_after(
    *,
    n: int = 100_000,
    dim: int = 128,
    graph_degree: int = 16,
    intermediate_degree: int = 32,
    build_algo: str = "nndescent",
    final_device: str = "gpu",
) -> dict[str, int]:
    initializer = {
        "nndescent": "nndescent_initializer_calls",
        "ivf_pq": "ivfpq_initializer_calls",
        "iterative": "iterative_initializer_calls",
    }[build_algo]
    final = {
        "cpu": "initializer_final_cpu_calls",
        "gpu": "initializer_final_gpu_calls",
        "npu": "initializer_final_npu_calls",
    }[final_device]
    values = {
        "successful_calls": 1,
        "rows": n,
        "dataset_copy_bytes": n * dim * 4,
        "initializer_graph_payload_bytes": n * intermediate_degree * 4,
        "published_graph_copy_bytes": n * graph_degree * 4,
        initializer: 1,
        final: 1,
        "total_wall_ns": 100,
        "dataset_copy_ns": 10,
        "initializer_ns": 40,
        "optimizer_prune_merge_ns": 30,
        "index_materialize_ns": 10,
    }
    if build_algo == "nndescent" and final_device == "gpu":
        values.update(
            nndescent_gpu_iterations=6,
            nndescent_gpu_converged_calls=1,
            nndescent_gpu_instrumented_calls=1,
            nndescent_gpu_allocation_calls=4,
            nndescent_gpu_allocation_bytes=4096,
            nndescent_gpu_d2h_calls=2,
            nndescent_gpu_d2h_bytes=16,
            nndescent_gpu_kernel_launches=12,
            nndescent_gpu_submission_calls=20,
            nndescent_gpu_wait_calls=6,
            nndescent_gpu_peak_owned_bytes_max=2048,
        )
    return cagra_build_snapshot(**values)


def successful_cagra_build_evidence(n: int) -> dict:
    return _cagra_build_stats_evidence(
        cagra_build_snapshot(),
        successful_cagra_build_after(n=n),
        build_succeeded=True,
        n=n,
        dim=128,
        graph_degree=16,
        intermediate_degree=32,
        build_algo="nndescent",
        build_policy="auto",
    )


def gate_dataset() -> dict:
    return {
        "status": "success",
        "kind": "sift1m",
        "n": 1_000_000,
        "source_n": 1_000_000,
        "dim": 128,
        "nq": 1_000,
        "source_nq": 10_000,
        "metric": "squared_l2",
        "source": {
            "checksum_valid": True,
            "sha256": SIFT_SHA256,
            "expected_sha256": SIFT_SHA256,
        },
        "ground_truth_hint": {"hdf5_neighbors_eligible": True},
    }


def gate_truth() -> dict:
    return {
        "status": "success",
        "method": "hdf5_neighbors",
        "exact": True,
        "shape": [1_000, 10],
        "validation": {
            "complete_base_selected": True,
            "query_prefix_selected": True,
            "id_count": 10_000,
            "invalid_id_count": 0,
            "duplicate_row_count": 0,
        },
    }


def valid_loaded_library_identity() -> dict:
    return {
        "status": "success",
        "resolved_path": "C:/Personal/ioVS/build-icpx/bin/ovvs.dll",
        "sha256": "a" * 64,
        "size_bytes": 1_048_576,
        "mtime_ns": 1_000_000_000,
        "requested_path": "C:/Personal/ioVS/build-icpx/bin/ovvs.dll",
        "matches_requested_path": True,
    }


def gate_lanes(cagra_recall: float = 0.88, hnsw_recall: float = 0.90,
               walks: int = 192, threads: int = 20) -> list[dict]:
    transfer = _cagra_transfer_evidence(
        {"walks": 0, "direct_walks": 0, "index_upload_calls": 0, "index_upload_bytes": 0},
        {
            "walks": walks,
            "direct_walks": walks,
            "index_upload_calls": 0,
            "index_upload_bytes": 0,
        },
    )
    cagra = successful_gpu_cagra_lane(transfer)
    cagra["implementation_metadata"] = {
        "loaded_library": valid_loaded_library_identity(),
    }
    measurement = {
        "warmup_passes": 1,
        "measured_passes": 5,
        "query_count_per_pass": 1_000,
        "query_batch_size": 32,
        "pass_latency_ms": {"samples": [1.0] * 5, "summary": {"count": 5}},
        "qps": {"samples": [123.0] * 5, "summary": {"count": 5, "median": 123.0}},
    }
    cagra["points"][0].update(
        parameters=CAGRA_RECALL_GATE_POINTS["cagra"],
        recall={"status": "success", "value": cagra_recall},
        measurement=json.loads(json.dumps(measurement)),
        policy_contract={
            "requested": "FORCE_GPU",
            "conforming": True,
            "reported_last_primitive_devices": ["GPU"],
        },
    )
    hnsw = successful_lane("hnswlib.hnsw")
    hnsw.update(implementation="hnswlib", algorithm="hnsw")
    hnsw["points"][0].update(
        parameters=CAGRA_RECALL_GATE_POINTS["hnsw"],
        recall={"status": "success", "value": hnsw_recall},
        measurement=json.loads(json.dumps(measurement)),
    )
    hnsw["implementation_metadata"] = {
        "M": 16,
        "ef_construction": 200,
        "random_seed": 7,
        "threading": "explicit",
        "num_threads": threads,
    }
    return [cagra, hnsw]


def preflight_dataset() -> dict:
    return {
        "status": "success",
        "kind": "sift_prefix",
        "n": 100_000,
        "source_n": 1_000_000,
        "dim": 128,
        "nq": 1_000,
        "source_nq": 10_000,
        "metric": "squared_l2",
        "source": {
            "checksum_valid": True,
            "sha256": SIFT_SHA256,
            "expected_sha256": SIFT_SHA256,
        },
        "ground_truth_hint": {
            "method": "faiss_index_flat_l2",
            "hdf5_neighbors_eligible": False,
        },
    }


def preflight_truth() -> dict:
    return {
        "status": "success",
        "method": "faiss_index_flat_l2",
        "exact": True,
        "shape": [1_000, 10],
        "validation": {
            "selected_base_count": 100_000,
            "source_base_count": 1_000_000,
            "id_count": 10_000,
            "invalid_id_count": 0,
            "duplicate_row_count": 0,
        },
    }


def preflight_lanes(cagra_recall: float = 0.40, hnsw_recall: float = 0.88,
                    walks: int = 192, threads: int = 20) -> list[dict]:
    lanes = gate_lanes(cagra_recall, hnsw_recall, walks, threads)
    measurement = {
        "warmup_passes": 1,
        "measured_passes": 5,
        "query_count_per_pass": 1_000,
        "query_batch_size": 32,
        "pass_latency_ms": {"samples": [1.0] * 5, "summary": {"count": 5}},
        "qps": {"samples": [100.0] * 5, "summary": {"count": 5, "median": 100.0}},
    }
    for lane in lanes:
        lane["elapsed_ms"] = 2.0
        lane["process_memory"] = {
            "status": "success",
            "peak_rss_bytes": 64 * 1024 * 1024,
            "source": "test",
            "scope": "isolated_worker_process_including_dataset_runtime_and_index",
        }
        lane["points"][0]["measurement"] = json.loads(json.dumps(measurement))
    lanes[0]["points"][0]["device_attribution"] = {
        "search_calls": walks,
        "successful_observations": walks,
        "failures": 0,
    }
    lanes[0]["build"]["cagra_build_stats"] = successful_cagra_build_evidence(100_000)
    return lanes


class PureHelperTests(unittest.TestCase):
    def test_duplicate_results_do_not_inflate_recall(self) -> None:
        self.assertEqual(duplicate_safe_recall([[1, 1]], [[1, 2]], 2), 0.5)
        self.assertEqual(duplicate_safe_recall([[1, 2]], [[1, 1]], 2), 1.0)

    def test_percentiles_and_summary_are_deterministic(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        summary = summarize_samples([1, 2, 3])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["p95"], 2.9)

    def test_loaded_ovvs_library_identity_hashes_actual_binary_and_matches_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            library = Path(raw) / "ovvs.dll"
            library.write_bytes(b"ovvs-test-binary")
            module = SimpleNamespace(_lib=SimpleNamespace(_name=str(library)))
            identity = _common.ovvs_loaded_library_identity(module, str(library))
            self.assertEqual(identity["status"], "success")
            self.assertEqual(
                identity["sha256"],
                hashlib.sha256(b"ovvs-test-binary").hexdigest(),
            )
            self.assertEqual(identity["size_bytes"], 16)
            self.assertTrue(identity["matches_requested_path"])

            other = Path(raw) / "other.dll"
            other.write_bytes(b"other")
            mismatch = _common.ovvs_loaded_library_identity(module, str(other))
            self.assertEqual(mismatch["status"], "invalid")
            self.assertFalse(mismatch["matches_requested_path"])

    def test_neighbor_validation_checks_duplicates_range_and_finiteness(self) -> None:
        result = validate_neighbors(
            np.asarray([[1, 1], [0, 9]]),
            np.asarray([[0.1, 0.2], [0.3, np.inf]]),
            nq=2,
            k=2,
            n=4,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["duplicate_row_count"], 1)
        self.assertEqual(result["invalid_id_count"], 1)
        self.assertEqual(result["nonfinite_distance_count"], 1)

    def test_selection_preserves_canonical_order_and_aliases(self) -> None:
        self.assertEqual(parse_selection("gpu,force_cpu,gpu", POLICY_ORDER, "policy"), ["cpu", "gpu"])
        self.assertEqual(parse_selection("all", ALGORITHM_ORDER, "algorithm"), list(ALGORITHM_ORDER))
        with self.assertRaises(ValueError):
            parse_selection("cuda", POLICY_ORDER, "policy")
        with self.assertRaises(ValueError):
            parse_selection(" , ", POLICY_ORDER, "policy")

    def test_gate_point_selector_returns_exactly_one_matched_point(self) -> None:
        profile = resolved_profile("sift1m")
        self.assertEqual(point_parameters(profile, "cagra"), profile["cagra_points"])
        self.assertEqual(point_parameters(profile, "hnsw"), profile["hnsw_points"])
        self.assertEqual(
            point_parameters(profile, "cagra", [CAGRA_RECALL_GATE_POINTS["cagra"]]),
            [CAGRA_RECALL_GATE_POINTS["cagra"]],
        )
        with self.assertRaises(ValueError):
            point_parameters(
                profile,
                "cagra",
                [{"itopk_size": 31, "search_width": 1, "query_batch_size": 32}],
            )

    def test_sift1m_has_targeted_full_thread_throughput_points(self) -> None:
        profile = resolved_profile("sift1m")
        for batch_size in (256, 1024):
            self.assertIn(
                {
                    "itopk_size": 64,
                    "search_width": 2,
                    "query_batch_size": batch_size,
                    "purpose": "b1_throughput",
                },
                profile["cagra_points"],
            )
            self.assertIn(
                {"ef": 64, "query_batch_size": batch_size, "purpose": "b1_throughput"},
                profile["hnsw_points"],
            )

    def test_explicit_hnsw_module_requires_matching_provenance_argument(self) -> None:
        args = bench.build_parser().parse_args(["--hnsw-module", "hnswlib.pyd"])
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            bench.normalize_hnsw_configuration(args)

    def test_hnsw_effective_thread_inference_exposes_binding_batch_threshold(self) -> None:
        batch32 = _worker._hnsw_inferred_search_threads(1_000, 32, 20, "0.8.0")
        batch256 = _worker._hnsw_inferred_search_threads(1_000, 256, 20, "0.8.0")
        batch1024 = _worker._hnsw_inferred_search_threads(1_000, 1024, 20, "0.8.0")
        self.assertEqual(batch32["effective_threads_histogram"], {"1": 32})
        self.assertEqual(batch32["batch_rows_histogram"], {"32": 31, "8": 1})
        self.assertEqual(batch256["effective_threads_histogram"], {"20": 4})
        self.assertEqual(batch256["batch_rows_histogram"], {"256": 3, "232": 1})
        self.assertEqual(batch1024["effective_threads_histogram"], {"20": 1})
        self.assertEqual(batch1024["batch_rows_histogram"], {"1000": 1})

    def test_explicit_hnsw_module_is_hash_matched_to_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            module_path = directory / "hnswlib.py"
            module_path.write_text("class Index:\n    pass\n", encoding="utf-8")
            module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
            manifest_path = directory / "hnswlib-build.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ovvs.hnswlib-build.v1",
                        "source": {
                            "repository": "https://github.com/nmslib/hnswlib.git",
                            "tag": "v0.8.0",
                            "version": "0.8.0",
                            "commit": "3f3429661187e4c24a490a0f148fc6bc89042b3d",
                            "tree": "fa3b6d92bd6b0c3a81a7236731c7ad7905c949a1",
                            "tracked_files_clean_after_build": True,
                        },
                        "toolchain": {
                            "compile_options": ["/O2", "/openmp", "/EHsc", "/arch:AVX2"],
                            "injected_environment": {"CL": "/O2 /openmp /arch:AVX2"},
                        },
                        "binary": {
                            "bytes": module_path.stat().st_size,
                            "sha256": module_sha256,
                        },
                        "verification": {
                            "pe_machine": "x64",
                            "import_add_query_smoke": {
                                "status": "success",
                                "finite_distances": True,
                            },
                            "candidate": {
                                "module": {
                                    "bytes": module_path.stat().st_size,
                                    "sha256": module_sha256,
                                },
                                "ymm_operands": 1,
                                "vsubps": 1,
                                "vmulps": 1,
                                "vaddps": 1,
                                "vmovups": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            _, metadata = _worker._load_hnswlib(
                {
                    "hnsw_module": str(module_path),
                    "hnsw_provenance": str(manifest_path),
                }
            )
            self.assertEqual(metadata["hnswlib_version"], "0.8.0")
            self.assertEqual(metadata["module_binary"]["sha256"], module_sha256)
            self.assertEqual(metadata["build_provenance"]["status"], "verified")
            self.assertFalse(metadata["build_provenance"]["active_cycle_attribution"])

            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            document["binary"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _worker._load_hnswlib(
                    {
                        "hnsw_module": str(module_path),
                        "hnsw_provenance": str(manifest_path),
                    }
                )

    def test_hnsw_avx2_provenance_rejects_semantic_contract_violations(self) -> None:
        binary = {"status": "verified", "bytes": 17, "sha256": "a" * 64}
        valid = {
            "schema_version": "ovvs.hnswlib-build.v1",
            "source": {
                "repository": "https://github.com/nmslib/hnswlib.git",
                "tag": "v0.8.0",
                "version": "0.8.0",
                "commit": "3f3429661187e4c24a490a0f148fc6bc89042b3d",
                "tree": "fa3b6d92bd6b0c3a81a7236731c7ad7905c949a1",
                "tracked_files_clean_after_build": True,
            },
            "toolchain": {
                "compile_options": ["/O2", "/openmp", "/EHsc", "/arch:AVX2"],
                "injected_environment": {"CL": "/O2 /openmp /arch:AVX2"},
            },
            "binary": {"bytes": 17, "sha256": "a" * 64},
            "verification": {
                "pe_machine": "x64",
                "import_add_query_smoke": {"status": "success", "finite_distances": True},
                "candidate": {
                    "module": {"bytes": 17, "sha256": "a" * 64},
                    "ymm_operands": 1,
                    "vsubps": 1,
                    "vmulps": 1,
                    "vaddps": 1,
                    "vmovups": 1,
                },
            },
        }
        cases = {
            "source identity": lambda doc: doc["source"].update(commit="0" * 40),
            "clean tracked source": lambda doc: doc["source"].update(
                tracked_files_clean_after_build=False
            ),
            "MSVC AVX2 options": lambda doc: doc["toolchain"].update(
                compile_options=["/O2", "/EHsc"]
            ),
            "finite import/add/query smoke": lambda doc: doc["verification"].update(
                import_add_query_smoke={"status": "success", "finite_distances": False}
            ),
            "AVX/YMM float-distance evidence": lambda doc: doc["verification"][
                "candidate"
            ].update(vmulps=0),
            "bound to the loaded module": lambda doc: doc["verification"][
                "candidate"
            ]["module"].update(sha256="b" * 64),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                document = json.loads(json.dumps(valid))
                mutate(document)
                with self.assertRaisesRegex(ValueError, message):
                    _worker._validate_hnswlib_avx2_provenance(document, binary)

    def test_worker_rejects_explicit_hnsw_module_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            module_path = Path(raw) / "hnswlib.py"
            module_path.write_text("class Index:\n    pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be supplied together"):
                _worker._load_hnswlib({"hnsw_module": str(module_path)})

    def test_invalid_hnsw_provenance_is_rejected_before_module_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            marker = directory / "executed.txt"
            module_path = directory / "hnswlib.py"
            module_path.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
                "class Index:\n    pass\n",
                encoding="utf-8",
            )
            manifest_path = directory / "hnswlib-build.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ovvs.hnswlib-build.v1",
                        "source": {"version": "0.8.0"},
                        "binary": {
                            "bytes": module_path.stat().st_size,
                            "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pinned v0.8.0 source identity"):
                _worker._load_hnswlib(
                    {
                        "hnsw_module": str(module_path),
                        "hnsw_provenance": str(manifest_path),
                    }
                )
            self.assertFalse(marker.exists())

    def test_gate_configuration_normalizes_conflicting_selection(self) -> None:
        args = bench.build_parser().parse_args(
            [
                "--gate-only", "cagra-recall",
                "--profile", "smoke",
                "--algorithms", "brute",
                "--policies", "cpu",
                "--build-policy", "gpu",
                "--cagra-build-algo", "iterative",
                "--warmups", "9",
                "--repeats", "9",
                "--seed", "99",
                "--timeout-seconds", "7200",
                "--include-hnsw-export",
            ]
        )
        with patch.object(bench.os, "cpu_count", return_value=20):
            bench.normalize_gate_configuration(args)
        self.assertEqual(args.profile, "sift1m")
        self.assertEqual(args.algorithms, "cagra")
        self.assertEqual(args.policies, "gpu")
        self.assertEqual(args.build_policy, "auto")
        self.assertEqual(args.cagra_build_algo, "nndescent")
        self.assertEqual((args.warmups, args.repeats, args.seed), (1, 5, 7))
        self.assertEqual(args.timeout_seconds, 7200)
        self.assertEqual(args.hnsw_threads, 20)
        self.assertTrue(args.no_energy)
        self.assertTrue(args.allow_unscalable_cagra)
        self.assertFalse(args.include_hnsw_export)

    def test_non_gate_configuration_preserves_existing_defaults(self) -> None:
        args = bench.build_parser().parse_args([])
        before = vars(args).copy()
        bench.normalize_gate_configuration(args)
        self.assertEqual(vars(args), before)

    def test_cagra_recall_gate_passes_at_absolute_boundary_without_using_qps(self) -> None:
        lanes = gate_lanes(cagra_recall=0.88, hnsw_recall=0.90)
        lanes[0]["points"][0]["measurement"]["qps"]["summary"]["median"] = None
        lanes[1]["points"][0]["measurement"]["qps"]["summary"]["median"] = 1_000_000.0
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["passed"])
        self.assertFalse(result["qps_median"]["used_for_verdict"])
        self.assertIsNone(result["qps_median"]["cagra"])

    def test_cagra_recall_gate_fails_only_above_absolute_boundary(self) -> None:
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"),
            gate_lanes(cagra_recall=0.8799, hnsw_recall=0.90), 20
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["validation"]["status"], "success")

    def test_cagra_recall_gate_rejects_non_nndescent_initializer(self) -> None:
        lanes = gate_lanes()
        lanes[0]["build"]["build_algo"] = "iterative"
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any("AUTO-policy attribution" in issue for issue in result["validation"]["issues"])
        )

    def test_fixed_gate_requires_current_gpu_build_telemetry(self) -> None:
        lanes = gate_lanes()
        del lanes[0]["build"]["cagra_build_stats"]
        missing = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(missing["status"], "invalid")
        self.assertTrue(
            any("telemetry is unavailable" in issue for issue in missing["validation"]["issues"])
        )

        lanes = gate_lanes()
        lanes[0]["build"]["cagra_build_stats"] = _cagra_build_stats_evidence(
            cagra_build_snapshot(),
            successful_cagra_build_after(n=1_000_000, final_device="cpu"),
            build_succeeded=True,
            n=1_000_000,
            dim=128,
            graph_degree=16,
            intermediate_degree=32,
            build_algo="nndescent",
            build_policy="auto",
        )
        cpu = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(cpu["status"], "invalid")
        self.assertTrue(
            any("instrumented GPU" in issue for issue in cpu["validation"]["issues"])
        )

    def test_fixed_gate_requires_loaded_binary_fingerprint_and_explicit_match(self) -> None:
        lanes = gate_lanes()
        del lanes[0]["implementation_metadata"]["loaded_library"]
        missing = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(missing["status"], "invalid")
        self.assertTrue(
            any("gate binary" in issue for issue in missing["validation"]["issues"])
        )

        lanes = gate_lanes()
        identity = lanes[0]["implementation_metadata"]["loaded_library"]
        identity["matches_requested_path"] = None
        unmatched = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(unmatched["status"], "invalid")
        self.assertTrue(
            any("explicit matching --library" in issue for issue in unmatched["validation"]["issues"])
        )

    def test_cagra_recall_gate_rejects_wrong_transfer_count_and_truth(self) -> None:
        truth = gate_truth()
        truth["validation"]["duplicate_row_count"] = 1
        result = cagra_recall_gate_result(
            gate_dataset(), truth, resolved_profile("sift1m"), gate_lanes(walks=191), 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("duplicate" in issue for issue in result["validation"]["issues"]))
        self.assertTrue(any("192/192/0/0" in issue for issue in result["validation"]["issues"]))

    def test_cagra_recall_gate_rejects_duplicate_lanes_and_non_integer_transfer(self) -> None:
        lanes = gate_lanes()
        lanes[0]["points"][0]["cagra_transfer"]["delta"]["index_upload_calls"] = False
        lanes.append(json.loads(json.dumps(lanes[0])))
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("expected exactly" in issue for issue in result["validation"]["issues"]))
        self.assertTrue(any("192/192/0/0" in issue for issue in result["validation"]["issues"]))

    def test_cagra_recall_gate_rejects_wrong_measurement_contract(self) -> None:
        lanes = gate_lanes()
        lanes[1]["points"][0]["measurement"]["query_batch_size"] = 1
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), resolved_profile("sift1m"), lanes, 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("batch-32" in issue for issue in result["validation"]["issues"]))

    def test_cagra_recall_gate_rejects_drifted_build_geometry(self) -> None:
        profile = resolved_profile("sift1m")
        profile["graph_degree"] = 32
        result = cagra_recall_gate_result(
            gate_dataset(), gate_truth(), profile, gate_lanes(), 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("graph_degree" in issue for issue in result["validation"]["issues"]))

    def test_gate_exit_code_uses_quality_status_not_partial_completion(self) -> None:
        self.assertEqual(
            bench.gate_exit_code({"quality_gate": {"status": "pass"}, "completion": {"status": "partial"}}),
            0,
        )
        for status in ("fail", "invalid", "unavailable", None):
            self.assertEqual(bench.gate_exit_code({"quality_gate": {"status": status}}), 3)

    def test_sift_100k_preflight_is_completion_not_recall_verdict(self) -> None:
        result = cagra_sift100k_preflight_result(
            preflight_dataset(), preflight_truth(), resolved_profile("sift-100k"),
            preflight_lanes(cagra_recall=0.40, hnsw_recall=0.88), 20,
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["completed"])
        self.assertFalse(result["recall"]["used_for_verdict"])
        self.assertAlmostEqual(result["recall"]["hnswlib_minus_cagra"], 0.48)
        self.assertEqual(result["expected_cagra_transfer_delta"]["walks"], 192)

    def test_sift_100k_preflight_rejects_missing_memory_and_wrong_transfer(self) -> None:
        lanes = preflight_lanes(walks=191)
        lanes[1]["process_memory"] = {"status": "unavailable"}
        result = cagra_sift100k_preflight_result(
            preflight_dataset(), preflight_truth(), resolved_profile("sift-100k"), lanes, 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("192/192/0/0" in issue for issue in result["validation"]["issues"]))
        self.assertTrue(any("peak RSS" in issue for issue in result["validation"]["issues"]))

    def test_sift_100k_preflight_requires_every_device_attribution(self) -> None:
        lanes = preflight_lanes()
        lanes[0]["points"][0]["device_attribution"] = {
            "search_calls": 192,
            "successful_observations": 191,
            "failures": 1,
        }
        result = cagra_sift100k_preflight_result(
            preflight_dataset(), preflight_truth(), resolved_profile("sift-100k"), lanes, 20
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("all 192 walk calls" in issue
                            for issue in result["validation"]["issues"]))

    def test_sift_100k_preflight_propagates_truth_terminal_status(self) -> None:
        for truth_status in ("timeout", "failed"):
            with self.subTest(truth_status=truth_status):
                truth = preflight_truth()
                truth["status"] = truth_status
                result = cagra_sift100k_preflight_result(
                    preflight_dataset(), truth, resolved_profile("sift-100k"),
                    preflight_lanes(), 20,
                )
                self.assertEqual(result["status"], truth_status)

    def test_preflight_exit_code_requires_complete_contract(self) -> None:
        self.assertEqual(bench.preflight_exit_code({"preflight": {"status": "complete"}}), 0)
        for status in ("invalid", "unavailable", "timeout", "failed", None):
            self.assertEqual(bench.preflight_exit_code({"preflight": {"status": status}}), 3)

    def test_peak_rss_uses_platform_counter(self) -> None:
        result = process_peak_rss()
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["peak_rss_bytes"], 0)

    def test_unix_peak_rss_normalizes_kib_to_bytes(self) -> None:
        fake_resource = SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _scope: SimpleNamespace(ru_maxrss=123),
        )
        with (
            patch.object(_worker.sys, "platform", "linux"),
            patch.dict(sys.modules, {"resource": fake_resource}),
        ):
            result = process_peak_rss()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["peak_rss_bytes"], 123 * 1024)

    def test_hash_helper_is_content_stable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.bin"
            path.write_bytes(b"ovvs\n")
            self.assertEqual(
                sha256_file(path),
                "ffcc64117ff38cfb09d58830acea7f1a155fee4cb85a7526c0f53a142939ec5c",
            )

    def test_unchanged_energy_counter_is_unavailable_not_zero_energy(self) -> None:
        readings = iter([100, 100])
        result = measure_energy(lambda: next(readings), lambda: None, nq=32, min_seconds=0, max_passes=1)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("microjoules_per_query", result)

    def test_lane_failure_reason_aggregates_curve_point_context(self) -> None:
        reason = point_failure_reason(
            [
                {"status": "success", "parameters": {"nprobe": 2}},
                {"status": "failed", "parameters": {"nprobe": 8}, "reason": "non-finite distances"},
            ]
        )
        self.assertEqual(reason, "[nprobe=8] non-finite distances")

    def test_device_unavailable_status_is_not_mislabeled_failed(self) -> None:
        exc = RuntimeError("forced device unavailable")
        exc.status = 7
        self.assertEqual(exception_point_status(exc), "unavailable")
        compile_fail = RuntimeError("forced device compile failed")
        compile_fail.status = 5
        self.assertEqual(exception_point_status(compile_fail), "failed")
        self.assertEqual(exception_point_status(RuntimeError("numeric corruption")), "failed")

    def test_cagra_transfer_zero_upload_evidence_completes_force_gpu_point(self) -> None:
        transfer = _cagra_transfer_evidence(
            {"walks": 4, "direct_walks": 4, "index_upload_calls": 0, "index_upload_bytes": 0},
            {"walks": 11, "direct_walks": 11, "index_upload_calls": 0, "index_upload_bytes": 0},
        )
        self.assertTrue(transfer["contract"]["conforming"])
        self.assertEqual(transfer["delta"]["walks"], 7)
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [successful_gpu_cagra_lane(transfer)],
        )
        self.assertEqual(issues, [])

    def test_cagra_transfer_upload_keeps_force_gpu_point_incomplete(self) -> None:
        transfer = _cagra_transfer_evidence(
            {"walks": 1, "direct_walks": 1, "index_upload_calls": 0, "index_upload_bytes": 0},
            {"walks": 4, "direct_walks": 3, "index_upload_calls": 1, "index_upload_bytes": 4096},
        )
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [successful_gpu_cagra_lane(transfer)],
        )
        self.assertTrue(any("uploaded the index" in issue for issue in issues))
        self.assertTrue(any("direct-walk delta" in issue for issue in issues))

    def test_cagra_transfer_missing_evidence_keeps_force_gpu_point_incomplete(self) -> None:
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [successful_gpu_cagra_lane(None)],
        )
        self.assertTrue(any("no transfer-counter evidence" in issue for issue in issues))

    def test_resource_metadata_includes_cagra_transfer_counters(self) -> None:
        counters = {"walks": 3, "direct_walks": 2, "index_upload_calls": 1, "index_upload_bytes": 64}
        resources = SimpleNamespace(_h=None, cagra_transfer_stats=lambda: counters)
        module = SimpleNamespace(_lib=SimpleNamespace(), sycl_enabled=lambda: True)
        self.assertEqual(ovvs_resource_metadata(module, resources)["cagra_transfer_stats"], counters)

    def test_resource_metadata_ivfpq_stats_is_optional_for_older_bindings(self) -> None:
        resources = SimpleNamespace(_h=None, cagra_transfer_stats=lambda: None)
        module = SimpleNamespace(_lib=SimpleNamespace(), sycl_enabled=lambda: True)
        self.assertIsNone(ovvs_resource_metadata(module, resources)["ivfpq_search_stats"])

    def test_ivfpq_stats_delta_is_complete_and_nonnegative(self) -> None:
        before = {
            "abi_version": 1,
            "struct_size": 200,
            **{key: 0 for key in _worker._IVFPQ_SEARCH_STATS_COUNTERS},
        }
        after = dict(before)
        after.update(successful_calls=3, blocks=6, queries=96, total_wall_ns=1234)
        evidence = _ivfpq_search_stats_evidence(before, after)
        self.assertEqual(evidence["status"], "success")
        self.assertEqual(evidence["delta"]["successful_calls"], 3)
        self.assertEqual(evidence["delta"]["queries"], 96)
        self.assertTrue(all(value >= 0 for value in evidence["delta"].values()))

    def test_ivfpq_phase_stats_rejects_backwards_counters(self) -> None:
        before = {
            "abi_version": 1,
            "struct_size": 200,
            **{key: 5 for key in _worker._IVFPQ_SEARCH_STATS_COUNTERS},
        }
        after = dict(before)
        after["gpu_wait_calls"] = 4
        evidence = _ivfpq_search_stats_evidence(before, after, "timed")
        self.assertEqual(evidence["status"], "invalid")
        self.assertEqual(evidence["scope"], "timed")
        self.assertEqual(evidence["delta"]["gpu_wait_calls"], -1)
        self.assertIn("moved backwards", evidence["reason"])

    def test_ivfpq_timed_scope_rejects_wrong_call_and_query_counts(self) -> None:
        snapshot = {
            "abi_version": 1,
            "struct_size": 200,
            **{key: 0 for key in _worker._IVFPQ_SEARCH_STATS_COUNTERS},
        }
        snapshots = {
            scope: {"before": dict(snapshot), "after": dict(snapshot)}
            for scope in ("warmup", "timed", "energy", "complete_point")
        }
        scopes = _worker._ivfpq_search_stats_scopes(
            snapshots,
            timed_expected_calls=5,
            timed_expected_queries=160,
        )
        self.assertEqual(scopes["timed"]["status"], "invalid")
        self.assertEqual(
            scopes["timed"]["expected"],
            {"successful_calls": 5, "queries": 160},
        )
        self.assertIn("repeat and query counts", scopes["timed"]["reason"])

    def test_ivfpq_timed_scope_rejects_impossible_rows_and_stage_sum(self) -> None:
        before = {
            "abi_version": 1,
            "struct_size": 200,
            **{key: 0 for key in _worker._IVFPQ_SEARCH_STATS_COUNTERS},
        }
        after = dict(before)
        after.update(
            successful_calls=1,
            queries=1,
            candidate_rows=1,
            selected_rows=2,
            total_wall_ns=10,
        )
        snapshots = {
            scope: {"before": dict(before), "after": dict(after)}
            for scope in ("warmup", "timed", "energy", "complete_point")
        }

        scopes = _worker._ivfpq_search_stats_scopes(
            snapshots,
            timed_expected_calls=1,
            timed_expected_queries=1,
        )

        self.assertEqual(scopes["timed"]["status"], "invalid")
        self.assertIn("selected more rows", scopes["timed"]["reason"])
        self.assertIn("do not sum", scopes["timed"]["reason"])


class CagraBuildTelemetryTests(unittest.TestCase):
    @staticmethod
    def evidence(
        before: dict | None,
        after: dict | None,
        *,
        build_succeeded: bool = True,
        build_algo: str = "nndescent",
        n: int = 100_000,
        dim: int = 128,
        graph_degree: int = 16,
        intermediate_degree: int = 32,
        build_policy: str = "auto",
    ) -> dict:
        return _cagra_build_stats_evidence(
            before,
            after,
            build_succeeded=build_succeeded,
            n=n,
            dim=dim,
            graph_degree=graph_degree,
            intermediate_degree=intermediate_degree,
            build_algo=build_algo,
            build_policy=build_policy,
        )

    def test_missing_symbol_is_explicit_and_nonblocking(self) -> None:
        evidence = self.evidence(None, None)
        self.assertEqual(evidence["status"], "unavailable")
        self.assertFalse(evidence["blocking"])
        self.assertIn("unavailable", evidence["reason"])

    def test_ctypes_v1_layout_freezes_every_field_offset(self) -> None:
        class FakeFunction:
            pass

        class FakeLibrary:
            def __init__(self) -> None:
                self.functions: dict[str, FakeFunction] = {}

            def __getattr__(self, name: str) -> FakeFunction:
                return self.functions.setdefault(name, FakeFunction())

        with tempfile.TemporaryDirectory() as raw:
            fake_library = Path(raw) / "ovvs.dll"
            fake_library.touch()
            binding_path = ROOT / "python" / "ovvs" / "__init__.py"
            spec = importlib.util.spec_from_file_location("_ovvs_abi_offset_test", binding_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            with (
                patch.dict(_common.os.environ, {"OVVS_LIBRARY": str(fake_library)}),
                patch.object(ctypes, "CDLL", return_value=FakeLibrary()),
                patch.object(_common.os, "add_dll_directory", return_value=object(), create=True),
            ):
                spec.loader.exec_module(module)

        structure = module._CagraBuildStatsV1
        self.assertEqual(ctypes.sizeof(structure), 256)
        self.assertEqual(structure.abi_version.offset, 0)
        self.assertEqual(structure.struct_size.offset, 4)
        for index, name in enumerate(module._CAGRA_BUILD_STATS_COUNTERS):
            with self.subTest(field=name):
                self.assertEqual(getattr(structure, name).offset, 8 + 8 * index)

    def test_incompatible_and_backward_snapshots_are_invalid(self) -> None:
        before = cagra_build_snapshot()
        incompatible = successful_cagra_build_after()
        incompatible["struct_size"] = 248
        evidence = self.evidence(before, incompatible)
        self.assertEqual(evidence["status"], "invalid")
        self.assertIn("incompatible", evidence["reason"])

        before = cagra_build_snapshot(successful_calls=2)
        after = cagra_build_snapshot(successful_calls=1)
        evidence = self.evidence(before, after)
        self.assertEqual(evidence["status"], "invalid")
        self.assertIn("backwards", evidence["reason"])

    def test_successful_large_auto_gpu_delta_validates_with_zero_h2d(self) -> None:
        evidence = self.evidence(
            cagra_build_snapshot(),
            successful_cagra_build_after(),
        )
        self.assertEqual(evidence["status"], "success")
        self.assertTrue(evidence["gpu_structural_validation_required"])
        self.assertEqual(evidence["delta"]["nndescent_gpu_h2d_calls"], 0)
        self.assertEqual(evidence["delta"]["nndescent_gpu_h2d_bytes"], 0)
        self.assertLessEqual(evidence["stage_sum_ns"], evidence["delta"]["total_wall_ns"])
        self.assertEqual(
            evidence["total_wall_ns_scope"],
            "inner build wall through persistent graph materialization, excluding telemetry merge "
            "and caller-handle publication",
        )

        inconsistent = successful_cagra_build_after()
        inconsistent["nndescent_gpu_h2d_calls"] = 1
        rejected = self.evidence(cagra_build_snapshot(), inconsistent)
        self.assertEqual(rejected["status"], "invalid")
        self.assertIn("H2D calls/bytes", rejected["reason"])

    def test_wrong_geometry_and_bytes_are_invalid(self) -> None:
        for key, value in (
            ("rows", 99_999),
            ("dataset_copy_bytes", 1),
            ("initializer_graph_payload_bytes", 2),
            ("published_graph_copy_bytes", 3),
        ):
            with self.subTest(key=key):
                after = successful_cagra_build_after()
                after[key] = value
                evidence = self.evidence(cagra_build_snapshot(), after)
                self.assertEqual(evidence["status"], "invalid")
                self.assertIn(key, evidence["reason"])

    def test_wrong_initializer_algorithm_is_invalid(self) -> None:
        after = successful_cagra_build_after()
        after["nndescent_initializer_calls"] = 0
        after["ivfpq_initializer_calls"] = 1
        evidence = self.evidence(cagra_build_snapshot(), after)
        self.assertEqual(evidence["status"], "invalid")
        self.assertIn("does not match requested nndescent", evidence["reason"])

    def test_stage_overflow_is_invalid(self) -> None:
        after = successful_cagra_build_after()
        after["initializer_ns"] = 101
        evidence = self.evidence(cagra_build_snapshot(), after)
        self.assertEqual(evidence["status"], "invalid")
        self.assertIn("stage time exceeds", evidence["reason"])

    def test_failed_build_must_leave_success_only_counters_unchanged(self) -> None:
        before = cagra_build_snapshot()
        unchanged = self.evidence(before, dict(before), build_succeeded=False)
        self.assertEqual(unchanged["status"], "success")
        self.assertEqual(unchanged["outcome"], "failed_build_unchanged")

        changed_after = dict(before)
        changed_after["rows"] = 1
        changed = self.evidence(before, changed_after, build_succeeded=False)
        self.assertEqual(changed["status"], "invalid")
        self.assertIn("success-only", changed["reason"])

    def test_cpu_initializer_cannot_publish_gpu_instrumentation(self) -> None:
        after = successful_cagra_build_after(final_device="cpu")
        after["nndescent_gpu_kernel_launches"] = 1
        evidence = self.evidence(cagra_build_snapshot(), after)
        self.assertEqual(evidence["status"], "invalid")
        self.assertIn("published GPU", evidence["reason"])


class CliAndLaneSemanticsTests(unittest.TestCase):
    def test_no_arg_profile_is_bounded(self) -> None:
        args = bench.build_parser().parse_args([])
        profile = resolved_profile(args.profile)
        self.assertEqual(args.profile, "smoke")
        self.assertLessEqual(profile["expected_n"], 2_000)
        self.assertLessEqual(profile["nq"], 32)
        self.assertLessEqual(profile["timeout_seconds"], 30)
        self.assertGreaterEqual(profile["repeats"], 2)
        self.assertEqual(args.build_policy, "auto")
        self.assertEqual(args.cagra_build_algo, "nndescent")

    def test_sift_100k_profile_is_bounded_preflight(self) -> None:
        profile = resolved_profile("sift-100k")
        self.assertEqual((profile["expected_n"], profile["expected_dim"], profile["nq"]),
                         (100_000, 128, 1_000))
        self.assertEqual(profile["graph_degree"], 16)
        self.assertEqual(profile["intermediate_degree"], 32)
        self.assertEqual(len(profile["cagra_points"]), 2)
        self.assertLess(profile["timeout_seconds"], resolved_profile("sift1m")["timeout_seconds"])

    def test_sift_100k_preflight_normalizes_conflicting_selection(self) -> None:
        args = bench.build_parser().parse_args(
            [
                "--preflight-only", CAGRA_SIFT100K_PREFLIGHT,
                "--profile", "smoke",
                "--algorithms", "brute",
                "--policies", "cpu",
                "--build-policy", "gpu",
                "--cagra-build-algo", "ivf_pq",
                "--warmups", "9",
                "--repeats", "9",
                "--seed", "99",
                "--timeout-seconds", "1200",
                "--include-hnsw-export",
            ]
        )
        with patch.object(bench.os, "cpu_count", return_value=20):
            bench.normalize_preflight_configuration(args)
        self.assertEqual(args.profile, "sift-100k")
        self.assertEqual(args.algorithms, "cagra")
        self.assertEqual(args.policies, "gpu")
        self.assertEqual(args.build_policy, "auto")
        self.assertEqual(args.cagra_build_algo, "nndescent")
        self.assertEqual((args.warmups, args.repeats, args.seed), (1, 5, 7))
        self.assertEqual(args.timeout_seconds, 1200)
        self.assertEqual(args.hnsw_threads, 20)
        self.assertTrue(args.no_energy)
        self.assertFalse(args.include_hnsw_export)
        self.assertTrue(args.allow_unscalable_cagra)

    def test_sift_100k_missing_fixture_has_no_synthetic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            result = prepare_dataset(
                SimpleNamespace(
                    profile="sift-100k",
                    base=None,
                    queries=None,
                    sift=str(directory / "missing.hdf5"),
                    seed=7,
                ),
                resolved_profile("sift-100k"),
                directory,
            )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("SIFT HDF5 is absent", result["reason"])

    def test_default_artifact_is_in_ignored_out_tree(self) -> None:
        path = default_output_path("smoke", "2026-08-28T01:02:03Z")
        self.assertEqual(path.parent, ROOT / "out" / "bench")

    def test_full_matrix_keeps_b5_and_npu_cagra_skips_distinct(self) -> None:
        lanes = enumerate_lanes(["cagra"], ["auto", "npu", "gpu"], 1_000_000, True, False)
        by_id = {lane.id: lane for lane in lanes}
        self.assertTrue(by_id["ovvs.cagra.auto"].blocking_skip)
        self.assertTrue(by_id["ovvs.cagra.auto"].expected_skip)
        self.assertFalse(by_id["ovvs.cagra.npu"].blocking_skip)
        self.assertIn("hnswlib.hnsw", by_id)

    def test_hnsw_export_lane_is_explicit_and_nonmandatory(self) -> None:
        ordinary = enumerate_lanes(["cagra"], ["auto"], 2_000, False, True)
        self.assertNotIn("hnswlib.ovvs-cagra", {lane.id for lane in ordinary})
        selected = enumerate_lanes(
            ["cagra"], ["auto"], 2_000, False, True, include_hnsw_export=True
        )
        lane = {item.id: item for item in selected}["hnswlib.ovvs-cagra"]
        self.assertEqual(lane.implementation, "hnswlib-export")
        self.assertFalse(lane.mandatory)

    def test_hnsw_export_lane_obeys_cagra_scale_gate(self) -> None:
        gated = enumerate_lanes(
            ["cagra"], ["gpu"], 1_000_000, True, False, include_hnsw_export=True
        )
        gated_lane = {item.id: item for item in gated}["hnswlib.ovvs-cagra"]
        self.assertTrue(gated_lane.expected_skip)
        self.assertTrue(gated_lane.blocking_skip)
        self.assertIn("--allow-unscalable-cagra", gated_lane.skip_reason)

        opted_in = enumerate_lanes(
            ["cagra"], ["gpu"], 1_000_000, True, True, include_hnsw_export=True
        )
        opted_in_lane = {item.id: item for item in opted_in}["hnswlib.ovvs-cagra"]
        self.assertIsNone(opted_in_lane.skip_reason)
        self.assertFalse(opted_in_lane.expected_skip)
        self.assertFalse(opted_in_lane.blocking_skip)

    def test_hnsw_export_completion_requires_nested_build_attribution(self) -> None:
        missing = successful_lane("hnswlib.ovvs-cagra")
        missing.update(implementation="hnswlib-export", algorithm="hnsw")
        missing_issues = completion_issues(
            "smoke",
            {"status": "success", "n": 3, "dim": 2},
            {"status": "success", "exact": True},
            [missing],
        )
        self.assertTrue(any("no ovVS CAGRA build record" in issue for issue in missing_issues))

        mismatched = successful_lane("hnswlib.ovvs-cagra")
        mismatched.update(implementation="hnswlib-export", algorithm="hnsw")
        mismatched["build"]["ovvs_cagra"] = {
            "status": "success",
            "requested_policy": "FORCE_GPU",
            "requested_policy_key": "gpu",
            "elapsed_ms": 1.0,
            "last_device": {"status": "reported", "code": 1, "label": "CPU"},
            "policy_contract": {"status": "mismatch", "conforming": False},
        }
        mismatch_issues = completion_issues(
            "smoke",
            {"status": "success", "n": 3, "dim": 2},
            {"status": "success", "exact": True},
            [mismatched],
        )
        self.assertTrue(any("forced build policy contract mismatch" in issue
                            for issue in mismatch_issues))

    def test_full_b5_skip_makes_completion_partial(self) -> None:
        lane = enumerate_lanes(["cagra"], ["auto"], 1_000_000, True, False)[0].as_dict()
        lane.update(status="skipped", reason=lane["skip_reason"])
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("ovvs.cagra.auto" in issue for issue in issues))

    def test_cagra_force_npu_expected_skip_can_be_nonblocking(self) -> None:
        lane = enumerate_lanes(["cagra"], ["npu"], 2_000, True, False)[0].as_dict()
        lane.update(status="skipped", reason=lane["skip_reason"])
        self.assertEqual(
            completion_issues(
                "sift1m",
                {"status": "success"},
                {"status": "success", "exact": True},
                [lane],
            ),
            [],
        )

    def test_ivf_pq_force_gpu_uses_strict_final_primitive_contract(self) -> None:
        conforming = policy_contract("ivf-pq", "gpu", [3])
        mismatch = policy_contract("ivf-pq", "gpu", [2])
        self.assertTrue(conforming["conforming"])
        self.assertEqual(conforming["status"], "conforming_last_primitive")
        self.assertFalse(mismatch["conforming"])
        self.assertEqual(mismatch["status"], "mismatch_or_unattributed")

    def test_ivf_pq_force_gpu_is_runnable_and_failures_block_completion(self) -> None:
        lanes = enumerate_lanes(["ivf-pq"], ["auto", "gpu"], 1_000_000, True, False)
        by_id = {lane.id: lane for lane in lanes}
        gpu = by_id["ovvs.ivf-pq.gpu"]
        self.assertFalse(gpu.expected_skip)
        self.assertFalse(gpu.blocking_skip)
        self.assertIsNone(gpu.skip_reason)
        self.assertIn("faiss-cpu.ivf-pq", by_id)
        failed = gpu.as_dict()
        failed.update(status="failed", reason="forced GPU compile failed")
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [failed],
        )
        self.assertTrue(any("ovvs.ivf-pq.gpu" in issue for issue in issues))

    def test_successful_ovvs_lane_requires_complete_build_attribution(self) -> None:
        lane = successful_lane("ovvs.cagra.auto")
        lane["implementation"] = "ovvs"
        lane["build"] = {
            "status": "success",
            "policy": "AUTO",
            "requested_policy": "AUTO",
            "requested_policy_key": "auto",
            "elapsed_ms": float("nan"),
            "last_device": {"status": "reported", "code": 1, "label": "CPU"},
            "policy_contract": {"status": "not_forced", "conforming": None},
        }
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("invalid elapsed time" in issue for issue in issues))
        lane["build"]["elapsed_ms"] = 1.0
        lane["build"].pop("requested_policy_key")
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("valid requested policy" in issue for issue in issues))
        lane["build"]["requested_policy_key"] = "auto"
        self.assertEqual(
            completion_issues(
                "sift1m",
                {"status": "success"},
                {"status": "success", "exact": True},
                [lane],
            ),
            [],
        )

    def test_successful_ovvs_lane_requires_build_device_and_forced_contract(self) -> None:
        lane = successful_lane("ovvs.brute.gpu")
        lane["implementation"] = "ovvs"
        lane["build"] = {
            "status": "success",
            "policy": "FORCE_GPU",
            "requested_policy": "FORCE_GPU",
            "requested_policy_key": "gpu",
            "elapsed_ms": 1.0,
            "last_device": {"status": "reported", "code": 1, "label": "CPU"},
            "policy_contract": {"status": "mismatch_or_unattributed", "conforming": False},
        }
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("forced build policy contract" in issue for issue in issues))

        lane["build"]["last_device"] = {"status": "unavailable", "code": None, "label": None}
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("physical final-primitive attribution" in issue for issue in issues))

    def test_smoke_failure_is_partial_evidence_even_though_cli_may_return_zero(self) -> None:
        issues = completion_issues(
            "smoke",
            {"status": "success"},
            {"status": "unavailable", "reason": "faiss absent"},
            [{"id": "faiss-cpu.brute", "status": "unavailable", "reason": "faiss absent"}],
        )
        self.assertTrue(any("ground truth" in issue for issue in issues))
        self.assertTrue(any("faiss-cpu.brute" in issue for issue in issues))

    def test_successful_artifact_point_must_have_repeat_stats(self) -> None:
        lane = successful_lane()
        lane["points"][0]["measurement"]["pass_latency_ms"]["summary"]["count"] = 1
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("repeat statistics" in issue for issue in issues))

    def test_complete_full_comparator_record_has_no_issues(self) -> None:
        self.assertEqual(
            completion_issues(
                "sift1m",
                {"status": "success"},
                {"status": "success", "exact": True},
                [successful_lane()],
            ),
            [],
        )

    def test_full_profile_rejects_incomplete_cagra_and_hnsw_point_multisets(self) -> None:
        profile = resolved_profile("sift1m")
        for lane_id, implementation, algorithm, expected in (
            ("ovvs.cagra.auto", "ovvs", "cagra", profile["cagra_points"]),
            ("hnswlib.hnsw", "hnswlib", "hnsw", profile["hnsw_points"]),
        ):
            with self.subTest(algorithm=algorithm):
                lane = successful_lane(lane_id)
                lane.update(implementation=implementation, algorithm=algorithm)
                template = lane["points"][0]
                lane["points"] = [
                    {**json.loads(json.dumps(template)), "parameters": point}
                    for point in expected
                ]
                complete = completion_issues(
                    "sift1m",
                    {"status": "success"},
                    {"status": "success", "exact": True},
                    [lane],
                    require_full_point_matrix=True,
                )
                self.assertFalse(any("point multiset" in issue for issue in complete))

                lane["points"] = lane["points"][:-1]
                incomplete = completion_issues(
                    "sift1m",
                    {"status": "success"},
                    {"status": "success", "exact": True},
                    [lane],
                    require_full_point_matrix=True,
                )
                self.assertTrue(any("incomplete configured" in issue for issue in incomplete))

                lane["points"].append(json.loads(json.dumps(lane["points"][0])))
                duplicated = completion_issues(
                    "sift1m",
                    {"status": "success"},
                    {"status": "success", "exact": True},
                    [lane],
                    require_full_point_matrix=True,
                )
                self.assertTrue(any("incomplete configured" in issue for issue in duplicated))

    def test_full_profile_requires_package_energy_evidence(self) -> None:
        lane = successful_lane()
        lane["points"][0]["energy"] = {"status": "skipped", "reason": "disabled by --no-energy"}
        issues = completion_issues(
            "sift1m",
            {"status": "success"},
            {"status": "success", "exact": True},
            [lane],
        )
        self.assertTrue(any("package-energy" in issue for issue in issues))

    def test_malformed_custom_dataset_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            base = directory / "base.npy"
            queries = directory / "queries.npy"
            base.write_bytes(b"not a numpy file")
            queries.write_bytes(b"not a numpy file")
            result = prepare_dataset(
                SimpleNamespace(
                    profile="embedding-100k",
                    base=str(base),
                    queries=str(queries),
                    sift="unused",
                    seed=7,
                ),
                resolved_profile("embedding-100k"),
                directory,
            )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("could not read custom NumPy inputs", result["reason"])

    def test_synthetic_embedding_is_provisional_and_temp_paths_are_not_published(self) -> None:
        artifact = bench._artifact(
            SimpleNamespace(profile="embedding-100k", allow_partial=False),
            resolved_profile("embedding-100k"),
            ["brute"],
            ["cpu"],
            {
                "status": "success",
                "kind": "synthetic",
                "base_path": "C:/temp/deleted-base.npy",
                "query_path": "C:/temp/deleted-query.npy",
                "source": {"generation_spec_sha256": "abc"},
            },
            {"status": "success", "exact": True},
            [successful_lane("ovvs.brute.cpu")],
            "2026-08-28T00:00:00Z",
        )
        self.assertEqual(artifact["completion"]["status"], "partial")
        self.assertTrue(any("provisional synthetic" in issue for issue in artifact["completion"]["issues"]))
        self.assertNotIn("base_path", artifact["dataset"])
        self.assertNotIn("query_path", artifact["dataset"])
        self.assertEqual(artifact["dataset"]["runtime_storage"], "ephemeral_files_deleted_after_run")
        self.assertNotIn("CAGRA initializer", _common.render_markdown(artifact))

    def test_sift_100k_artifact_is_always_noncanonical(self) -> None:
        artifact = bench._artifact(
            SimpleNamespace(
                profile="sift-100k",
                gate_only=None,
                hnsw_threads=20,
                build_policy="auto",
                no_energy=False,
                allow_partial=True,
            ),
            resolved_profile("sift-100k"),
            ["cagra"],
            ["gpu"],
            {"status": "success", "kind": "sift_prefix", "n": 100_000, "dim": 128, "nq": 1_000},
            {"status": "success", "exact": True},
            [successful_lane()],
            "2026-08-28T00:00:00Z",
        )
        self.assertEqual(artifact["completion"]["status"], "partial")
        self.assertTrue(any("noncanonical prefix preflight" in issue
                            for issue in artifact["completion"]["issues"]))
        self.assertFalse(artifact["selection"]["canonical"])
        self.assertFalse(artifact["completion"]["canonical_b1_evidence"])
        self.assertFalse(artifact["completion"]["full_profile_strict"])

    def test_completed_sift_100k_preflight_artifact_stays_partial(self) -> None:
        artifact = bench._artifact(
            SimpleNamespace(
                profile="sift-100k",
                gate_only=None,
                preflight_only=CAGRA_SIFT100K_PREFLIGHT,
                hnsw_threads=20,
                build_policy="auto",
                no_energy=True,
                allow_partial=False,
                seed=7,
            ),
            resolved_profile("sift-100k"),
            ["cagra"],
            ["gpu"],
            preflight_dataset(),
            preflight_truth(),
            preflight_lanes(),
            "2026-08-28T00:00:00Z",
        )
        self.assertEqual(artifact["preflight"]["status"], "complete")
        self.assertEqual(artifact["completion"]["status"], "partial")
        self.assertNotIn("quality_gate", artifact)
        markdown = _common.render_markdown(artifact)
        self.assertIn("Recall (reported, not a verdict)", markdown)
        self.assertIn("full SIFT1M quality gate remains required", markdown)

    def test_gate_artifact_is_partial_noncanonical_even_when_gate_passes(self) -> None:
        artifact = bench._artifact(
            SimpleNamespace(
                profile="sift1m",
                gate_only="cagra-recall",
                hnsw_threads=20,
                build_policy="auto",
                allow_partial=False,
            ),
            resolved_profile("sift1m"),
            ["cagra"],
            ["gpu"],
            gate_dataset(),
            gate_truth(),
            gate_lanes(),
            "2026-08-28T00:00:00Z",
        )
        self.assertEqual(artifact["quality_gate"]["status"], "pass")
        self.assertEqual(artifact["completion"]["status"], "partial")
        self.assertFalse(artifact["completion"]["canonical_b1_evidence"])
        self.assertFalse(artifact["completion"]["full_profile_strict"])
        self.assertFalse(artifact["selection"]["canonical"])
        markdown = _common.render_markdown(artifact)
        self.assertIn("QPS median (reported, not part of verdict)", markdown)

    def test_official_ground_truth_rejects_duplicate_rows(self) -> None:
        class FakeFile:
            def __enter__(self):
                return {"neighbors": np.asarray([[0, 0], [1, 2]], dtype=np.int64)}

            def __exit__(self, *_args):
                return False

        fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: FakeFile())
        with tempfile.TemporaryDirectory() as raw:
            spec = {
                "dataset": {
                    "ground_truth_hint": {"hdf5_neighbors_eligible": True},
                    "path": "fixture.hdf5",
                    "neighbors_key": "neighbors",
                    "nq": 2,
                    "n": 4,
                    "source_n": 4,
                },
                "profile": {"k": 2},
                "truth_path": str(Path(raw) / "truth.npy"),
            }
            with patch.dict(sys.modules, {"h5py": fake_h5py}):
                with self.assertRaisesRegex(RuntimeError, "duplicate IDs"):
                    _worker.compute_ground_truth(spec)

    def test_official_ground_truth_rejects_wrong_shape_and_invalid_ids(self) -> None:
        cases = (
            (np.asarray([[0, 1]], dtype=np.int64), "neighbor shape"),
            (np.asarray([[0, 4], [1, 2]], dtype=np.int64), "out-of-range IDs"),
        )
        for neighbors, message in cases:
            with self.subTest(message=message):
                class FakeFile:
                    def __enter__(self):
                        return {"neighbors": neighbors}

                    def __exit__(self, *_args):
                        return False

                fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: FakeFile())
                with tempfile.TemporaryDirectory() as raw:
                    spec = {
                        "dataset": {
                            "ground_truth_hint": {"hdf5_neighbors_eligible": True},
                            "path": "fixture.hdf5",
                            "neighbors_key": "neighbors",
                            "nq": 2,
                            "n": 4,
                            "source_n": 4,
                        },
                        "profile": {"k": 2},
                        "truth_path": str(Path(raw) / "truth.npy"),
                    }
                    with patch.dict(sys.modules, {"h5py": fake_h5py}):
                        with self.assertRaisesRegex(RuntimeError, message):
                            _worker.compute_ground_truth(spec)


class HnswThreadingTests(unittest.TestCase):
    def test_export_graph_stats_validate_header_and_entry_reachability(self) -> None:
        uint = struct.Struct("@I")
        count, m, max_m0 = 3, 2, 4
        data_offset = uint.size + max_m0 * uint.size
        label_offset = data_offset + 2 * struct.calcsize("@f")
        row_bytes = label_offset + struct.calcsize("@N")
        header = b"".join(
            (
                struct.pack(
                    "@6N", 0, count, count, row_bytes, label_offset, data_offset
                ),
                struct.pack("@i", 0),
                struct.pack("@I", 0),
                struct.pack("@3N", m, max_m0, m),
                struct.pack("@d", 1.0),
                struct.pack("@N", 200),
            )
        )
        payload = bytearray(header)
        for node, neighbor in enumerate((1, 2, 0)):
            row = bytearray(row_bytes)
            uint.pack_into(row, 0, 1)
            uint.pack_into(row, uint.size, neighbor)
            row[label_offset : label_offset + struct.calcsize("@N")] = struct.pack("@N", node)
            payload.extend(row)
        payload.extend(uint.pack(0) * count)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.hnsw"
            path.write_bytes(payload)
            stats = _worker._hnsw_export_graph_stats(path)
        self.assertEqual(stats["status"], "success")
        self.assertEqual(stats["max_elements"], count)
        self.assertEqual(stats["entry_reachable_nodes"], count)
        self.assertEqual(stats["entry_reachable_fraction"], 1.0)
        self.assertEqual(stats["dimension"], 2)
        self.assertEqual(stats["native_size_t_bytes"], struct.calcsize("@N"))
        self.assertTrue(stats["labels_match_base_rows"])
        self.assertEqual(stats["deletion_marked_nodes"], 0)
        self.assertEqual(stats["upper_layer_bytes"], 0)
        self.assertEqual(len(stats["sha256"]), 64)
        self.assertEqual(stats["out_degree"], {"min": 1, "max": 1, "mean": 1.0})

        bad_label = bytearray(payload)
        first_label = len(header) + label_offset
        bad_label[first_label : first_label + struct.calcsize("@N")] = struct.pack("@N", 2)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad-label.hnsw"
            path.write_bytes(bad_label)
            with self.assertRaisesRegex(ValueError, "labels must equal base row ids"):
                _worker._hnsw_export_graph_stats(path)

        deleted = bytearray(payload)
        deleted[len(header) + 2] = 1
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "deleted.hnsw"
            path.write_bytes(deleted)
            with self.assertRaisesRegex(ValueError, "deletion/reserved bits"):
                _worker._hnsw_export_graph_stats(path)

        upper_layer = bytearray(payload)
        uint.pack_into(upper_layer, len(upper_layer) - count * uint.size, uint.size)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "upper-layer.hnsw"
            path.write_bytes(upper_layer)
            with self.assertRaisesRegex(ValueError, "nonzero upper-layer block"):
                _worker._hnsw_export_graph_stats(path)

    def test_hybrid_markdown_exposes_topology_timing_and_attribution(self) -> None:
        artifact = {
            "profile": {"name": "smoke", "settings": {"k": 10}},
            "started_at": "2026-08-28T00:00:00Z",
            "completion": {"status": "complete", "issues": []},
            "dataset": {"kind": "synthetic", "n": 3, "dim": 2, "nq": 1},
            "ground_truth": {"status": "success", "method": "exact"},
            "lanes": [
                {
                    "id": "hnswlib.ovvs-cagra",
                    "implementation": "hnswlib-export",
                    "status": "success",
                    "build": {
                        "status": "success",
                        "elapsed_ms": 12.0,
                        "scope": "construction_pipeline_resource_creation_through_loaded_stock_hnswlib_index",
                        "instrumentation_adjusted_ms": 11.5,
                        "production_stage_sum_ms": 11.0,
                        "ovvs_cagra": {
                            "requested_policy": "AUTO",
                            "elapsed_ms": 8.0,
                            "last_device": {"label": "CPU"},
                            "fallback_telemetry": {"npu_fallbacks": {"delta": 0}},
                        },
                        "stages": {"ovvs_cagra_build": {"status": "success", "elapsed_ms": 8.0}},
                        "graph": {
                            "count": 3,
                            "entry_reachable_nodes": 2,
                            "entry_reachable_fraction": 2 / 3,
                            "M": 2,
                            "maxM0": 4,
                            "maxlevel": 0,
                            "dimension": 2,
                            "out_degree": {"min": 1, "mean": 1.0, "max": 1},
                            "file_bytes": 128,
                            "sha256": "abc",
                            "labels_match_base_rows": True,
                            "deletion_marked_nodes": 0,
                            "upper_layer_bytes": 0,
                        },
                    },
                    "points": [],
                    "implementation_metadata": {"construction_caveat": "base-only"},
                }
            ],
        }
        markdown = _common.render_markdown(artifact)
        self.assertIn("## HNSW export diagnostics", markdown)
        self.assertIn("reachable=2/3 (0.6667)", markdown)
        self.assertIn("policy=`AUTO`", markdown)
        self.assertIn("Construction caveat: base-only", markdown)

    def test_explicit_hnsw_threads_apply_to_build_and_search_and_are_recorded(self) -> None:
        class FakeIndex:
            def __init__(self, **_kwargs) -> None:
                self.set_threads: list[int] = []
                self.add_threads: list[int | None] = []
                self.query_threads: list[int | None] = []

            def init_index(self, **_kwargs) -> None:
                pass

            def set_num_threads(self, value: int) -> None:
                self.set_threads.append(value)

            def add_items(self, _base, _ids, num_threads=None) -> None:
                self.add_threads.append(num_threads)

            def set_ef(self, _value: int) -> None:
                pass

            def knn_query(self, batch, k: int, num_threads=None):
                self.query_threads.append(num_threads)
                return (
                    np.zeros((len(batch), k), dtype=np.int64),
                    np.zeros((len(batch), k), dtype=np.float32),
                )

        index = FakeIndex()
        fake_hnswlib = SimpleNamespace(Index=lambda **_kwargs: index)
        profile = resolved_profile("smoke")
        profile.update(k=1, warmups=0, repeats=2)
        base = np.zeros((2, 2), dtype=np.float32)
        queries = np.zeros((1, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as raw:
            truth_path = Path(raw) / "truth.npy"
            np.save(truth_path, np.zeros((1, 1), dtype=np.int64), allow_pickle=False)
            spec = {
                "dataset": {"n": 2, "nq": 1, "dim": 2},
                "profile": profile,
                "truth_path": str(truth_path),
                "energy": False,
                "seed": 7,
                "library": None,
                "gate_only": "cagra-recall",
                "point_selection": {"hnsw": [CAGRA_RECALL_GATE_POINTS["hnsw"]]},
                "hnsw_threads": 3,
            }
            lane = {
                "id": "hnswlib.hnsw",
                "implementation": "hnswlib",
                "algorithm": "hnsw",
            }
            with (
                patch.dict(sys.modules, {"hnswlib": fake_hnswlib}),
                patch.object(_worker, "load_dataset", return_value=(base, queries)),
                patch.object(_worker, "_optional_energy_reader", return_value=(None, None)),
                patch.object(_worker, "package_version", return_value="0.8.0"),
            ):
                result = _worker.run_hnsw(spec, lane)

        self.assertEqual(result["status"], "success")
        self.assertEqual(index.set_threads, [3])
        self.assertEqual(index.add_threads, [3])
        self.assertTrue(index.query_threads)
        self.assertEqual(set(index.query_threads), {3})
        self.assertEqual(result["implementation_metadata"]["threading"], "explicit")
        self.assertEqual(result["implementation_metadata"]["num_threads"], 3)
        self.assertEqual(
            result["points"][0]["search_threading"]["effective_threads_histogram"],
            {"1": 1},
        )


class OvvsBuildPolicyTests(unittest.TestCase):
    class FakeResources:
        def __init__(self) -> None:
            self.policies: list[int] = []
            self.closed = False
            self.walks = 0
            self.ivfpq_calls = 0
            self.ivfpq_queries = 0

        def set_policy(self, value: int) -> None:
            self.policies.append(value)

        def last_device(self) -> int:
            return 3

        def energy_uj(self):
            return None

        def cagra_transfer_stats(self) -> dict[str, int]:
            return {
                "walks": self.walks,
                "direct_walks": self.walks,
                "index_upload_calls": 0,
                "index_upload_bytes": 0,
            }

        def ivfpq_search_stats(self) -> dict[str, int]:
            return {
                "abi_version": 1,
                "struct_size": 200,
                **{
                    key: (
                        self.ivfpq_calls
                        if key == "successful_calls"
                        else self.ivfpq_queries if key == "queries" else 0
                    )
                    for key in _worker._IVFPQ_SEARCH_STATS_COUNTERS
                },
            }

        def close(self) -> None:
            self.closed = True

    class FakeIndex:
        def __init__(self, resources=None) -> None:
            self.closed = False
            self.resources = resources

        def search(self, batch, **kwargs):
            if self.resources is not None:
                self.resources.walks += 1
                self.resources.ivfpq_calls += 1
                self.resources.ivfpq_queries += len(batch)
            return np.asarray([[0]], dtype=np.int64), np.asarray([[0.0]], dtype=np.float32)

        def close(self) -> None:
            self.closed = True

    @staticmethod
    def fake_module(resources: "OvvsBuildPolicyTests.FakeResources") -> SimpleNamespace:
        return SimpleNamespace(
            Resources=lambda: resources,
            version=lambda: "test",
        )

    @staticmethod
    def spec() -> dict:
        return {
            "build_policy": "auto",
            "dataset": {"n": 8, "nq": 1, "dim": 2},
            "profile": {"k": 1},
            "truth_path": "does-not-exist.npy",
            "energy": False,
        }

    @staticmethod
    def lane(policy: str = "gpu") -> dict:
        return {
            "id": f"ovvs.cagra.{policy}",
            "implementation": "ovvs",
            "algorithm": "cagra",
            "policy_key": policy,
            "policy": policy.upper(),
        }

    def test_auto_build_policy_is_applied_before_independent_search_policy(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex()
        metadata = [
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 2, "npu_fallbacks": 3},
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 2, "npu_fallbacks": 3},
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 4, "npu_fallbacks": 8},
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 4, "npu_fallbacks": 8},
        ]
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), object())),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(_worker, "point_parameters", return_value=[]),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(self.spec(), self.lane("gpu"))

        self.assertEqual(resources.policies, [0, 5])
        self.assertTrue(resources.closed)
        self.assertTrue(index.closed)
        self.assertEqual(result["build"]["status"], "success")
        self.assertEqual(result["build"]["policy"], "AUTO")
        self.assertEqual(result["build"]["requested_policy"], "AUTO")
        self.assertEqual(result["build"]["elapsed_ms_scope"], "complete Python/API build wall")
        self.assertEqual(result["build"]["last_device"]["label"], "GPU")
        self.assertEqual(result["build"]["policy_contract"]["status"], "not_forced")
        self.assertEqual(result["build"]["resource_deltas"]["npu_compile_fails"], 2)
        self.assertEqual(result["build"]["fallback_telemetry"]["npu_compile_fails"]["delta"], 2)
        self.assertEqual(result["build"]["fallback_telemetry"]["npu_fallbacks"]["delta"], 5)

    def test_early_loaded_ovvs_failure_retains_binary_identity(self) -> None:
        resources = self.FakeResources()
        identity = valid_loaded_library_identity()
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(
                _worker,
                "ovvs_loaded_library_identity",
                return_value=identity,
            ),
            patch.object(
                _worker,
                "ovvs_resource_metadata",
                return_value={"gpu_available": 0, "npu_available": 1},
            ),
        ):
            result = _worker.run_ovvs(self.spec(), self.lane("gpu"))

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("resource probe reports no GPU", result["reason"])
        self.assertEqual(result["implementation_metadata"]["loaded_library"], identity)
        self.assertEqual(result["implementation_metadata"]["ovvs_version"], "test")
        self.assertTrue(resources.closed)

    def test_early_loaded_hnsw_export_failure_retains_binary_identity(self) -> None:
        resources = self.FakeResources()
        identity = valid_loaded_library_identity()
        spec = self.spec()
        lane = {
            "id": "ovvs.cagra-hnsw-export.auto",
            "implementation": "hnswlib-export",
            "algorithm": "cagra-hnsw-export",
        }
        with (
            patch.dict(sys.modules, {"hnswlib": SimpleNamespace(Index=lambda **_kwargs: None)}),
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(
                _worker,
                "ovvs_loaded_library_identity",
                return_value=identity,
            ),
            patch.object(
                _worker,
                "load_dataset",
                side_effect=FileNotFoundError("fixture is missing"),
            ),
        ):
            result = _worker.run_hnsw_export(spec, lane)

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("fixture is missing", result["reason"])
        self.assertEqual(result["implementation_metadata"]["loaded_library"], identity)
        self.assertEqual(result["implementation_metadata"]["ovvs_version"], "test")

    def test_failed_build_keeps_status_reason_duration_and_telemetry(self) -> None:
        resources = self.FakeResources()
        unavailable = RuntimeError("GPU build unavailable")
        unavailable.status = 7
        metadata = [
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 0, "npu_fallbacks": 0},
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 0, "npu_fallbacks": 0},
            {"gpu_available": 1, "npu_available": 1, "npu_compile_fails": 1, "npu_fallbacks": 1},
        ]
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), object())),
            patch.object(_worker, "_build_ovvs_index", side_effect=unavailable),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(self.spec(), self.lane("auto"))

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["build"]["status"], "unavailable")
        self.assertIn("index build under AUTO", result["build"]["reason"])
        self.assertGreaterEqual(result["build"]["elapsed_ms"], 0)
        self.assertEqual(result["build"]["last_device"]["status"], "snapshot_unverified_after_failure")
        self.assertIn("resource default", result["build"]["last_device"]["reason"])
        self.assertEqual(result["build"]["resource_before"]["npu_fallbacks"], 0)
        self.assertEqual(result["build"]["resource_after"]["npu_fallbacks"], 1)
        self.assertEqual(result["build"]["fallback_telemetry"]["npu_fallbacks"]["delta"], 1)
        self.assertEqual(result["points"], [])
        self.assertTrue(resources.closed)

    def test_fixed_mode_worker_rejects_missing_build_telemetry(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex()
        spec = self.spec()
        spec["gate_only"] = "cagra-recall"
        metadata = {
            "gpu_available": 1,
            "npu_available": 1,
            "npu_compile_fails": 0,
            "npu_fallbacks": 0,
        }
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), object())),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(_worker, "ovvs_resource_metadata", return_value=metadata),
        ):
            result = _worker.run_ovvs(spec, self.lane("gpu"))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["build"]["status"], "success")
        self.assertEqual(result["build"]["build_algo"], "nndescent")
        self.assertEqual(result["build"]["cagra_build_stats"]["status"], "unavailable")
        self.assertIn("fixed-mode", result["reason"])
        self.assertEqual(resources.policies, [0])
        self.assertTrue(index.closed)
        self.assertTrue(resources.closed)

    def test_cagra_point_delta_includes_measurement_and_energy_searches(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(warmups=1, repeats=2)
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, current_resources):
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "cagra_transfer_stats": current_resources.cagra_transfer_stats(),
            }

        def measure(search, _queries, _batch_size, _warmups, _repeats):
            search(queries)
            ids, distances = search(queries)
            return {
                "pass_latency_ms": {"summary": {"count": 2}},
                "qps": {"summary": {"median": 1.0}},
            }, ids, distances

        def energy(_spec, _reader, search_once, _nq):
            search_once()
            return {"status": "success", "microjoules_per_query": 1.0}

        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"itopk_size": 32, "search_width": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "measure_search", side_effect=measure),
            patch.object(_worker, "_energy", side_effect=energy),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, self.lane("gpu"))

        transfer = result["points"][0]["cagra_transfer"]
        self.assertEqual(transfer["scope"], "warmups_timed_repeats_and_energy_passes")
        self.assertEqual(transfer["delta"]["walks"], 3)
        self.assertEqual(transfer["delta"]["direct_walks"], 3)
        self.assertEqual(transfer["delta"]["index_upload_calls"], 0)
        self.assertEqual(transfer["delta"]["index_upload_bytes"], 0)
        self.assertEqual(
            result["points"][0]["device_attribution"],
            {"search_calls": 3, "successful_observations": 3, "failures": 0},
        )

    def test_failed_cagra_point_preserves_partial_transfer_delta(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(warmups=1, repeats=2)
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, current_resources):
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "cagra_transfer_stats": current_resources.cagra_transfer_stats(),
            }

        def fail_after_one_search(search, _queries, _batch_size, _warmups, _repeats):
            search(queries)
            raise RuntimeError("search failed after transfer")

        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"itopk_size": 32, "search_width": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "measure_search", side_effect=fail_after_one_search),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, self.lane("gpu"))

        point = result["points"][0]
        self.assertEqual(point["status"], "failed")
        self.assertEqual(point["cagra_transfer"]["delta"]["walks"], 1)
        self.assertEqual(point["cagra_transfer"]["delta"]["direct_walks"], 1)

    def test_ivfpq_timed_delta_excludes_warmup_and_energy_searches(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(krefine=1, warmups=1, repeats=2)
        spec["energy"] = True
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, current_resources):
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "ivfpq_search_stats": current_resources.ivfpq_search_stats(),
            }

        def measure(
            search,
            _queries,
            _batch_size,
            warmups,
            repeats,
            phase_callback=None,
        ):
            phase_callback("warmup", "before")
            for _ in range(warmups):
                search(queries)
            phase_callback("warmup", "after")
            phase_callback("timed", "before")
            for _ in range(repeats):
                ids, distances = search(queries)
            phase_callback("timed", "after")
            return {
                "pass_latency_ms": {"summary": {"count": repeats}},
                "qps": {"summary": {"median": 1.0}},
            }, ids, distances

        def energy(_spec, _reader, search_once, _nq):
            search_once()
            return {"status": "success", "microjoules_per_query": 1.0}

        lane = {
            "id": "ovvs.ivf-pq.gpu",
            "implementation": "ovvs",
            "algorithm": "ivf-pq",
            "policy_key": "gpu",
            "policy": "FORCE_GPU",
        }
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"nprobe": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "measure_search", side_effect=measure),
            patch.object(_worker, "_energy", side_effect=energy),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, lane)

        telemetry = result["points"][0]["ivfpq_search_stats"]
        self.assertEqual(telemetry["status"], "success")
        self.assertEqual(telemetry["scope"], "timed")
        self.assertEqual(telemetry["before"]["successful_calls"], 1)
        self.assertEqual(telemetry["after"]["successful_calls"], 3)
        self.assertEqual(telemetry["delta"]["successful_calls"], 2)
        scopes = result["points"][0]["ivfpq_search_stats_scopes"]
        self.assertEqual(scopes["warmup"]["delta"]["successful_calls"], 1)
        self.assertEqual(scopes["timed"]["delta"]["successful_calls"], 2)
        self.assertEqual(scopes["energy"]["delta"]["successful_calls"], 1)
        self.assertEqual(scopes["complete_point"]["delta"]["successful_calls"], 4)
        self.assertEqual(
            result["points"][0]["device_attribution"],
            {"search_calls": 4, "successful_observations": 4, "failures": 0},
        )

    def test_ivfpq_missing_telemetry_keeps_all_scopes_explicit(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(krefine=1, warmups=0, repeats=1)
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, _resources):
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "ivfpq_search_stats": None,
            }

        lane = {
            "id": "ovvs.ivf-pq.gpu",
            "implementation": "ovvs",
            "algorithm": "ivf-pq",
            "policy_key": "gpu",
            "policy": "FORCE_GPU",
        }
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"nprobe": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, lane)

        point = result["points"][0]
        self.assertEqual(point["status"], "success")
        self.assertEqual(point["ivfpq_search_stats"]["status"], "unavailable")
        self.assertEqual(
            set(point["ivfpq_search_stats_scopes"]),
            {"warmup", "timed", "energy", "complete_point"},
        )
        self.assertTrue(
            all(
                evidence["status"] == "unavailable"
                for evidence in point["ivfpq_search_stats_scopes"].values()
            )
        )

    def test_ivfpq_present_invalid_timed_telemetry_fails_point(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(krefine=1, warmups=0, repeats=1)
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, current_resources):
            stats = current_resources.ivfpq_search_stats()
            # A nonzero total with zero named stages is a present but invalid
            # complete-call record; it must not be silently treated as evidence.
            stats["total_wall_ns"] = current_resources.ivfpq_calls
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "ivfpq_search_stats": stats,
            }

        lane = {
            "id": "ovvs.ivf-pq.gpu",
            "implementation": "ovvs",
            "algorithm": "ivf-pq",
            "policy_key": "gpu",
            "policy": "FORCE_GPU",
        }
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"nprobe": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, lane)

        point = result["points"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(point["status"], "failed")
        self.assertEqual(point["ivfpq_search_stats"]["status"], "invalid")
        self.assertIn("do not sum", point["reason"])

    def test_ivfpq_failed_point_preserves_completed_phase_evidence(self) -> None:
        resources = self.FakeResources()
        index = self.FakeIndex(resources)
        spec = self.spec()
        spec["profile"].update(krefine=1, warmups=0, repeats=1)
        queries = np.zeros((1, 2), dtype=np.float32)

        def metadata(_module, current_resources):
            return {
                "gpu_available": 1,
                "npu_available": 1,
                "npu_compile_fails": 0,
                "npu_fallbacks": 0,
                "ivfpq_search_stats": current_resources.ivfpq_search_stats(),
            }

        def fail_after_timed_search(
            search,
            _queries,
            _batch_size,
            _warmups,
            _repeats,
            phase_callback=None,
        ):
            phase_callback("warmup", "before")
            phase_callback("warmup", "after")
            phase_callback("timed", "before")
            search(queries)
            phase_callback("timed", "after")
            raise RuntimeError("search failed after one timed call")

        lane = {
            "id": "ovvs.ivf-pq.gpu",
            "implementation": "ovvs",
            "algorithm": "ivf-pq",
            "policy_key": "gpu",
            "policy": "FORCE_GPU",
        }
        with (
            patch.object(_worker, "load_ovvs", return_value=self.fake_module(resources)),
            patch.object(_worker, "load_dataset", return_value=(object(), queries)),
            patch.object(_worker, "_build_ovvs_index", return_value=index),
            patch.object(
                _worker,
                "point_parameters",
                return_value=[{"nprobe": 1, "query_batch_size": 1}],
            ),
            patch.object(_worker, "measure_search", side_effect=fail_after_timed_search),
            patch.object(_worker, "ovvs_resource_metadata", side_effect=metadata),
        ):
            result = _worker.run_ovvs(spec, lane)

        point = result["points"][0]
        self.assertEqual(point["status"], "failed")
        self.assertIn("failed after one timed call", point["reason"])
        self.assertEqual(point["ivfpq_search_stats"]["status"], "success")
        self.assertEqual(point["ivfpq_search_stats"]["delta"]["successful_calls"], 1)
        scopes = point["ivfpq_search_stats_scopes"]
        self.assertEqual(scopes["warmup"]["delta"]["successful_calls"], 0)
        self.assertEqual(scopes["timed"]["delta"]["successful_calls"], 1)
        self.assertEqual(scopes["energy"]["status"], "unavailable")
        self.assertEqual(scopes["complete_point"]["delta"]["successful_calls"], 1)


if __name__ == "__main__":
    unittest.main()
