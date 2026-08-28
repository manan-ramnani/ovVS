from __future__ import annotations

import json
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
from _worker import _cagra_transfer_evidence, exception_point_status, process_peak_rss
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
        "elapsed_ms": 1.0,
        "last_device": {"status": "reported", "code": 1, "label": "CPU"},
        "policy_contract": {"status": "not_forced", "conforming": None},
    }
    if transfer is not None:
        lane["points"][0]["cagra_transfer"] = transfer
    return lane


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

    def test_gate_configuration_normalizes_conflicting_selection(self) -> None:
        args = bench.build_parser().parse_args(
            [
                "--gate-only", "cagra-recall",
                "--profile", "smoke",
                "--algorithms", "brute",
                "--policies", "cpu",
                "--build-policy", "gpu",
                "--warmups", "9",
                "--repeats", "9",
                "--seed", "99",
                "--timeout-seconds", "7200",
            ]
        )
        with patch.object(bench.os, "cpu_count", return_value=20):
            bench.normalize_gate_configuration(args)
        self.assertEqual(args.profile, "sift1m")
        self.assertEqual(args.algorithms, "cagra")
        self.assertEqual(args.policies, "gpu")
        self.assertEqual(args.build_policy, "auto")
        self.assertEqual((args.warmups, args.repeats, args.seed), (1, 5, 7))
        self.assertEqual(args.timeout_seconds, 7200)
        self.assertEqual(args.hnsw_threads, 20)
        self.assertTrue(args.no_energy)
        self.assertTrue(args.allow_unscalable_cagra)

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
                "--warmups", "9",
                "--repeats", "9",
                "--seed", "99",
                "--timeout-seconds", "1200",
            ]
        )
        with patch.object(bench.os, "cpu_count", return_value=20):
            bench.normalize_preflight_configuration(args)
        self.assertEqual(args.profile, "sift-100k")
        self.assertEqual(args.algorithms, "cagra")
        self.assertEqual(args.policies, "gpu")
        self.assertEqual(args.build_policy, "auto")
        self.assertEqual((args.warmups, args.repeats, args.seed), (1, 5, 7))
        self.assertEqual(args.timeout_seconds, 1200)
        self.assertEqual(args.hnsw_threads, 20)
        self.assertTrue(args.no_energy)
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

    def test_ivf_pq_force_gpu_expected_skip_is_nonblocking(self) -> None:
        lanes = enumerate_lanes(["ivf-pq"], ["auto", "gpu"], 1_000_000, True, False)
        by_id = {lane.id: lane for lane in lanes}
        gpu = by_id["ovvs.ivf-pq.gpu"]
        self.assertTrue(gpu.expected_skip)
        self.assertFalse(gpu.blocking_skip)
        self.assertIn("no iGPU backend", gpu.skip_reason)
        self.assertIn("faiss-cpu.ivf-pq", by_id)
        skipped = gpu.as_dict()
        skipped.update(status="skipped", reason=gpu.skip_reason)
        self.assertEqual(
            completion_issues(
                "sift1m",
                {"status": "success"},
                {"status": "success", "exact": True},
                [skipped],
            ),
            [],
        )

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
            ):
                result = _worker.run_hnsw(spec, lane)

        self.assertEqual(result["status"], "success")
        self.assertEqual(index.set_threads, [3])
        self.assertEqual(index.add_threads, [3])
        self.assertTrue(index.query_threads)
        self.assertEqual(set(index.query_threads), {3})
        self.assertEqual(result["implementation_metadata"]["threading"], "explicit")
        self.assertEqual(result["implementation_metadata"]["num_threads"], 3)


class OvvsBuildPolicyTests(unittest.TestCase):
    class FakeResources:
        def __init__(self) -> None:
            self.policies: list[int] = []
            self.closed = False
            self.walks = 0

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

        def close(self) -> None:
            self.closed = True

    class FakeIndex:
        def __init__(self, resources=None) -> None:
            self.closed = False
            self.resources = resources

        def search(self, batch, **kwargs):
            if self.resources is not None:
                self.resources.walks += 1
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
        self.assertEqual(result["build"]["last_device"]["label"], "GPU")
        self.assertEqual(result["build"]["policy_contract"]["status"], "not_forced")
        self.assertEqual(result["build"]["resource_deltas"]["npu_compile_fails"], 2)
        self.assertEqual(result["build"]["fallback_telemetry"]["npu_compile_fails"]["delta"], 2)
        self.assertEqual(result["build"]["fallback_telemetry"]["npu_fallbacks"]["delta"], 5)

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


if __name__ == "__main__":
    unittest.main()
