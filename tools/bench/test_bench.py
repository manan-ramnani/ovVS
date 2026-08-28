from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import bench
import _worker
from _worker import exception_point_status
from _common import (
    ALGORITHM_ORDER,
    POLICY_ORDER,
    ROOT,
    completion_issues,
    default_output_path,
    duplicate_safe_recall,
    enumerate_lanes,
    measure_energy,
    parse_selection,
    percentile,
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


class OvvsBuildPolicyTests(unittest.TestCase):
    class FakeResources:
        def __init__(self) -> None:
            self.policies: list[int] = []
            self.closed = False

        def set_policy(self, value: int) -> None:
            self.policies.append(value)

        def last_device(self) -> int:
            return 3

        def close(self) -> None:
            self.closed = True

    class FakeIndex:
        def __init__(self) -> None:
            self.closed = False

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


if __name__ == "__main__":
    unittest.main()
