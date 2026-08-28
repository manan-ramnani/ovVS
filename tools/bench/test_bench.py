from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import bench
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


class CliAndLaneSemanticsTests(unittest.TestCase):
    def test_no_arg_profile_is_bounded(self) -> None:
        args = bench.build_parser().parse_args([])
        profile = resolved_profile(args.profile)
        self.assertEqual(args.profile, "smoke")
        self.assertLessEqual(profile["expected_n"], 2_000)
        self.assertLessEqual(profile["nq"], 32)
        self.assertLessEqual(profile["timeout_seconds"], 30)
        self.assertGreaterEqual(profile["repeats"], 2)

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


if __name__ == "__main__":
    unittest.main()
