from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from processing.nordic_pool import (
    EXPECTED_REGION_IDS,
    build_coverage,
    persist_snapshot,
    queue_candidate,
    refresh_snapshots,
    stable_candidate_id,
    stage_a_gate,
    validate_pool,
    validate_queue,
)
from processing.nordic_preflight import preflight_batch, write_evaluation_registry
from processing.nordic_seeds import load_nordic_seeds


def complete_candidate() -> dict:
    candidate = {
        "id": stable_candidate_id("File:Example.webm"),
        "canonical_title": "File:Example.webm",
        "title": "Example",
        "provider": "Wikimedia Commons",
        "regions": ["norway"],
        "source_page": "https://commons.wikimedia.org/wiki/File:Example.webm",
        "media_url": "https://upload.wikimedia.org/example.webm",
        "author": "Example Author",
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "status": "verified",
        },
        "duration_seconds": 180.0,
        "duration_authority": "fixture",
        "resolution": [1920, 1080],
        "source_sha1": "a" * 40,
        "source_size_bytes": 100,
        "mime": "video/webm",
        "media_type": "VIDEO",
        "confirmed_video": True,
        "downloadable": True,
        "commons_categories": [],
        "discovered_categories": ["Category:Drone videos from Norway"],
        "discovery_paths": [["Category:Drone videos from Norway"]],
        "evaluation_stage": "metadata",
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "metadata_authority": "fixture",
    }
    candidate["stage_a"] = stage_a_gate(candidate)
    return candidate


def pool_with(candidate: dict) -> dict:
    return {
        "schema_version": 1,
        "snapshot_state": "api",
        "authority": {},
        "seed_config_sha256": "b" * 64,
        "regions": list(EXPECTED_REGION_IDS),
        "minimum_duration_seconds": 120.0,
        "raw_discovered_file_count_by_region": {
            region: int(region == "norway") for region in EXPECTED_REGION_IDS
        },
        "candidates": [candidate],
        "discovery_failures": [],
        "metadata_failures": [],
    }


def registry_fixture() -> dict:
    stages = {
        "metadata": {"purpose": "fixture", "fields": []},
        "preflight": {"purpose": "fixture", "fields": []},
        "colmap": {"purpose": "fixture", "fields": []},
        "splat": {"purpose": "fixture", "fields": []},
    }
    return {
        "schema_version": 2,
        "default": "existing",
        "evaluation_policy": {"stages": stages, "principle": "fixture"},
        "videos": [
            {
                "id": "existing",
                "evaluation_stage": "metadata",
                "measurements": {"preflight": None, "colmap": None, "splat": None},
            }
        ],
    }


class NordicPoolTests(unittest.TestCase):
    def test_stage_a_is_fail_closed_for_unknown_license(self) -> None:
        candidate = complete_candidate()
        candidate["license"] = {"name": "CC BY 4.0", "url": None, "status": "unknown"}
        result = stage_a_gate(candidate)
        self.assertFalse(result["eligible_for_preflight"])
        self.assertIn("license_present", result["failed_requirements"])
        self.assertIn("license_verified", result["failed_requirements"])

    def test_stage_a_requires_commons_identity_and_keeps_duration_gate_separate(self) -> None:
        candidate = complete_candidate()
        candidate["duration_seconds"] = 119.0
        candidate["source_sha1"] = None
        result = stage_a_gate(candidate)
        self.assertFalse(result["eligible_for_preflight"])
        self.assertFalse(result["duration_threshold_pass"])
        self.assertIn("source_sha1_present", result["failed_requirements"])
        self.assertNotIn("score", result)
        self.assertNotIn("rank", result)

    def test_pool_rejects_duplicate_ids_and_heuristic_fields(self) -> None:
        candidate = complete_candidate()
        duplicate = copy.deepcopy(candidate)
        duplicate["canonical_title"] = "File:Other.webm"
        payload = pool_with(candidate)
        payload["candidates"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate Nordic candidate id"):
            validate_pool(payload)

        payload = pool_with(candidate)
        payload["candidates"][0]["score"] = 0.9
        with self.assertRaisesRegex(ValueError, "Heuristic ranking fields"):
            validate_pool(payload)

    def test_coverage_always_contains_exact_eight_regions(self) -> None:
        payload = pool_with(complete_candidate())
        coverage = build_coverage(payload, load_nordic_seeds())
        self.assertEqual(
            tuple(row["region_id"] for row in coverage["regions"]),
            EXPECTED_REGION_IDS,
        )
        self.assertEqual(coverage["totals"]["region_count"], 8)
        denmark = next(row for row in coverage["regions"] if row["region_id"] == "denmark")
        aland = next(row for row in coverage["regions"] if row["region_id"] == "aland")
        self.assertEqual(denmark["discovery_state"], "zero_candidates")
        self.assertEqual(aland["discovery_state"], "missing_seed")

    def test_persist_snapshot_ignores_timestamp_only_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            first = pool_with(complete_candidate())
            first["generated_at"] = "2026-01-01T00:00:00Z"
            self.assertTrue(persist_snapshot(path, first))
            before = path.read_text(encoding="utf-8")
            second = copy.deepcopy(first)
            second["generated_at"] = "2026-01-02T00:00:00Z"
            self.assertFalse(persist_snapshot(path, second))
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_partial_refresh_preserves_previous_snapshots(self) -> None:
        failed_pool = pool_with(complete_candidate())
        failed_pool["discovery_failures"] = [{"region_id": "norway", "error": "fixture failure"}]
        with tempfile.TemporaryDirectory() as tmp:
            pool_path = Path(tmp) / "pool.json"
            coverage_path = Path(tmp) / "coverage.json"
            pool_path.write_text('{"sentinel":"pool"}\n', encoding="utf-8")
            coverage_path.write_text('{"sentinel":"coverage"}\n', encoding="utf-8")
            with mock.patch("processing.nordic_pool.build_pool", return_value=failed_pool):
                with self.assertRaisesRegex(RuntimeError, "previous good snapshots"):
                    refresh_snapshots(pool_path=pool_path, coverage_path=coverage_path)
            self.assertEqual(pool_path.read_text(encoding="utf-8"), '{"sentinel":"pool"}\n')
            self.assertEqual(
                coverage_path.read_text(encoding="utf-8"),
                '{"sentinel":"coverage"}\n',
            )

    def test_queue_is_explicit_idempotent_and_does_not_touch_final_registry(self) -> None:
        candidate = complete_candidate()
        with tempfile.TemporaryDirectory() as tmp:
            pool_path = Path(tmp) / "pool.json"
            queue_path = Path(tmp) / "queue.json"
            registry_path = Path(tmp) / "videos.json"
            pool_path.write_text(json.dumps(pool_with(candidate)), encoding="utf-8")
            registry_path.write_text(json.dumps(registry_fixture()), encoding="utf-8")
            before_registry = registry_path.read_text(encoding="utf-8")

            first = queue_candidate(candidate["id"], pool_path=pool_path, queue_path=queue_path)
            second = queue_candidate(candidate["id"], pool_path=pool_path, queue_path=queue_path)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(len(queue["candidates"]), 1)
            self.assertEqual(registry_path.read_text(encoding="utf-8"), before_registry)
            validate_queue(queue, pool_with(candidate))

    def test_evaluation_registry_is_separate_from_final_twenty_registry(self) -> None:
        candidate = complete_candidate()
        with tempfile.TemporaryDirectory() as tmp:
            canonical_path = Path(tmp) / "videos.json"
            evaluation_path = Path(tmp) / "evaluation.json"
            canonical_path.write_text(json.dumps(registry_fixture()), encoding="utf-8")
            before = canonical_path.read_text(encoding="utf-8")

            write_evaluation_registry(
                candidate,
                sha256="c" * 64,
                downloaded_size=100,
                destination=evaluation_path,
                canonical_registry_path=canonical_path,
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

            self.assertEqual(canonical_path.read_text(encoding="utf-8"), before)
            self.assertEqual(evaluation["default"], candidate["id"])
            self.assertEqual(len(evaluation["videos"]), 1)

    def test_preflight_batch_preserves_other_results_after_failure(self) -> None:
        def fake_preflight(candidate_id: str, **_kwargs) -> dict:
            if candidate_id == "bad":
                raise RuntimeError("boom")
            return {"candidate_id": candidate_id, "measurements": {}}

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "processing.nordic_preflight.preflight_candidate",
                side_effect=fake_preflight,
            ):
                result = preflight_batch(["good-1", "bad", "good-2"], output_root=tmp)
            self.assertEqual(result["measured_count"], 2)
            self.assertEqual(result["failed_count"], 1)
            self.assertFalse(result["automatic_colmap_promotion"])
            self.assertTrue((Path(tmp) / "batch.json").is_file())


if __name__ == "__main__":
    unittest.main()
