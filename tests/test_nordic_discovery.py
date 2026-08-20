from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.nordic_discovery import (
    REQUIRED_REGIONS,
    MediaWikiClient,
    build_coverage,
    enumerate_region_files,
    load_seed_config,
    persist_snapshot,
    promote_candidate,
    stage_a_gate,
)


def _seed_config() -> dict:
    regions = []
    for region_id in REQUIRED_REGIONS:
        if region_id == "aland":
            seeds = [
                {
                    "state": "missing",
                    "type": "video",
                    "category": None,
                    "url": None,
                    "reason": "No confirmed category.",
                }
            ]
        else:
            category = f"Category:Test {region_id}"
            url = "https://commons.wikimedia.org/wiki/" + category.replace(" ", "_")
            seeds = [
                {
                    "state": "active",
                    "type": "drone",
                    "category": category,
                    "url": url,
                    "recursive": True,
                    "max_depth": 3,
                }
            ]
        regions.append({"id": region_id, "name": region_id, "seeds": seeds})
    return {"schema_version": 1, "minimum_duration_seconds": 120, "regions": regions}


def _candidate(source_id: str = "candidate") -> dict:
    return {
        "id": source_id,
        "status": "discovered",
        "evaluation_stage": "metadata",
        "regions": ["norway"],
        "title": "Example.webm",
        "target": None,
        "provider": "Wikimedia Commons",
        "canonical_file_title": "File:Example.webm",
        "source_page": "https://commons.wikimedia.org/wiki/File:Example.webm",
        "media_url": "https://upload.wikimedia.org/example.webm",
        "author": "Example",
        "license": {
            "name": "CC BY 4.0",
            "status": "verified",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "duration_seconds": 180.0,
        "resolution": [1920, 1080],
        "source_sha1": "a" * 40,
        "source_size_bytes": 123,
        "mime": "video/webm",
        "mediatype": "VIDEO",
        "downloadable": True,
        "discovered_categories": ["Category:Test norway"],
        "file_categories": [],
        "measurements": {"preflight": None, "colmap": None, "splat": None},
    }


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def category_members(self, category: str) -> list[dict]:
        self.calls.append(category)
        if category == "Category:Root":
            return [
                {"title": "Category:Nested", "type": "subcat"},
                {"title": "File:One.webm", "type": "file"},
            ]
        if category == "Category:Nested":
            return [
                {"title": "Category:Root", "type": "subcat"},
                {"title": "File:One.webm", "type": "file"},
                {"title": "File:Two.webm", "type": "file"},
            ]
        return []


class NordicDiscoveryTests(unittest.TestCase):
    def test_seed_config_requires_all_eight_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.json"
            path.write_text(json.dumps(_seed_config()), encoding="utf-8")
            loaded = load_seed_config(path)
        self.assertEqual(tuple(region["id"] for region in loaded["regions"]), REQUIRED_REGIONS)

    def test_categorymembers_follows_continue_token(self) -> None:
        client = MediaWikiClient()
        responses = [
            {
                "continue": {"cmcontinue": "next", "continue": "-||"},
                "query": {"categorymembers": [{"title": "File:A.webm", "type": "file"}]},
            },
            {
                "query": {"categorymembers": [{"title": "File:B.webm", "type": "file"}]},
            },
        ]
        with patch.object(client, "get_json", side_effect=responses) as get_json:
            members = client.category_members("Category:Root")
        self.assertEqual([item["title"] for item in members], ["File:A.webm", "File:B.webm"])
        self.assertEqual(get_json.call_count, 2)
        self.assertEqual(get_json.call_args_list[1].args[0]["cmcontinue"], "next")

    def test_recursive_enumeration_deduplicates_and_breaks_cycles(self) -> None:
        client = _FakeClient()
        region = {
            "id": "norway",
            "seeds": [
                {
                    "state": "active",
                    "category": "Category:Root",
                    "recursive": True,
                    "max_depth": 4,
                }
            ],
        }
        files, failures, reachable = enumerate_region_files(client, region)
        self.assertEqual(set(files), {"File:One.webm", "File:Two.webm"})
        self.assertFalse(failures)
        self.assertEqual(reachable, 1)
        self.assertEqual(client.calls.count("Category:Root"), 1)

    def test_stage_a_passes_only_complete_verified_candidate(self) -> None:
        candidate = _candidate()
        result = stage_a_gate(candidate, minimum_duration_seconds=120)
        self.assertTrue(result["eligible_for_preflight"])
        self.assertEqual(result["failed_requirements"], [])
        self.assertFalse({"rank", "score", "expected_success"}.intersection(result))

    def test_needs_review_license_fails_stage_a(self) -> None:
        candidate = _candidate()
        candidate["license"]["status"] = "needs_review"
        result = stage_a_gate(candidate, minimum_duration_seconds=120)
        self.assertFalse(result["eligible_for_preflight"])
        self.assertIn("license_verified", result["failed_requirements"])

    def test_snapshot_does_not_churn_only_for_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            payload = {"schema_version": 1, "candidates": []}
            self.assertTrue(persist_snapshot(path, payload))
            first = path.read_text(encoding="utf-8")
            self.assertFalse(persist_snapshot(path, payload))
            self.assertEqual(first, path.read_text(encoding="utf-8"))

    def test_promotion_is_idempotent_and_preserves_registry_order(self) -> None:
        candidate = _candidate("nordic-scene")
        candidate["stage_a"] = stage_a_gate(candidate, minimum_duration_seconds=120)
        pool = {"schema_version": 1, "candidates": [candidate]}
        registry = {
            "schema_version": 2,
            "default": "existing",
            "evaluation_policy": {
                "stages": {
                    "metadata": {},
                    "preflight": {},
                    "colmap": {},
                    "splat": {},
                }
            },
            "videos": [
                {
                    "id": "existing",
                    "evaluation_stage": "metadata",
                    "measurements": {"preflight": None, "colmap": None, "splat": None},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pool_path = Path(tmp) / "pool.json"
            registry_path = Path(tmp) / "videos.json"
            pool_path.write_text(json.dumps(pool), encoding="utf-8")
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            promote_candidate("nordic-scene", pool_path=pool_path, registry_path=registry_path)
            promote_candidate("nordic-scene", pool_path=pool_path, registry_path=registry_path)
            updated = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in updated["videos"]], ["existing", "nordic-scene"])
        self.assertEqual(updated["default"], "existing")

    def test_coverage_always_contains_eight_regions(self) -> None:
        config = _seed_config()
        candidate = _candidate()
        candidate["stage_a"] = stage_a_gate(candidate, minimum_duration_seconds=120)
        pool = {
            "schema_version": 1,
            "candidates": [candidate],
            "discovery_failures": [],
        }
        coverage = build_coverage(pool, config)
        self.assertEqual([row["region"] for row in coverage["regions"]], list(REQUIRED_REGIONS))
        self.assertEqual(coverage["totals"]["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
