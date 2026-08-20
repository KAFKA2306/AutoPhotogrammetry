from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from processing.video_sources import get_video_source, load_video_registry, video_sources


class VideoSourceRegistryTests(unittest.TestCase):
    def test_registry_has_many_candidates_without_heuristic_scores(self) -> None:
        videos = video_sources()
        self.assertGreaterEqual(len(videos), 20)
        for video in videos:
            self.assertNotIn("rank", video)
            self.assertNotIn("score", video)
            self.assertNotIn("expected_success", video)

    def test_policy_uses_measured_3dgs_stages(self) -> None:
        registry = load_video_registry()
        self.assertEqual(
            set(registry["evaluation_policy"]["stages"]),
            {"metadata", "preflight", "colmap", "splat"},
        )

    def test_metadata_candidates_have_no_fake_measurements(self) -> None:
        candidates = [video for video in video_sources() if video["evaluation_stage"] == "metadata"]
        self.assertTrue(candidates)
        for video in candidates:
            self.assertIsNone(video["measurements"]["preflight"])
            self.assertIsNone(video["measurements"]["colmap"])
            self.assertIsNone(video["measurements"]["splat"])

    def test_default_source_keeps_measured_colmap_evidence(self) -> None:
        registry = load_video_registry()
        source = get_video_source()
        self.assertEqual(source["id"], registry["default"])
        self.assertEqual(source["status"], "verified")
        self.assertEqual(source["evaluation_stage"], "colmap")
        self.assertEqual(source["license"]["status"], "verified")
        self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(source["measurements"]["colmap"]["registered_images"], 78)
        self.assertEqual(source["measurements"]["colmap"]["input_images"], 78)
        self.assertEqual(source["measurements"]["colmap"]["registration_ratio"], 1.0)
        self.assertEqual(source["measurements"]["colmap"]["submodel_count"], 1)
        self.assertEqual(source["measurements"]["colmap"]["sparse_points"], 32782)
        self.assertAlmostEqual(
            source["measurements"]["colmap"]["mean_reprojection_error_px"],
            0.37083,
        )
        self.assertIsNone(source["measurements"]["splat"])

    def test_puente_san_ignacio_03_keeps_verified_source_metadata(self) -> None:
        source = get_video_source("puente-san-ignacio-03")
        self.assertEqual(source["author"], "Luisalvaz")
        self.assertEqual(source["license"]["name"], "CC0 1.0 Universal")
        self.assertEqual(source["license"]["status"], "verified")
        self.assertEqual(
            source["license"]["url"],
            "https://creativecommons.org/publicdomain/zero/1.0/",
        )
        self.assertEqual(source["duration_seconds"], 140.661)
        self.assertEqual(source["resolution"], [3840, 2160])
        self.assertEqual(
            source["media_url"],
            "https://upload.wikimedia.org/wikipedia/commons/9/97/Puente_de_San_Ignacio_desde_un_dron_03.webm",
        )
        self.assertEqual(source["evaluation_stage"], "metadata")
        self.assertIsNone(source["measurements"]["preflight"])
        self.assertIsNone(source["measurements"]["colmap"])
        self.assertIsNone(source["measurements"]["splat"])

    def test_ids_and_source_pages_are_unique(self) -> None:
        videos = video_sources()
        self.assertEqual(len({video["id"] for video in videos}), len(videos))
        self.assertEqual(len({video["source_page"] for video in videos}), len(videos))
        self.assertTrue(
            all(
                video["source_page"].startswith("https://commons.wikimedia.org/wiki/File:")
                for video in videos
            )
        )

    def test_unknown_source_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_video_source("does-not-exist")

    def test_untrusted_registry_structure_is_rejected_before_semantic_policy(self) -> None:
        malformed = {
            "schema_version": 2,
            "default": "scene",
            "evaluation_policy": {
                "stages": {stage: {} for stage in ("metadata", "preflight", "colmap", "splat")}
            },
            "videos": [{"id": "scene", "evaluation_stage": "metadata"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "videos.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_video_registry(path)


if __name__ == "__main__":
    unittest.main()
