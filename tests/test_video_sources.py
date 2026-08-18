from __future__ import annotations

import unittest

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

    def test_ids_and_source_pages_are_unique(self) -> None:
        videos = video_sources()
        self.assertEqual(len({video["id"] for video in videos}), len(videos))
        self.assertEqual(len({video["source_page"] for video in videos}), len(videos))
        self.assertTrue(
            all(video["source_page"].startswith("https://commons.wikimedia.org/wiki/File:") for video in videos)
        )

    def test_unknown_source_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_video_source("does-not-exist")


if __name__ == "__main__":
    unittest.main()
