from __future__ import annotations

import unittest

from processing.video_sources import get_video_source, load_video_registry, ranked_video_sources


class VideoSourceRegistryTests(unittest.TestCase):
    def test_registry_has_many_ranked_candidates(self) -> None:
        videos = ranked_video_sources()
        self.assertGreaterEqual(len(videos), 20)
        self.assertEqual([video["rank"] for video in videos], list(range(1, len(videos) + 1)))
        self.assertEqual({video["expected_success"] for video in videos}, {"high", "medium", "low"})

    def test_default_source_is_verified_and_frozen(self) -> None:
        registry = load_video_registry()
        source = get_video_source()
        self.assertEqual(source["id"], registry["default"])
        self.assertEqual(source["status"], "verified")
        self.assertEqual(source["license"]["status"], "verified")
        self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(source["expected_frame_count"], 0)

    def test_ids_and_source_pages_are_unique(self) -> None:
        videos = ranked_video_sources()
        self.assertEqual(len({video["id"] for video in videos}), len(videos))
        self.assertEqual(len({video["source_page"] for video in videos}), len(videos))
        self.assertTrue(all(video["source_page"].startswith("https://commons.wikimedia.org/wiki/File:") for video in videos))

    def test_unknown_source_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_video_source("does-not-exist")


if __name__ == "__main__":
    unittest.main()
