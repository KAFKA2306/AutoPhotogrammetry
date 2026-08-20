import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.media_hash import update_unhashed_registry_sources


class MediaHashBatchTests(unittest.TestCase):
    def test_batch_skips_verified_persists_success_and_isolates_failure(self):
        registry = {
            "videos": [
                {
                    "id": "already-done",
                    "sha256": "a" * 64,
                    "metadata_evidence": {"sha256_verified_from_downloaded_bytes": True},
                },
                {"id": "success", "media_url": "https://example.invalid/success"},
                {"id": "failure", "media_url": "https://example.invalid/failure"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "videos.json"
            path.write_text(json.dumps(registry), encoding="utf-8")

            def fake_hash(source, *, timeout_seconds):
                self.assertEqual(timeout_seconds, 3.0)
                if source["id"] == "failure":
                    raise OSError("network unavailable")
                return "b" * 64, 123

            with (
                patch("processing.media_hash.load_video_registry", return_value=registry),
                patch("processing.media_hash.hash_source_media", side_effect=fake_hash),
            ):
                result = update_unhashed_registry_sources(path, timeout_seconds=3.0)

            self.assertEqual(result["hashed_count"], 1)
            self.assertEqual(result["failed_count"], 1)
            self.assertEqual(result["skipped_verified_count"], 1)
            self.assertEqual(result["hashed"][0]["id"], "success")
            self.assertEqual(result["failed"], [{"id": "failure", "error": "network unavailable"}])
            self.assertEqual(result["skipped_verified"], ["already-done"])

            persisted = json.loads(path.read_text(encoding="utf-8"))
            success = next(item for item in persisted["videos"] if item["id"] == "success")
            self.assertEqual(success["sha256"], "b" * 64)
            self.assertEqual(success["metadata_evidence"]["downloaded_size_bytes"], 123)
            self.assertTrue(success["metadata_evidence"]["sha256_verified_from_downloaded_bytes"])

    def test_existing_unverified_hash_must_match_download(self):
        registry = {
            "videos": [
                {
                    "id": "mismatch",
                    "sha256": "a" * 64,
                    "media_url": "https://example.invalid/mismatch",
                    "metadata_evidence": {},
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "videos.json"
            path.write_text(json.dumps(registry), encoding="utf-8")

            with (
                patch("processing.media_hash.load_video_registry", return_value=registry),
                patch("processing.media_hash.hash_source_media", return_value=("b" * 64, 123)),
            ):
                result = update_unhashed_registry_sources(path)

            self.assertEqual(result["hashed_count"], 0)
            self.assertEqual(result["failed_count"], 1)
            self.assertIn("does not match downloaded bytes", result["failed"][0]["error"])


if __name__ == "__main__":
    unittest.main()
