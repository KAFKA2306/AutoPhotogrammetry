import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.media_hash import (
    sha256_stream,
    update_registry_source_hash,
    update_unhashed_registry_sources,
)


def _valid_registry(video: dict) -> dict:
    video.setdefault("evaluation_stage", "metadata")
    video.setdefault("measurements", {"preflight": None, "colmap": None, "splat": None})
    return {
        "schema_version": 2,
        "default": "sample",
        "evaluation_policy": {
            "stages": {
                "metadata": {},
                "preflight": {},
                "colmap": {},
                "splat": {},
            }
        },
        "videos": [video],
    }


class MediaHashBatchTests(unittest.TestCase):
    def test_sha256_stream_hashes_exact_bytes(self):
        digest, size = sha256_stream(io.BytesIO(b"abc"), chunk_size=2)
        self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(size, 3)

    def test_update_registry_source_hash_records_downloaded_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "videos.json"
            registry = _valid_registry(
                {
                    "id": "sample",
                    "media_url": "https://upload.wikimedia.org/sample.webm",
                    "metadata_evidence": {"source_size_bytes": 3},
                }
            )
            path.write_text(json.dumps(registry), encoding="utf-8")
            expected = hashlib.sha256(b"abc").hexdigest()
            with patch("processing.media_hash.hash_source_media", return_value=(expected, 3)):
                result = update_registry_source_hash("sample", path)
            saved = json.loads(path.read_text(encoding="utf-8"))["videos"][0]
            self.assertEqual(result, {"id": "sample", "sha256": expected, "size_bytes": 3})
            self.assertEqual(saved["sha256"], expected)
            self.assertEqual(saved["metadata_evidence"]["downloaded_size_bytes"], 3)
            self.assertTrue(saved["metadata_evidence"]["sha256_verified_from_downloaded_bytes"])

    def test_update_registry_source_hash_refuses_existing_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "videos.json"
            registry = _valid_registry(
                {
                    "id": "sample",
                    "media_url": "https://upload.wikimedia.org/sample.webm",
                    "sha256": "0" * 64,
                    "metadata_evidence": {"source_size_bytes": 3},
                }
            )
            path.write_text(json.dumps(registry), encoding="utf-8")
            with patch(
                "processing.media_hash.hash_source_media",
                return_value=(hashlib.sha256(b"abc").hexdigest(), 3),
            ):
                with self.assertRaisesRegex(ValueError, "does not match downloaded bytes"):
                    update_registry_source_hash("sample", path)

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
