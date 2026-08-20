from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.media_hash import sha256_stream, update_registry_source_hash


def _registry(path: Path, *, sha256: str | None = None, expected_size: int | None = 3) -> None:
    source = {
        "id": "sample",
        "status": "candidate",
        "evaluation_stage": "metadata",
        "title": "sample",
        "provider": "Wikimedia Commons",
        "source_page": "https://commons.wikimedia.org/wiki/File:sample.webm",
        "media_url": "https://upload.wikimedia.org/sample.webm",
        "author": "author",
        "license": {
            "name": "CC0",
            "status": "verified",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        "duration_seconds": 1,
        "resolution": [1, 1],
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "metadata_evidence": {"source_size_bytes": expected_size},
    }
    if sha256 is not None:
        source["sha256"] = sha256
    path.write_text(
        json.dumps(
            {
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
                "videos": [source],
            }
        ),
        encoding="utf-8",
    )


class MediaHashTest(unittest.TestCase):
    def test_sha256_stream_hashes_exact_bytes(self) -> None:
        digest, size = sha256_stream(io.BytesIO(b"abc"), chunk_size=2)
        self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(size, 3)

    def test_update_registry_source_hash_records_downloaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "videos.json"
            _registry(registry)
            expected = hashlib.sha256(b"abc").hexdigest()

            with patch("processing.media_hash.hash_source_media", return_value=(expected, 3)):
                result = update_registry_source_hash("sample", registry)

            saved = json.loads(registry.read_text(encoding="utf-8"))["videos"][0]
            self.assertEqual(result, {"id": "sample", "sha256": expected, "size_bytes": 3})
            self.assertEqual(saved["sha256"], expected)
            self.assertEqual(saved["metadata_evidence"]["downloaded_size_bytes"], 3)
            self.assertIs(saved["metadata_evidence"]["sha256_verified_from_downloaded_bytes"], True)

    def test_update_registry_source_hash_refuses_existing_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "videos.json"
            _registry(registry, sha256="0" * 64)
            actual = hashlib.sha256(b"abc").hexdigest()

            with patch("processing.media_hash.hash_source_media", return_value=(actual, 3)):
                with self.assertRaisesRegex(ValueError, "does not match downloaded bytes"):
                    update_registry_source_hash("sample", registry)


if __name__ == "__main__":
    unittest.main()
