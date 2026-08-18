import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.batch import _file_title, ensure_source, resolve_media_url, run_all_videos


class BatchTests(unittest.TestCase):
    def test_file_title_removes_namespace_prefix(self):
        self.assertEqual(
            _file_title("https://commons.wikimedia.org/wiki/File:Example_video.webm"),
            "Example video.webm",
        )

    def test_resolver_reuses_registry_media_url(self):
        result = resolve_media_url({"media_url": "https://example.test/video.webm"})
        self.assertEqual(result["media_url"], "https://example.test/video.webm")
        self.assertEqual(result["resolved_via"], "registry")

    def test_resolver_reads_wikimedia_imageinfo(self):
        payload = {
            "query": {
                "pages": [{
                    "imageinfo": [{
                        "url": "https://upload.test/video.webm",
                        "size": 123,
                        "sha1": "abc",
                        "mime": "video/webm",
                        "extmetadata": {
                            "Artist": {"value": "Author"},
                            "LicenseShortName": {"value": "CC0"},
                        },
                    }]
                }]
            }
        }
        response = io.BytesIO(json.dumps(payload).encode())
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: None
        with patch("processing.batch.urlopen", return_value=response):
            result = resolve_media_url({
                "source_page": "https://commons.wikimedia.org/wiki/File:Example_video.webm"
            })
        self.assertEqual(result["media_url"], "https://upload.test/video.webm")
        self.assertEqual(result["source_size_bytes"], 123)
        self.assertEqual(result["license"], "CC0")

    def test_source_hash_mismatch_fails_closed(self):
        payload = b"complete video bytes"
        response = io.BytesIO(payload)
        response.__enter__ = lambda: response
        response.__exit__ = lambda *args: None
        with tempfile.TemporaryDirectory() as tmp, patch(
            "processing.batch.urlopen", return_value=response
        ):
            with self.assertRaisesRegex(RuntimeError, "Source SHA-1 mismatch"):
                ensure_source(
                    Path(tmp) / "source.webm",
                    url="https://upload.test/video.webm",
                    expected_sha1="not-the-object-sha1",
                    expected_size=len(payload),
                )

    def test_batch_records_each_video_result(self):
        registry = {
            "schema_version": 2,
            "default": "one",
            "evaluation_policy": {"stages": {
                "metadata": {}, "preflight": {}, "colmap": {}, "splat": {}
            }},
            "videos": [
                {"id": "one", "evaluation_stage": "metadata", "measurements": {"preflight": None, "colmap": None, "splat": None}},
                {"id": "two", "evaluation_stage": "metadata", "measurements": {"preflight": None, "colmap": None, "splat": None}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "videos.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with patch(
                "processing.batch.run_video",
                side_effect=[
                    {"status": "success", "manifest_path": "one/manifest.json", "splatfacto": {"ply_path": "one.ply", "ply_sha256": "a"}},
                    RuntimeError("no GPU"),
                ],
            ):
                result = run_all_videos(
                    registry_path=registry_path,
                    input_root=root / "input",
                    output_root=root / "output",
                )
        self.assertEqual(result["requested"], 2)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual([item["id"] for item in result["results"]], ["one", "two"])
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
