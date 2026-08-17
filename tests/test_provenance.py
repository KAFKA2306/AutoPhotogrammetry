import json
import tempfile
import unittest
from pathlib import Path

from processing.provenance import VideoSource, write_source_manifest


class ProvenanceTests(unittest.TestCase):
    def test_source_manifest_hashes_exact_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.webm"
            video.write_bytes(b"video bytes")
            source = VideoSource(
                title="Example",
                source_page="https://example.org/file",
                media_url="https://example.org/file.webm",
                author="Author",
                license="CC BY 3.0",
                license_url="https://creativecommons.org/licenses/by/3.0/",
                target="Example target",
            )
            manifest_path = root / "manifest.json"
            manifest = write_source_manifest(
                video,
                source,
                {
                    "format": {
                        "duration": "123",
                        "format_name": "matroska,webm",
                    }
                },
                manifest_path,
                downloaded_at="2026-08-17T00:00:00+00:00",
            )
            self.assertEqual(manifest["video"]["size_bytes"], 11)
            self.assertEqual(len(manifest["video"]["sha256"]), 64)
            self.assertEqual(
                json.loads(manifest_path.read_text())["source"]["license"],
                "CC BY 3.0",
            )


if __name__ == "__main__":
    unittest.main()
