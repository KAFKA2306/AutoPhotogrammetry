from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.provenance import source_revision, write_json

REVISION = "a" * 40


class ProvenanceRevisionTest(unittest.TestCase):
    def test_environment_revision_is_exact_authority(self):
        with patch.dict(os.environ, {"AUTOPHOTOGRAMMETRY_SOURCE_REVISION": REVISION}, clear=False):
            self.assertEqual(REVISION, source_revision())

    def test_reconstruction_manifest_gets_source_revision_before_write(self):
        manifest = {
            "schema_version": 2,
            "dataset": "demo",
            "status": "running",
            "started_at": "2026-08-20T00:00:00Z",
            "commands": [],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AUTOPHOTOGRAMMETRY_SOURCE_REVISION": REVISION},
                clear=False,
            ),
        ):
            path = Path(directory) / "manifest.json"
            write_json(path, manifest)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(REVISION, manifest["source_revision"])
        self.assertEqual(REVISION, saved["source_revision"])

    def test_unrelated_json_is_not_modified(self):
        payload = {"status": "success", "value": 1}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"AUTOPHOTOGRAMMETRY_SOURCE_REVISION": REVISION},
                clear=False,
            ),
        ):
            path = Path(directory) / "result.json"
            write_json(path, payload)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("source_revision", payload)
        self.assertNotIn("source_revision", saved)

    def test_invalid_revision_fails_closed(self):
        with patch.dict(
            os.environ,
            {"AUTOPHOTOGRAMMETRY_SOURCE_REVISION": "main"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "40-character Git commit SHA"):
                source_revision()


if __name__ == "__main__":
    unittest.main()
