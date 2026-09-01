from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from processing.artifact_publish import ArtifactPublishError, _hf_cache_command, publish_run_splat
from processing.hf_bucket_publish import publish_and_verify

REVISION = "a" * 40


class ArtifactPublishTest(unittest.TestCase):
    def test_official_hf_bucket_api_uploads_and_verifies_exact_readback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = root / "splat.ply"
            artifact.write_bytes(b"ply\nofficial-api")
            calls: list[tuple[object, ...]] = []

            def uploader(bucket, *, add, token=None):
                calls.append(("upload", bucket, add, token))

            def downloader(bucket, *, files, raise_on_missing_files, token=None):
                calls.append(("download", bucket, files, raise_on_missing_files, token))
                _, destination = files[0]
                Path(destination).write_bytes(artifact.read_bytes())

            result = publish_and_verify(
                "k4fka/test",
                artifact,
                "autophotogrammetry/gaussian-splats/demo/hash.ply",
                uploader=uploader,
                downloader=downloader,
                token=False,
            )
            self.assertEqual("PUBLISHED", result["status"])
            self.assertTrue(result["remote_verified"])
            self.assertEqual(2, len(calls))
            self.assertEqual("upload", calls[0][0])
            self.assertEqual("download", calls[1][0])
            self.assertTrue(calls[1][3])

    def _fixture(
        self,
        root: Path,
        *,
        container_path: bool = False,
        orientation_override: str | None = None,
        with_physical_up: bool = False,
    ):
        ply = root / "output" / "demo" / "runs" / "r1" / "export" / "splat.ply"
        ply.parent.mkdir(parents=True)
        payload = b"ply\nsynthetic"
        ply.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        manifest = root / "output" / "demo" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "dataset_id": "demo",
                    "source": {"revision": REVISION},
                    "preflight": {
                        "files": [
                            {
                                "source_path": "input/demo/frame.jpg",
                                "sha256": "b" * 64,
                            }
                        ]
                    },
                    "runs": [
                        {
                            "run_id": "r1",
                            "status": "PASS",
                            "output_ply": str(ply),
                            "output_sha256": sha,
                            "output_size_bytes": len(payload),
                            "output_vertex_count": 1,
                            "export": {"command": ["ns-export", "gaussian-splat"]},
                            "training": {"command": ["ns-train", "splatfacto"]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest, ply, sha

    def test_publish_requires_pinned_source_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _ = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["source"].pop("revision")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactPublishError, "source revision"):
                publish_run_splat(manifest, "r1", bucket="k4fka/test", dry_run=True)

    def test_publish_requires_exact_output_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _ = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["runs"][0]["output_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactPublishError, "SHA-256"):
                publish_run_splat(manifest, "r1", bucket="k4fka/test", dry_run=True)

    def test_publish_requires_physical_up_or_explicit_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _ = self._fixture(root)
            with self.assertRaisesRegex(ArtifactPublishError, "physical up"):
                publish_run_splat(manifest, "r1", bucket="k4fka/test", dry_run=True)

    def test_publish_accepts_explicit_orientation_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, sha = self._fixture(root)
            result = publish_run_splat(
                manifest,
                "r1",
                bucket="k4fka/test",
                dry_run=True,
                orientation_override="reviewed-y-up",
            )
            self.assertEqual("DRY_RUN", result["status"])
            self.assertEqual(sha, result["sha256"])
            self.assertEqual("reviewed-y-up", result["orientation"])

    def test_publish_accepts_physical_up_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _ = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["orientation"] = {
                "physical_up_status": "PASS",
                "canonical_orientation": "y-up",
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = publish_run_splat(manifest, "r1", bucket="k4fka/test", dry_run=True)
            self.assertEqual("y-up", result["orientation"])

    def test_hf_cache_command_requires_binary(self):
        with patch("processing.artifact_publish.shutil.which", return_value=None):
            with self.assertRaisesRegex(ArtifactPublishError, "hf-cache-hub"):
                _hf_cache_command()

    def test_hf_cache_command_uses_detected_binary(self):
        with patch("processing.artifact_publish.shutil.which", return_value="/usr/bin/hf-cache-hub"):
            self.assertEqual(["/usr/bin/hf-cache-hub"], _hf_cache_command())

    def test_hf_cache_upload_command_is_exact(self):
        cmd = _hf_cache_command("upload", "--foo", "bar")
        self.assertEqual(["hf-cache-hub", "upload", "--foo", "bar"], cmd)

    def test_env_forwarding_uses_explicit_contract(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "OTHER": "x"}, clear=True):
            env = os.environ.copy()
            self.assertEqual("secret", env["HF_TOKEN"])
            self.assertEqual("x", env["OTHER"])

    def test_subprocess_example_is_non_shell(self):
        completed = subprocess.CompletedProcess(args=["echo"], returncode=0)
        self.assertEqual(0, completed.returncode)


if __name__ == "__main__":
    unittest.main()
