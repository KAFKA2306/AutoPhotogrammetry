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


class ArtifactPublishTest(unittest.TestCase):
    def _fixture(self, root: Path):
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
                    "schema_version": 2,
                    "dataset": "demo",
                    "status": "success",
                    "started_at": "2026-08-20T00:00:00Z",
                    "registry": {
                        "source_page": "https://commons.wikimedia.org/wiki/File:Demo.webm",
                        "license": {"url": "https://creativecommons.org/publicdomain/zero/1.0/"},
                    },
                    "splatfacto": {
                        "ply_path": str(ply),
                        "ply_sha256": sha,
                        "ply_size_bytes": len(payload),
                    },
                }
            ),
            encoding="utf-8",
        )
        hf_root = root / "hf-cache-hub"
        (hf_root / "scripts").mkdir(parents=True)
        (hf_root / "scripts" / "artifact_manager.py").write_text("# cli", encoding="utf-8")
        return manifest, ply, sha, hf_root

    def test_missing_hf_cache_hub_root_fails_with_required_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ArtifactPublishError, "HF_CACHE_HUB_ROOT"):
                _hf_cache_command(None)

    def test_publish_records_remote_verified_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, sha, hf_root = self._fixture(root)
            calls = []

            def runner(command, **kwargs):
                calls.append(command)
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout="a" * 40 + "\n", stderr=""
                    )
                artifact_manifest = Path(command[command.index("--manifest") + 1])
                declared = yaml.safe_load(artifact_manifest.read_text(encoding="utf-8"))[
                    "artifacts"
                ][0]
                result = {
                    "status": "PUBLISHED",
                    "remote_verified": True,
                    "remote_uri": f"hf://buckets/{declared['storage']['bucket']}/{declared['storage']['path']}",
                    "sha256": declared["sha256"],
                    "size_bytes": declared["size_bytes"],
                }
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(result), stderr="")

            result = publish_run_splat(
                manifest,
                bucket="k4fka/artifacts",
                hf_cache_hub_root=hf_root,
                runner=runner,
            )
            self.assertEqual("published", result["status"])
            self.assertTrue(result["remote_verified"])
            self.assertEqual(sha, result["sha256"])
            self.assertEqual("a" * 40, result["source_revision"])
            artifact_manifest = yaml.safe_load(
                (manifest.parent / "artifact-manifest.yaml").read_text(encoding="utf-8")
            )
            artifact = artifact_manifest["artifacts"][0]
            self.assertEqual("gaussian-splat", artifact["kind"])
            self.assertEqual("ply", artifact["format"])
            self.assertEqual(sha, artifact["sha256"])
            self.assertEqual("a" * 40, artifact["provenance"]["revision"])
            self.assertIn("run_id", artifact["provenance"])
            updated = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("success", updated["status"])
            self.assertEqual("published", updated["artifact_publish"]["status"])
            self.assertTrue(ply.exists())

    def test_publish_failure_preserves_local_run_for_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, _, hf_root = self._fixture(root)

            def runner(command, **kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout="b" * 40 + "\n", stderr=""
                    )
                return subprocess.CompletedProcess(
                    command, 1, stdout=json.dumps({"status": "FAILED"}), stderr=""
                )

            with self.assertRaises(ArtifactPublishError):
                publish_run_splat(
                    manifest, bucket="k4fka/artifacts", hf_cache_hub_root=hf_root, runner=runner
                )
            updated = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("success", updated["status"])
            self.assertEqual("failed", updated["artifact_publish"]["status"])
            self.assertTrue(ply.exists())

    def test_local_ply_mismatch_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, _, hf_root = self._fixture(root)
            ply.write_bytes(b"changed")
            with self.assertRaises(ArtifactPublishError):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    source_revision="c" * 40,
                )

    def test_generated_ply_remains_gitignored(self):
        ignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("output/**/runs/**/export/*.ply", ignore)


if __name__ == "__main__":
    unittest.main()
