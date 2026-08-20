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


REVISION = "a" * 40


class ArtifactPublishTest(unittest.TestCase):
    def _fixture(self, root: Path, *, include_revision: bool = True, container_path: bool = False):
        ply = root / "output" / "demo" / "runs" / "r1" / "export" / "splat.ply"
        ply.parent.mkdir(parents=True)
        payload = b"ply\nsynthetic"
        ply.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        manifest = root / "output" / "demo" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "schema_version": 2,
            "dataset": "demo",
            "status": "success",
            "started_at": "2026-08-20T00:00:00Z",
            "registry": {
                "source_page": "https://commons.wikimedia.org/wiki/File:Demo.webm",
                "license": {"url": "https://creativecommons.org/publicdomain/zero/1.0/"},
            },
            "splatfacto": {
                "ply_path": (
                    "/workspace/output/demo/runs/r1/export/splat.ply"
                    if container_path
                    else str(ply)
                ),
                "ply_sha256": sha,
                "ply_size_bytes": len(payload),
            },
        }
        if include_revision:
            body["source_revision"] = REVISION
        manifest.write_text(json.dumps(body), encoding="utf-8")
        hf_root = root / "hf-cache-hub"
        (hf_root / "scripts").mkdir(parents=True)
        (hf_root / "scripts" / "artifact_manager.py").write_text("# cli", encoding="utf-8")
        return manifest, ply, sha, hf_root

    @staticmethod
    def _successful_runner(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            raise AssertionError("publish must not infer provenance from current HEAD")
        artifact_manifest = Path(command[command.index("--manifest") + 1])
        declared = yaml.safe_load(artifact_manifest.read_text(encoding="utf-8"))["artifacts"][0]
        result = {
            "status": "PUBLISHED",
            "remote_verified": True,
            "remote_uri": f"hf://buckets/{declared['storage']['bucket']}/{declared['storage']['path']}",
            "sha256": declared["sha256"],
            "size_bytes": declared["size_bytes"],
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(result), stderr="")

    def test_missing_hf_cache_hub_root_fails_with_required_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ArtifactPublishError, "HF_CACHE_HUB_ROOT"):
                _hf_cache_command(None)

    def test_publish_uses_generation_time_revision_without_git_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, sha, hf_root = self._fixture(root)
            result = publish_run_splat(
                manifest,
                bucket="k4fka/artifacts",
                hf_cache_hub_root=hf_root,
                runner=self._successful_runner,
            )
            self.assertEqual("published", result["status"])
            self.assertTrue(result["remote_verified"])
            self.assertEqual(sha, result["sha256"])
            self.assertEqual(REVISION, result["source_revision"])
            artifact_manifest = yaml.safe_load(
                (manifest.parent / "artifact-manifest.yaml").read_text(encoding="utf-8")
            )
            artifact = artifact_manifest["artifacts"][0]
            self.assertEqual("gaussian-splat", artifact["kind"])
            self.assertEqual("ply", artifact["format"])
            self.assertEqual(sha, artifact["sha256"])
            self.assertEqual(REVISION, artifact["provenance"]["revision"])
            self.assertIn("run_id", artifact["provenance"])
            self.assertNotIn("source_path", artifact["provenance"])
            updated = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("success", updated["status"])
            self.assertEqual("published", updated["artifact_publish"]["status"])
            self.assertTrue(ply.exists())

    def test_container_output_path_is_resolved_on_host(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, sha, hf_root = self._fixture(root, container_path=True)
            result = publish_run_splat(
                manifest,
                bucket="k4fka/artifacts",
                hf_cache_hub_root=hf_root,
                runner=self._successful_runner,
            )
            self.assertEqual(sha, result["sha256"])
            self.assertTrue(ply.exists())

    def test_legacy_manifest_requires_explicit_audited_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root, include_revision=False)
            with self.assertRaisesRegex(ArtifactPublishError, "audited legacy run"):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    runner=self._successful_runner,
                )
            result = publish_run_splat(
                manifest,
                bucket="k4fka/artifacts",
                hf_cache_hub_root=hf_root,
                source_revision="b" * 40,
                runner=self._successful_runner,
            )
            self.assertEqual("b" * 40, result["source_revision"])

    def test_explicit_revision_cannot_override_recorded_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root)
            with self.assertRaisesRegex(ArtifactPublishError, "does not match"):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    source_revision="b" * 40,
                    runner=self._successful_runner,
                )

    def test_publish_failure_preserves_local_run_for_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, ply, _, hf_root = self._fixture(root)

            def runner(command, **kwargs):
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=json.dumps({"status": "FAILED"}),
                    stderr="",
                )

            with self.assertRaises(ArtifactPublishError):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    runner=runner,
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
                )

    def test_generated_ply_remains_gitignored(self):
        ignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("output/**/runs/**/export/*.ply", ignore)


if __name__ == "__main__":
    unittest.main()
