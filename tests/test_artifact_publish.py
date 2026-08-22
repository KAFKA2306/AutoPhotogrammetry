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
        transforms = manifest.parent / "nerfstudio-data" / "transforms.json"
        transforms.parent.mkdir()
        transforms_payload: dict[str, object] = {
            "frames": [
                {
                    "file_path": "images/frame-000001.jpg",
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
            ]
        }
        if orientation_override is not None:
            transforms_payload["orientation_override"] = orientation_override
        transforms.write_text(json.dumps(transforms_payload), encoding="utf-8")
        body: dict[str, object] = {
            "schema_version": 2,
            "dataset": "demo",
            "status": "success",
            "started_at": "2026-08-20T00:00:00Z",
            "source_revision": REVISION,
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
        if with_physical_up:
            physical_path = manifest.parent / "physical-up-evidence.json"
            physical_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "authority_type": "imu_gravity",
                        "authority_source": "fixture://imu.json",
                        "authority_source_sha256": "c" * 64,
                        "source_frame": "imu-frame",
                        "vector_semantics": "gravity_down",
                        "source_vector": [0.0, 0.5, -0.8660254037844386],
                        "source_to_model_matrix3x3": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "angular_uncertainty_deg": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            body["physical_up_evidence_path"] = physical_path.name
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

    def test_hf_cache_hub_python_can_be_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            script = root / "scripts" / "artifact_manager.py"
            script.parent.mkdir()
            script.write_text("# cli", encoding="utf-8")
            with patch.dict(os.environ, {"HF_CACHE_HUB_PYTHON": "/opt/hf/bin/python"}, clear=True):
                self.assertEqual(
                    ["/opt/hf/bin/python", str(script)],
                    _hf_cache_command(root),
                )

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
            self.assertEqual("accepted", result["orientation_status"])
            self.assertEqual("coordinate_basis_only", result["orientation_scope"])
            self.assertEqual("review_required", result["physical_up_status"])
            self.assertIsNone(result["physical_up_authority_type"])
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
            self.assertEqual("accepted", artifact["orientation"]["status"])
            self.assertEqual("review_required", artifact["orientation"]["physical_up"]["status"])
            self.assertEqual(sha, artifact["orientation"]["ply_sha256"])
            self.assertEqual("orientation-evidence.json", artifact["orientation"]["evidence_path"])
            self.assertEqual(
                [2**-0.5, 0.0, 0.0, 2**-0.5],
                artifact["orientation"]["consumer_application"]["quaternion_xyzw"],
            )
            updated = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("success", updated["status"])
            self.assertEqual("accepted", updated["orientation"]["status"])
            self.assertEqual("published", updated["artifact_publish"]["status"])
            self.assertTrue((manifest.parent / "orientation-evidence.json").is_file())
            self.assertTrue(ply.exists())

    def test_external_physical_up_is_published_with_portable_evidence_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root, with_physical_up=True)
            result = publish_run_splat(
                manifest,
                bucket="k4fka/artifacts",
                hf_cache_hub_root=hf_root,
                runner=self._successful_runner,
            )
            self.assertEqual("coordinate_basis_plus_physical_up", result["orientation_scope"])
            self.assertEqual("accepted", result["physical_up_status"])
            self.assertEqual("imu_gravity", result["physical_up_authority_type"])
            artifact = yaml.safe_load(
                (manifest.parent / "artifact-manifest.yaml").read_text(encoding="utf-8")
            )["artifacts"][0]
            physical = artifact["orientation"]["physical_up"]
            self.assertEqual("accepted", physical["status"])
            self.assertEqual("physical-up-evidence.json", physical["evidence_path"])
            self.assertEqual("c" * 64, physical["authority_source_sha256"])

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

    def test_missing_generation_revision_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root)
            body = json.loads(manifest.read_text(encoding="utf-8"))
            body.pop("source_revision")
            manifest.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactPublishError, "must record generation-time source_revision"):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
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
            self.assertEqual("accepted", updated["orientation"]["status"])
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

    def test_missing_transforms_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root)
            (manifest.parent / "nerfstudio-data" / "transforms.json").unlink()
            with self.assertRaisesRegex(ArtifactPublishError, "transforms.json"):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    runner=self._successful_runner,
                )

    def test_declared_missing_physical_up_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root)
            body = json.loads(manifest.read_text(encoding="utf-8"))
            body["physical_up_evidence_path"] = "missing-physical-up.json"
            manifest.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactPublishError, "physical-up evidence file is missing"
            ):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    runner=self._successful_runner,
                )

    def test_non_gravity_orientation_method_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, _, hf_root = self._fixture(root, orientation_override="pca")
            with self.assertRaisesRegex(
                ArtifactPublishError, "only accepted orientation basis evidence"
            ):
                publish_run_splat(
                    manifest,
                    bucket="k4fka/artifacts",
                    hf_cache_hub_root=hf_root,
                    runner=self._successful_runner,
                )

    def test_generated_ply_remains_gitignored(self):
        ignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("output/**/runs/**/export/*.ply", ignore)


if __name__ == "__main__":
    unittest.main()
