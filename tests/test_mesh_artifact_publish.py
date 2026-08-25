from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from processing.artifact_publish import ArtifactPublishError
from processing.mesh_artifact_publish import (
    build_mesh_artifact_manifest,
    publish_mesh_export,
)

REVISION = "a" * 40
SOURCE_GAUSSIAN_SHA = "b" * 64
RAW_MESH_SHA = "c" * 64


class MeshArtifactPublishTest(unittest.TestCase):
    def _fixture(self, root: Path, *, output_format: str = "glb"):
        mesh = root / f"asset.{output_format}"
        payload = b"mesh-artifact"
        mesh.write_bytes(payload)
        mesh_sha = hashlib.sha256(payload).hexdigest()
        manifest = root / f"{output_format}-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "success",
                    "input": {"sha256": "d" * 64},
                    "output": {
                        "format": output_format,
                        "files": [
                            {
                                "path": str(mesh),
                                "sha256": mesh_sha,
                                "size_bytes": len(payload),
                            }
                        ],
                        "readback": {"vertex_count": 10, "face_count": 12},
                    },
                    "transform": {
                        "coordinate_frame": "source-mesh-unmodified",
                        "metric_scale": {"status": "unverified", "unit": None},
                    },
                }
            ),
            encoding="utf-8",
        )
        hf_root = root / "hf-cache-hub"
        (hf_root / "scripts").mkdir(parents=True)
        (hf_root / "scripts" / "artifact_manager.py").write_text("# cli", encoding="utf-8")
        return manifest, mesh, mesh_sha, len(payload), hf_root

    @staticmethod
    def _runner(command, **kwargs):
        artifact_manifest = Path(command[command.index("--manifest") + 1])
        artifact = yaml.safe_load(artifact_manifest.read_text(encoding="utf-8"))["artifacts"][0]
        result = {
            "status": "PUBLISHED",
            "remote_verified": True,
            "remote_uri": f"hf://buckets/{artifact['storage']['bucket']}/{artifact['storage']['path']}",
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(result), stderr="")

    def test_glb_manifest_preserves_mesh_lineage_and_license(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, mesh_sha, mesh_size, _ = self._fixture(root)
            artifact_manifest, artifact_id, _ = build_mesh_artifact_manifest(
                manifest,
                dataset="demo",
                bucket="k4fka/kafka-data-lake",
                source_revision=REVISION,
                source_gaussian_sha256=SOURCE_GAUSSIAN_SHA,
                raw_mesh_sha256=RAW_MESH_SHA,
                source_url="https://commons.wikimedia.org/wiki/File:Demo.webm",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            artifact = artifact_manifest["artifacts"][0]
            self.assertEqual("autophotogrammetry/demo/mesh/glb", artifact_id)
            self.assertEqual("mesh", artifact["kind"])
            self.assertEqual("glb", artifact["format"])
            self.assertFalse(artifact["generated"])
            self.assertEqual(mesh_sha, artifact["sha256"])
            self.assertEqual(mesh_size, artifact["size_bytes"])
            self.assertEqual(SOURCE_GAUSSIAN_SHA, artifact["provenance"]["source_gaussian_sha256"])
            self.assertEqual(RAW_MESH_SHA, artifact["provenance"]["raw_mesh_sha256"])
            self.assertEqual("d" * 64, artifact["provenance"]["parent_mesh_sha256"])
            self.assertEqual("unverified", artifact["geometry"]["metric_scale"]["status"])
            self.assertEqual("https://creativecommons.org/licenses/by/4.0/", artifact["license_url"])

    def test_publish_requires_remote_hash_and_size_readback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, _, mesh_sha, mesh_size, hf_root = self._fixture(root)
            result = publish_mesh_export(
                manifest,
                dataset="demo",
                bucket="k4fka/kafka-data-lake",
                source_revision=REVISION,
                source_gaussian_sha256=SOURCE_GAUSSIAN_SHA,
                raw_mesh_sha256=RAW_MESH_SHA,
                hf_cache_hub_root=hf_root,
                runner=self._runner,
            )
            self.assertEqual("published", result["status"])
            self.assertEqual(mesh_sha, result["sha256"])
            self.assertEqual(mesh_size, result["size_bytes"])
            self.assertTrue(result["remote_verified"])
            published_manifest = yaml.safe_load(
                (root / "glb-artifact-manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual("mesh", published_manifest["artifacts"][0]["kind"])

    def test_local_mesh_drift_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, mesh, _, _, hf_root = self._fixture(root)
            mesh.write_bytes(b"changed")
            with self.assertRaisesRegex(ArtifactPublishError, "no longer matches"):
                publish_mesh_export(
                    manifest,
                    dataset="demo",
                    bucket="k4fka/kafka-data-lake",
                    source_revision=REVISION,
                    source_gaussian_sha256=SOURCE_GAUSSIAN_SHA,
                    raw_mesh_sha256=RAW_MESH_SHA,
                    hf_cache_hub_root=hf_root,
                    runner=self._runner,
                )

    def test_obj_group_is_rejected_until_group_publish_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, _, _, _, _ = self._fixture(Path(d), output_format="obj")
            with self.assertRaisesRegex(ArtifactPublishError, "single-file GLB or STL"):
                build_mesh_artifact_manifest(
                    manifest,
                    dataset="demo",
                    bucket="k4fka/kafka-data-lake",
                    source_revision=REVISION,
                    source_gaussian_sha256=SOURCE_GAUSSIAN_SHA,
                    raw_mesh_sha256=RAW_MESH_SHA,
                )

    def test_invalid_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, _, _, _, _ = self._fixture(Path(d))
            with self.assertRaisesRegex(ArtifactPublishError, "source_gaussian_sha256"):
                build_mesh_artifact_manifest(
                    manifest,
                    dataset="demo",
                    bucket="k4fka/kafka-data-lake",
                    source_revision=REVISION,
                    source_gaussian_sha256="bad",
                    raw_mesh_sha256=RAW_MESH_SHA,
                )


if __name__ == "__main__":
    unittest.main()
