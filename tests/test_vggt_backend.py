import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.backend_evaluation import build_nerfstudio_dataset_contract
from processing.vggt_backend import (
    VGGT_DEMO_CHECKPOINT_URL,
    VGGT_REVISION,
    materialize_vggt_scene,
    run_vggt_colmap,
    verify_vggt_checkout,
)


class VggtBackendTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source.webm"
        source.write_bytes(b"video")
        data = root / "data"
        images = data / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(4):
            image = images / f"frame-{index:03d}.jpg"
            image.write_bytes(f"image-{index}".encode())
            frames.append(
                {
                    "file_path": f"images/{image.name}",
                    "transform_matrix": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
            )
        transforms = data / "transforms.json"
        transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
        dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=1)
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        checkout = root / "vggt"
        checkout.mkdir()
        (checkout / "demo_colmap.py").write_text(
            f'_URL = "{VGGT_DEMO_CHECKPOINT_URL}"\n', encoding="utf-8"
        )
        return dataset, dataset_path, transforms, checkout

    def test_checkout_requires_exact_revision_and_research_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, checkout = self._fixture(root)
            with patch("processing.vggt_backend._git_head", return_value=VGGT_REVISION):
                info = verify_vggt_checkout(checkout)
            self.assertEqual(info["revision"], VGGT_REVISION)
            self.assertFalse(info["checkpoint"]["production_eligible"])
            self.assertEqual(info["checkpoint"]["usage"], "non-commercial-research-only")

            with patch("processing.vggt_backend._git_head", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "revision mismatch"):
                    verify_vggt_checkout(checkout)

    def test_materialization_preserves_dataset_hashes_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _, transforms, _ = self._fixture(root)
            manifest = materialize_vggt_scene(dataset, transforms, root / "scene")
            self.assertEqual(manifest["frame_count"], 4)
            self.assertEqual(
                {entry["sha256"] for entry in manifest["frames"]},
                {entry["sha256"] for entry in dataset["frames"]},
            )
            self.assertEqual(
                sum(entry["split"] == "holdout" for entry in manifest["frames"]),
                1,
            )

    def test_success_records_colmap_artifact_and_never_claims_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_path, transforms, checkout = self._fixture(root)

            def fake_run(command, *, cwd):
                scene = Path(command[command.index("--scene_dir") + 1])
                sparse = scene / "sparse"
                sparse.mkdir(parents=True)
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    (sparse / name).write_bytes(name.encode())
                (sparse / "points.ply").write_bytes(b"ply")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with (
                patch("processing.vggt_backend._git_head", return_value=VGGT_REVISION),
                patch("processing.vggt_backend._run_command", side_effect=fake_run),
            ):
                result = run_vggt_colmap(
                    dataset_path,
                    transforms,
                    checkout,
                    root / "out",
                )

            self.assertEqual(result["status"], "success")
            self.assertTrue(result["metrics"]["camera_pose_available"])
            self.assertEqual(result["artifact"]["format"], "colmap-sparse-manifest")
            self.assertFalse(result["config"]["production_eligible"])
            self.assertEqual(
                result["dataset_id"],
                __import__(
                    "processing.backend_evaluation", fromlist=["dataset_identity"]
                ).dataset_identity(dataset),
            )
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_command_failure_is_explicit_failed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, dataset_path, transforms, checkout = self._fixture(root)
            failed = subprocess.CompletedProcess(["python"], 9, "", "gpu failed")
            with (
                patch("processing.vggt_backend._git_head", return_value=VGGT_REVISION),
                patch("processing.vggt_backend._run_command", return_value=failed),
            ):
                result = run_vggt_colmap(
                    dataset_path,
                    transforms,
                    checkout,
                    root / "out",
                )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_phase"], "vggt-colmap")
            self.assertFalse(result["metrics"]["reconstruction_success"])
            self.assertIsNone(result["artifact"])


if __name__ == "__main__":
    unittest.main()
