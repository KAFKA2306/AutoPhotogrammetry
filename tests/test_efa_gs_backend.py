import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.backend_evaluation import build_nerfstudio_dataset_contract, dataset_identity
from processing.efa_gs_backend import EFA_REVISION, run_efa_gs


class EfaGsBackendTests(unittest.TestCase):
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
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
            )
        transforms = data / "transforms.json"
        transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
        dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=1)
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        colmap = root / "colmap" / "0"
        colmap.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (colmap / name).write_bytes(name.encode())

        checkout = root / "efa"
        for relative in ("3DGS/train.py", "3DGS/render.py", "3DGS/metrics.py", "LICENSE.md"):
            path = checkout / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return dataset, dataset_path, transforms, colmap, checkout

    def test_runs_official_efa_sequence_and_preserves_native_metrics_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_path, transforms, colmap, checkout = self._fixture(root)

            def fake_step(command, *, cwd, log_root, name, **kwargs):
                model = root / "out" / "model"
                model.mkdir(parents=True, exist_ok=True)
                if name == "train":
                    ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                    ply.parent.mkdir(parents=True, exist_ok=True)
                    ply.write_bytes(b"ply")
                if name == "metrics":
                    (model / "results.json").write_text(
                        json.dumps({"ours_30000": {"PSNR": 22.0, "SSIM": 0.81, "LPIPS": 0.19}}),
                        encoding="utf-8",
                    )
                return {
                    "name": name,
                    "command": list(command),
                    "return_code": 0,
                    "wall_clock_seconds": 2.0,
                    "peak_gpu_memory_bytes": 200 if name == "train" else 80,
                    "stdout_log": str(root / f"{name}.out"),
                    "stderr_log": str(root / f"{name}.err"),
                }

            metrics = {
                "primitive_count": 20,
                "output_size_bytes": 3,
                "low_opacity_primitive_count": 2,
                "low_opacity_primitive_ratio": 0.1,
                "scale_anisotropy_above_10_count": 3,
                "scale_anisotropy_above_10_ratio": 0.15,
            }
            with (
                patch("processing.external_research.git_head", return_value=EFA_REVISION),
                patch("processing.efa_gs_backend.run_recorded_gpu_step", side_effect=fake_step),
                patch("processing.efa_gs_backend.gaussian_artifact_metrics", return_value=metrics),
            ):
                result = run_efa_gs(
                    dataset_path,
                    transforms,
                    colmap,
                    checkout,
                    root / "out",
                )

            self.assertEqual(result["dataset_id"], dataset_identity(dataset))
            self.assertFalse(result["config"]["production_eligible"])
            self.assertEqual(result["metrics"]["peak_gpu_memory_bytes"], 200)
            self.assertIsNone(result["metrics"]["psnr"])
            self.assertIsNotNone(result["native_metrics"])
            self.assertEqual(result["artifact"]["format"], "ply")
            self.assertEqual(len(result["steps"]), 3)

    def test_invalid_iteration_count_fails_before_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, dataset_path, transforms, colmap, checkout = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "positive"):
                run_efa_gs(
                    dataset_path,
                    transforms,
                    colmap,
                    checkout,
                    root / "out",
                    iterations=0,
                )


if __name__ == "__main__":
    unittest.main()
