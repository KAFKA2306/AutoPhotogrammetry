import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.backend_evaluation import build_nerfstudio_dataset_contract, dataset_identity
from processing.longsplat_backend import LONGSPLAT_REVISION, run_longsplat


class LongSplatBackendTests(unittest.TestCase):
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
        checkout = root / "longsplat"
        for relative in (
            "train.py",
            "render.py",
            "metrics.py",
            "convert_3dgs.py",
            "scripts/train_custom.sh",
            "LICENSE.md",
        ):
            path = checkout / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return dataset, dataset_path, transforms, checkout

    def test_runs_official_sequence_and_records_research_only_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_path, transforms, checkout = self._fixture(root)

            def fake_step(command, *, cwd, log_root, name, **kwargs):
                output_root = root / "out"
                model = output_root / "model"
                model.mkdir(parents=True, exist_ok=True)
                if name == "metrics":
                    (model / "results.json").write_text(
                        json.dumps({"ours": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}}),
                        encoding="utf-8",
                    )
                if name == "convert-3dgs":
                    converted = model / "converted_3dgs"
                    converted.mkdir(parents=True, exist_ok=True)
                    (converted / "point_cloud.ply").write_bytes(b"ply")
                return {
                    "name": name,
                    "command": list(command),
                    "return_code": 0,
                    "wall_clock_seconds": 1.0,
                    "peak_gpu_memory_bytes": 100 if name == "train" else 50,
                    "stdout_log": str(root / f"{name}.out"),
                    "stderr_log": str(root / f"{name}.err"),
                }

            metrics = {
                "primitive_count": 10,
                "output_size_bytes": 3,
                "low_opacity_primitive_count": 1,
                "low_opacity_primitive_ratio": 0.1,
                "scale_anisotropy_above_10_count": 2,
                "scale_anisotropy_above_10_ratio": 0.2,
            }
            with patch("processing.external_research.git_head", return_value=LONGSPLAT_REVISION), patch(
                "processing.longsplat_backend.run_recorded_gpu_step", side_effect=fake_step
            ), patch(
                "processing.longsplat_backend.gaussian_artifact_metrics", return_value=metrics
            ):
                result = run_longsplat(
                    dataset_path,
                    transforms,
                    checkout,
                    root / "out",
                )

            self.assertEqual(result["dataset_id"], dataset_identity(dataset))
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["config"]["production_eligible"])
            self.assertIsNone(result["metrics"]["psnr"])
            self.assertEqual(result["metrics"]["peak_gpu_memory_bytes"], 100)
            self.assertEqual(len(result["steps"]), 4)
            self.assertEqual(result["artifact"]["format"], "ply")
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_rejects_invalid_prune_ratio_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, dataset_path, transforms, checkout = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "prune_ratio"):
                run_longsplat(
                    dataset_path,
                    transforms,
                    checkout,
                    root / "out",
                    prune_ratio=1.0,
                )


if __name__ == "__main__":
    unittest.main()
