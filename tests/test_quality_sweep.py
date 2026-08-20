import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.quality_sweep import quality_sweep_train_args, run_quality_sweep


class QualitySweepTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path]:
        source = root / "source.webm"
        source.write_bytes(b"video")
        data = root / "data"
        images = data / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(10):
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
        (data / "transforms.json").write_text(
            json.dumps({"frames": frames}),
            encoding="utf-8",
        )
        return data, source

    def test_variants_change_only_expected_training_argument(self):
        baseline = quality_sweep_train_args(iterations=30000, variant="default")
        scale = quality_sweep_train_args(iterations=30000, variant="scale-regularized")
        mcmc = quality_sweep_train_args(iterations=30000, variant="mcmc")

        self.assertEqual(scale[: len(baseline)], baseline)
        self.assertEqual(
            scale[len(baseline) :],
            ("--pipeline.model.use-scale-regularization", "True"),
        )
        self.assertEqual(mcmc[: len(baseline)], baseline)
        self.assertEqual(
            mcmc[len(baseline) :],
            ("--pipeline.model.strategy", "mcmc"),
        )

    def test_quality_sweep_records_common_dataset_eval_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source = self._dataset(root)
            calls = []

            def fake_run(data_path, output_root, **kwargs):
                calls.append((Path(data_path), kwargs["train_extra_args"]))
                run_dir = Path(output_root) / "splatfacto" / "run"
                (run_dir / "export").mkdir(parents=True)
                ply = run_dir / "export" / "splat.ply"
                ply.write_bytes(b"ply")
                config = run_dir / "config.yml"
                config.write_text("config", encoding="utf-8")
                manifest = run_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "training": {
                        "command": ["ns-train", "splatfacto"],
                        "config_path": "config.yml",
                        "peak_gpu_memory_bytes": 123456,
                    },
                    "output": {
                        "ply_path": "export/splat.ply",
                        "sha256": "a" * 64,
                        "size_bytes": 3,
                    },
                }

            def fake_eval(config_path, output_root, **kwargs):
                eval_root = Path(output_root)
                eval_root.mkdir(parents=True)
                manifest = eval_root / "eval-manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "metrics": {"psnr": 25.0, "ssim": 0.9, "lpips": 0.1},
                    "render_count": 2,
                    "renders": [{"path": "0.png", "sha256": "b" * 64, "size_bytes": 1}],
                }

            environment = {"nerfstudio_revision": "revision", "gsplat_version": "1.4.0"}
            runtime = {
                "gpu_name": "test-gpu",
                "compute_capability": "12.0",
                "torch_version": "2.7.1",
                "torch_cuda_version": "12.8",
                "container_image_ref": "image",
                "container_image_id": "sha256:image",
            }
            metrics = {
                "primitive_count": 100,
                "output_size_bytes": 3,
                "low_opacity_primitive_count": 10,
                "low_opacity_primitive_ratio": 0.1,
                "scale_anisotropy_above_10_count": 20,
                "scale_anisotropy_above_10_ratio": 0.2,
            }
            with (
                patch(
                    "processing.quality_sweep.verify_gpu_runtime",
                    return_value=runtime,
                ),
                patch(
                    "processing.quality_sweep.verify_research_environment",
                    return_value=environment,
                ),
                patch(
                    "processing.quality_sweep.run_splatfacto_export",
                    side_effect=fake_run,
                ),
                patch(
                    "processing.quality_sweep.run_nerfstudio_eval",
                    side_effect=fake_eval,
                ),
                patch(
                    "processing.quality_sweep.gaussian_artifact_metrics",
                    return_value=metrics,
                ),
            ):
                result = run_quality_sweep(
                    data,
                    source,
                    root / "out",
                    nerfstudio_source=root / "nerfstudio",
                    iterations=30000,
                    holdout_count=2,
                )

            self.assertEqual(len(calls), 3)
            self.assertTrue(all(path.name == "evaluation-transforms.json" for path, _ in calls))
            self.assertEqual(
                [entry["variant"] for entry in result["variants"]],
                ["default", "scale-regularized", "mcmc"],
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["train_frame_sha256"]), 8)
            self.assertEqual(len(result["holdout_frame_sha256"]), 2)
            self.assertEqual(result["variants"][0]["metrics"]["psnr"], 25.0)
            self.assertEqual(result["variants"][0]["metrics"]["ssim"], 0.9)
            self.assertEqual(result["variants"][0]["metrics"]["lpips"], 0.1)
            self.assertEqual(result["variants"][0]["metrics"]["peak_gpu_memory_bytes"], 123456)
            self.assertEqual(result["variants"][0]["render_count"], 2)
            self.assertEqual(len(result["comparison"]["results"]), 3)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertTrue(Path(result["dataset_manifest_path"]).is_file())
            self.assertTrue(Path(result["split_transforms_path"]).is_file())

    def test_quality_sweep_attempts_remaining_variants_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source = self._dataset(root)
            calls = []

            def fake_run(data_path, output_root, **kwargs):
                args = tuple(kwargs["train_extra_args"])
                calls.append(args)
                variant_root = Path(output_root)
                if "--pipeline.model.use-scale-regularization" in args:
                    run_dir = variant_root / "splatfacto" / "failed"
                    run_dir.mkdir(parents=True)
                    (run_dir / "manifest.json").write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "failed_phase": "training",
                                "training": {
                                    "command": ["ns-train", "splatfacto", "--scale"],
                                    "return_code": 7,
                                    "peak_gpu_memory_bytes": 987654,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise subprocess.CalledProcessError(7, ["ns-train", "splatfacto", "--scale"])

                run_dir = variant_root / "splatfacto" / "run"
                (run_dir / "export").mkdir(parents=True)
                (run_dir / "export" / "splat.ply").write_bytes(b"ply")
                (run_dir / "config.yml").write_text("config", encoding="utf-8")
                manifest = run_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "training": {
                        "command": ["ns-train", "splatfacto"],
                        "config_path": "config.yml",
                        "peak_gpu_memory_bytes": 123,
                    },
                    "output": {
                        "ply_path": "export/splat.ply",
                        "sha256": "a" * 64,
                        "size_bytes": 3,
                    },
                }

            def fake_eval(config_path, output_root, **kwargs):
                eval_root = Path(output_root)
                eval_root.mkdir(parents=True)
                manifest = eval_root / "eval-manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "metrics": {"psnr": 25.0, "ssim": 0.9, "lpips": 0.1},
                    "render_count": 1,
                    "renders": [{"path": "0.png", "sha256": "b" * 64, "size_bytes": 1}],
                }

            environment = {"nerfstudio_revision": "revision", "gsplat_version": "1.4.0"}
            runtime = {
                "gpu_name": "test-gpu",
                "compute_capability": "12.0",
                "torch_version": "2.7.1",
                "torch_cuda_version": "12.8",
                "container_image_ref": "image",
                "container_image_id": "sha256:image",
            }
            ply_metrics = {
                "primitive_count": 100,
                "output_size_bytes": 3,
                "low_opacity_primitive_count": 10,
                "low_opacity_primitive_ratio": 0.1,
                "scale_anisotropy_above_10_count": 20,
                "scale_anisotropy_above_10_ratio": 0.2,
            }
            output_root = root / "out"
            with (
                patch(
                    "processing.quality_sweep.verify_gpu_runtime",
                    return_value=runtime,
                ),
                patch(
                    "processing.quality_sweep.verify_research_environment",
                    return_value=environment,
                ),
                patch(
                    "processing.quality_sweep.run_splatfacto_export",
                    side_effect=fake_run,
                ),
                patch(
                    "processing.quality_sweep.run_nerfstudio_eval",
                    side_effect=fake_eval,
                ),
                patch(
                    "processing.quality_sweep.gaussian_artifact_metrics",
                    return_value=ply_metrics,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "scale-regularized"):
                    run_quality_sweep(
                        data,
                        source,
                        output_root,
                        nerfstudio_source=root / "nerfstudio",
                        iterations=30000,
                        holdout_count=2,
                    )

            self.assertEqual(len(calls), 3)
            summary = json.loads((output_root / "quality-sweep.json").read_text())
            comparison = json.loads((output_root / "comparison.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failed_variants"], ["scale-regularized"])
            self.assertEqual(
                [entry["status"] for entry in summary["variants"]],
                ["success", "failed", "success"],
            )
            self.assertEqual(summary["variants"][1]["metrics"]["peak_gpu_memory_bytes"], 987654)
            self.assertEqual(len(comparison["results"]), 3)
            self.assertEqual(
                [row["status"] for row in comparison["results"]],
                ["success", "failed", "success"],
            )


if __name__ == "__main__":
    unittest.main()
