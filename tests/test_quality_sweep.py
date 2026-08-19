import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.quality_sweep import quality_sweep_train_args, run_quality_sweep


class QualitySweepTests(unittest.TestCase):
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

    def test_quality_sweep_records_three_variants_ply_metrics_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            calls = []

            def fake_run(data_dir, output_root, **kwargs):
                calls.append(kwargs["train_extra_args"])
                run_dir = Path(output_root) / "splatfacto" / "run"
                (run_dir / "export").mkdir(parents=True)
                ply = run_dir / "export" / "splat.ply"
                ply.write_bytes(b"ply")
                manifest = run_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "output": {
                        "ply_path": "export/splat.ply",
                        "sha256": "a" * 64,
                        "size_bytes": 3,
                    },
                }

            environment = {
                "nerfstudio_revision": "revision",
                "gsplat_version": "1.4.0",
            }
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
                "opacity": {"below_0_1_ratio": 0.1},
                "scale_anisotropy_ratio": {"above_10_ratio": 0.2},
            }
            with patch(
                "processing.quality_sweep.verify_gpu_runtime",
                return_value=runtime,
            ), patch(
                "processing.quality_sweep.verify_research_environment",
                return_value=environment,
            ), patch(
                "processing.quality_sweep.run_splatfacto_export",
                side_effect=fake_run,
            ), patch(
                "processing.quality_sweep.gaussian_ply_metrics",
                return_value=metrics,
            ):
                result = run_quality_sweep(
                    data,
                    root / "out",
                    nerfstudio_source=root / "nerfstudio",
                    iterations=30000,
                )

            self.assertEqual(len(calls), 3)
            self.assertEqual(
                [entry["variant"] for entry in result["variants"]],
                ["default", "scale-regularized", "mcmc"],
            )
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertEqual(result["variants"][0]["ply_metrics"], metrics)
            self.assertEqual(result["runtime"], runtime)


if __name__ == "__main__":
    unittest.main()
