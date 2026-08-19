import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.splatfacto_quality import quality_train_args, run_quality_comparison


class SplatfactoQualityTests(unittest.TestCase):
    def test_quality_train_args_change_only_scale_regularization(self):
        baseline = quality_train_args(iterations=2000, scale_regularization=False)
        regularized = quality_train_args(iterations=2000, scale_regularization=True)
        self.assertEqual(
            baseline,
            (
                "--max-num-iterations",
                "2000",
                "--viewer.quit-on-train-completion",
                "True",
            ),
        )
        self.assertEqual(regularized[: len(baseline)], baseline)
        self.assertEqual(
            regularized[len(baseline) :],
            ("--pipeline.model.use-scale-regularization", "True"),
        )

    def test_quality_comparison_runs_same_budget_with_and_without_regularization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            calls = []

            def fake_run(data_dir, output_root, **kwargs):
                calls.append((Path(data_dir), Path(output_root), kwargs))
                run_dir = Path(output_root) / "splatfacto" / "run"
                export_dir = run_dir / "export"
                export_dir.mkdir(parents=True)
                (export_dir / "splat.ply").write_bytes(b"ply")
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

            with patch(
                "processing.splatfacto_quality.run_splatfacto_export",
                side_effect=fake_run,
            ):
                result = run_quality_comparison(
                    data,
                    root / "comparison",
                    iterations=(2000,),
                )

            self.assertEqual(len(calls), 2)
            baseline_args = calls[0][2]["train_extra_args"]
            regularized_args = calls[1][2]["train_extra_args"]
            self.assertNotIn("--pipeline.model.use-scale-regularization", baseline_args)
            self.assertIn("--pipeline.model.use-scale-regularization", regularized_args)
            self.assertEqual(len(result["experiments"]), 2)
            self.assertFalse(result["experiments"][0]["scale_regularization"])
            self.assertTrue(result["experiments"][1]["scale_regularization"])
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_quality_comparison_rejects_invalid_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            with self.assertRaisesRegex(ValueError, "positive"):
                run_quality_comparison(data, Path(tmp) / "out", iterations=(0,))


if __name__ == "__main__":
    unittest.main()
