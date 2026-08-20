import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.quality_followup import (
    culling_train_args,
    run_budget_sweep,
    run_culling_sweep,
)


class QualityFollowupTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path]:
        source = root / "source.webm"
        source.write_bytes(b"video")
        data = root / "data"
        images = data / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(6):
            image = images / f"frame-{index:03d}.jpg"
            image.write_bytes(f"image-{index}".encode())
            frames.append(
                {
                    "file_path": f"images/{image.name}",
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
            )
        (data / "transforms.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        return source, data

    def _fake_experiment(self, **kwargs):
        name = kwargs["name"]
        return (
            {"name": name, "status": "success", "metrics": {}},
            {
                "schema_version": 1,
                "dataset_id": __import__("processing.backend_evaluation", fromlist=["dataset_identity"]).dataset_identity(kwargs["dataset"]),
                "backend": {"name": name, "upstream_revision": "revision"},
                "command": ["ns-train", "splatfacto"],
                "config": dict(kwargs["config"]),
                "return_code": 0,
                "status": "success",
                "failure_phase": None,
                "artifact": {"path": "artifact.ply", "format": "ply", "size_bytes": 1, "sha256": "a" * 64},
                "metrics": {},
            },
        )

    def test_culling_changes_only_one_named_parameter_after_winner(self):
        base = culling_train_args(winner="mcmc", iterations=30000)
        changed = culling_train_args(
            winner="mcmc",
            iterations=30000,
            parameter="cull_alpha_thresh",
            value=0.05,
        )
        self.assertEqual(changed[: len(base)], base)
        self.assertEqual(
            changed[len(base) :],
            ("--pipeline.model.cull-alpha-thresh", "0.05"),
        )

    def test_culling_sweep_runs_baseline_plus_each_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, data = self._dataset(root)
            with patch("processing.quality_followup.verify_gpu_runtime", return_value={"gpu": "test"}), patch(
                "processing.quality_followup.verify_research_environment",
                return_value={"nerfstudio_revision": "revision"},
            ), patch(
                "processing.quality_followup.run_splatfacto_experiment",
                side_effect=self._fake_experiment,
            ) as runner:
                result = run_culling_sweep(
                    data,
                    source,
                    root / "out",
                    winner="scale-regularized",
                    parameter="cull_scale_thresh",
                    values=[0.25, 0.5],
                    holdout_count=1,
                )
        self.assertEqual(runner.call_count, 3)
        self.assertEqual(len(result["comparison"]["results"]), 3)
        self.assertTrue(result["all_experiments_succeeded"])

    def test_budget_sweep_requires_two_budgets_and_freezes_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, data = self._dataset(root)
            with patch("processing.quality_followup.verify_gpu_runtime", return_value={"gpu": "test"}), patch(
                "processing.quality_followup.verify_research_environment",
                return_value={"nerfstudio_revision": "revision"},
            ), patch(
                "processing.quality_followup.run_splatfacto_experiment",
                side_effect=self._fake_experiment,
            ) as runner:
                result = run_budget_sweep(
                    data,
                    source,
                    root / "out",
                    winner="default",
                    budgets=[30000, 60000],
                    holdout_count=1,
                )
        self.assertEqual(runner.call_count, 2)
        first_args = runner.call_args_list[0].kwargs["train_args"]
        second_args = runner.call_args_list[1].kwargs["train_args"]
        self.assertIn("30000", first_args)
        self.assertIn("60000", second_args)
        self.assertEqual(len(result["comparison"]["results"]), 2)


if __name__ == "__main__":
    unittest.main()
