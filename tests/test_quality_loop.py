from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from processing.provenance import sha256_file
from processing.quality_loop import evaluate_candidate, run_bounded_culling_loop


def row(**overrides):
    value = {
        "status": "success",
        "psnr": 30.0,
        "ssim": 0.90,
        "lpips": 0.10,
        "wall_clock_seconds": 100.0,
        "peak_gpu_memory_bytes": 1000.0,
        "low_opacity_primitive_ratio": 0.20,
        "scale_anisotropy_above_10_ratio": 0.10,
    }
    value.update(overrides)
    return value


def write_evaluation(root: Path, name: str, values: tuple[int, int]) -> dict:
    evaluation_root = root / name / "evaluation"
    render_root = evaluation_root / "renders"
    render_root.mkdir(parents=True)
    manifest_path = evaluation_root / "eval-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    records = []
    for index, value in enumerate(values):
        path = render_root / f"holdout-{index:02d}.png"
        Image.new("RGB", (32, 24), (value, value, value)).save(path)
        records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "name": name,
        "status": "success",
        "evaluation_manifest_path": str(manifest_path),
        "renders": records,
    }


class QualityLoopTests(unittest.TestCase):
    def test_candidate_requires_artifact_improvement_without_image_regression(self):
        decision = evaluate_candidate(
            row(),
            row(
                low_opacity_primitive_ratio=0.15,
                scale_anisotropy_above_10_ratio=0.08,
                psnr=29.8,
            ),
        )
        self.assertTrue(decision["eligible"])
        self.assertGreater(decision["score"], 0)

    def test_image_regression_rejects_artifact_gain(self):
        decision = evaluate_candidate(
            row(),
            row(
                low_opacity_primitive_ratio=0.10,
                scale_anisotropy_above_10_ratio=0.05,
                psnr=28.0,
            ),
        )
        self.assertFalse(decision["eligible"])
        self.assertIn("psnr_regression", decision["reasons"])

    def test_bounded_loop_updates_last_good_and_stops_after_no_improvement(self):
        calls = []

        def runner(data, source, out, **kwargs):
            value = float(kwargs["values"][0])
            calls.append(value)
            candidate = row(
                low_opacity_primitive_ratio=0.20 - value,
                scale_anisotropy_above_10_ratio=0.10,
            )
            if value >= 0.03:
                candidate = row(
                    low_opacity_primitive_ratio=0.20,
                    scale_anisotropy_above_10_ratio=0.10,
                )
            return {
                "dataset_id": "same-dataset",
                "comparison_path": str(Path(out) / "comparison.json"),
                "comparison": {
                    "dataset_id": "same-dataset",
                    "results": [row(), candidate],
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_bounded_culling_loop(
                "data",
                "source",
                Path(tmp) / "loop",
                winner="default",
                parameter="cull_alpha_thresh",
                values=[0.01, 0.02, 0.03, 0.04, 0.05],
                max_iterations=5,
                no_improvement_limit=2,
                runner=runner,
            )
        self.assertEqual(calls, [0.01, 0.02, 0.03, 0.04])
        self.assertEqual(result["stop_reason"], "no_improvement")
        self.assertEqual(result["last_good"]["value"], 0.02)

    def test_all_failed_is_distinct_from_no_improvement(self):
        calls = []

        def runner(data, source, out, **kwargs):
            value = float(kwargs["values"][0])
            calls.append(value)
            return {
                "dataset_id": "same-dataset",
                "comparison_path": str(Path(out) / "comparison.json"),
                "comparison": {
                    "dataset_id": "same-dataset",
                    "results": [row(), row(status="failed")],
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_bounded_culling_loop(
                "data",
                "source",
                Path(tmp) / "loop",
                winner="default",
                parameter="cull_alpha_thresh",
                values=[0.01, 0.02],
                max_iterations=4,
                no_improvement_limit=2,
                runner=runner,
            )

        self.assertEqual(calls, [0.01, 0.02])
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stop_reason"], "all_failed")

    def test_successful_sweep_writes_hash_bound_before_after_montage(self):
        def runner(data, source, out, **kwargs):
            root = Path(out)
            baseline = write_evaluation(root, "baseline", (24, 48))
            candidate_evidence = write_evaluation(root, "candidate", (32, 64))
            candidate = row(
                low_opacity_primitive_ratio=0.10,
                scale_anisotropy_above_10_ratio=0.08,
            )
            return {
                "dataset_id": "same-dataset",
                "comparison_path": str(root / "comparison.json"),
                "comparison": {
                    "dataset_id": "same-dataset",
                    "results": [row(), candidate],
                },
                "experiments": [baseline, candidate_evidence],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_bounded_culling_loop(
                "data",
                "source",
                Path(tmp) / "loop",
                winner="default",
                parameter="cull_alpha_thresh",
                values=[0.01],
                max_iterations=2,
                no_improvement_limit=2,
                runner=runner,
            )
            evidence = result["attempts"][0]["renderEvidence"]
            self.assertIsNotNone(evidence)
            assert evidence is not None
            montage = Path(evidence["path"])
            self.assertTrue(montage.is_file())
            self.assertEqual(evidence["sha256"], sha256_file(montage))
            self.assertEqual(evidence["layout"]["rows"], ["baseline", "candidate"])
            self.assertEqual(
                evidence["layout"]["columns"],
                ["holdout-00.png", "holdout-01.png"],
            )
            self.assertEqual(len(evidence["baseline_renders"]), 2)
            self.assertEqual(len(evidence["candidate_renders"]), 2)

    def test_resume_skips_already_attempted_value(self):
        calls = []

        def runner(data, source, out, **kwargs):
            value = float(kwargs["values"][0])
            calls.append(value)
            return {
                "dataset_id": "same-dataset",
                "comparison_path": str(Path(out) / "comparison.json"),
                "comparison": {
                    "dataset_id": "same-dataset",
                    "results": [
                        row(),
                        row(low_opacity_primitive_ratio=0.20 - value),
                    ],
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "loop"
            run_bounded_culling_loop(
                "data",
                "source",
                root,
                winner="default",
                parameter="cull_alpha_thresh",
                values=[0.01],
                max_iterations=3,
                no_improvement_limit=3,
                runner=runner,
            )
            run_bounded_culling_loop(
                "data",
                "source",
                root,
                winner="default",
                parameter="cull_alpha_thresh",
                values=[0.01, 0.02],
                max_iterations=3,
                no_improvement_limit=3,
                runner=runner,
            )
        self.assertEqual(calls, [0.01, 0.02])

    def test_dataset_drift_fails_loudly(self):
        count = 0

        def runner(data, source, out, **kwargs):
            nonlocal count
            count += 1
            dataset = f"d{count}"
            return {
                "dataset_id": dataset,
                "comparison": {
                    "dataset_id": dataset,
                    "results": [
                        row(),
                        row(low_opacity_primitive_ratio=0.1),
                    ],
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "dataset identity changed"):
                run_bounded_culling_loop(
                    "data",
                    "source",
                    Path(tmp) / "loop",
                    winner="default",
                    parameter="cull_alpha_thresh",
                    values=[0.01, 0.02],
                    max_iterations=2,
                    no_improvement_limit=2,
                    runner=runner,
                )


if __name__ == "__main__":
    unittest.main()
