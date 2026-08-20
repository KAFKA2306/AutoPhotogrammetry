import json
import tempfile
import unittest
from pathlib import Path

from processing.backend_evaluation import (
    artifact_record,
    build_dataset_contract,
    build_nerfstudio_dataset_contract,
    compare_backend_results,
    dataset_identity,
    empty_metrics,
    validate_backend_result,
    write_comparison,
    write_nerfstudio_split_transforms,
)


class BackendEvaluationTests(unittest.TestCase):
    def _dataset(self, root: Path):
        video = root / "source.webm"
        video.write_bytes(b"video")
        frames = root / "frames"
        frames.mkdir()
        for index in range(6):
            (frames / f"frame-{index:03d}.jpg").write_bytes(f"frame-{index}".encode())
        return build_dataset_contract(video, frames, holdout_count=2)

    def test_split_is_content_addressed_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._dataset(root)
            second = build_dataset_contract(root / "source.webm", root / "frames", holdout_count=2)
            self.assertEqual(first, second)
            self.assertEqual(len(first["train_frame_sha256"]), 4)
            self.assertEqual(len(first["holdout_frame_sha256"]), 2)
            self.assertTrue(
                set(first["train_frame_sha256"]).isdisjoint(first["holdout_frame_sha256"])
            )
            self.assertEqual(len(dataset_identity(first)), 64)

    def test_nerfstudio_contract_writes_explicit_split_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.webm"
            video.write_bytes(b"video")
            data = root / "nerfstudio-data"
            images = data / "images"
            images.mkdir(parents=True)
            frames = []
            for index in range(10):
                image = images / f"frame-{index:03d}.jpg"
                image.write_bytes(f"processed-{index}".encode())
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

            dataset = build_nerfstudio_dataset_contract(video, transforms, holdout_count=2)
            result = write_nerfstudio_split_transforms(
                transforms,
                dataset,
                root / "quality" / "evaluation-transforms.json",
            )
            split = json.loads(Path(result["transforms_path"]).read_text(encoding="utf-8"))
            original = json.loads(transforms.read_text(encoding="utf-8"))

            self.assertEqual(result["train_frame_count"], 8)
            self.assertEqual(result["holdout_frame_count"], 2)
            self.assertEqual(len(split["train_filenames"]), 8)
            self.assertEqual(split["val_filenames"], split["test_filenames"])
            self.assertEqual(len(split["val_filenames"]), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in split["train_filenames"]))
            self.assertNotIn("train_filenames", original)
            self.assertEqual(result["dataset_id"], dataset_identity(dataset))

    def test_success_requires_real_artifact_and_traceable_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            output = root / "model.sog"
            output.write_bytes(b"model")
            metrics = empty_metrics()
            metrics.update(
                {
                    "reconstruction_success": True,
                    "input_frame_count": 6,
                    "train_frame_count": 4,
                    "holdout_frame_count": 2,
                    "psnr": 27.5,
                    "wall_clock_seconds": 12.25,
                    "output_size_bytes": 5,
                    "camera_pose_available": True,
                }
            )
            result = {
                "schema_version": 1,
                "dataset_id": dataset_identity(dataset),
                "backend": {"name": "example", "upstream_revision": "abc123"},
                "command": ["example", "--config", "config.yml"],
                "config": {"quality": "test"},
                "started_at": "2026-08-19T00:00:00+00:00",
                "finished_at": "2026-08-19T00:00:12+00:00",
                "return_code": 0,
                "status": "success",
                "failure_phase": None,
                "artifact": artifact_record(output, format="sog"),
                "metrics": metrics,
            }
            validate_backend_result(result, dataset)
            rows = compare_backend_results([result], dataset)
            self.assertEqual(rows[0]["psnr"], 27.5)
            self.assertIsNone(rows[0]["ssim"])
            self.assertEqual(rows[0]["artifact_format"], "sog")

    def test_failed_backend_does_not_require_fake_artifact_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            failed = {
                "schema_version": 1,
                "dataset_id": dataset_identity(dataset),
                "backend": {"name": "broken", "upstream_revision": "deadbeef"},
                "command": ["broken"],
                "config": {},
                "started_at": "2026-08-19T00:00:00+00:00",
                "finished_at": "2026-08-19T00:00:01+00:00",
                "return_code": 2,
                "status": "failed",
                "failure_phase": "training",
                "artifact": None,
                "metrics": {},
            }
            validate_backend_result(failed, dataset)
            comparison = write_comparison(root / "comparison.json", [failed], dataset)
            self.assertEqual(comparison["results"][0]["status"], "failed")
            self.assertIsNone(comparison["results"][0]["psnr"])

    def test_dataset_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            result = {
                "dataset_id": "0" * 64,
                "backend": {"name": "example", "upstream_revision": "abc123"},
                "command": ["example"],
                "return_code": 1,
                "status": "failed",
                "failure_phase": "setup",
                "metrics": {},
            }
            with self.assertRaisesRegex(ValueError, "dataset contract"):
                validate_backend_result(result, dataset)

    def test_unknown_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            result = {
                "dataset_id": dataset_identity(dataset),
                "backend": {"name": "example", "upstream_revision": "abc123"},
                "command": ["example"],
                "return_code": 1,
                "status": "failed",
                "failure_phase": "setup",
                "metrics": {"made_up_score": 1.0},
            }
            with self.assertRaisesRegex(ValueError, "unknown metrics"):
                validate_backend_result(result, dataset)


if __name__ == "__main__":
    unittest.main()
