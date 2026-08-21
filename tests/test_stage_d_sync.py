import json
import tempfile
import unittest
from pathlib import Path

from processing.stage_d_sync import apply_stage_d_to_registry, stage_d_measurements


def _registry() -> dict:
    return {
        "schema_version": 2,
        "default": "scene",
        "evaluation_policy": {
            "stages": {
                "metadata": {},
                "preflight": {},
                "colmap": {},
                "splat": {},
            }
        },
        "videos": [
            {
                "id": "scene",
                "evaluation_stage": "colmap",
                "measurements": {
                    "preflight": {"scene_cut_count": 0},
                    "colmap": {"registered_images": 10},
                    "splat": None,
                },
            }
        ],
    }


def _backend_result() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "dataset",
        "status": "success",
        "backend": {"name": "splatfacto-default", "upstream_revision": "abc"},
        "artifact": {
            "path": "/tmp/splat.ply",
            "format": "ply",
            "size_bytes": 123456,
            "sha256": "a" * 64,
        },
        "metrics": {
            "reconstruction_success": True,
            "train_frame_count": 18,
            "holdout_frame_count": 2,
            "psnr": 20.5,
            "ssim": 0.72,
            "lpips": 0.31,
        },
        "training_manifest_path": "train/manifest.json",
        "evaluation_manifest_path": "eval/manifest.json",
    }


class StageDSyncTests(unittest.TestCase):
    def test_stage_d_requires_holdout_metrics(self):
        result = _backend_result()
        result["metrics"]["psnr"] = None
        with self.assertRaisesRegex(ValueError, "hold-out metrics"):
            stage_d_measurements(result)

    def test_export_only_result_cannot_be_called_stage_d(self):
        result = _backend_result()
        result["metrics"]["holdout_frame_count"] = 0
        with self.assertRaisesRegex(ValueError, "hold-out frame"):
            stage_d_measurements(result)

    def test_successful_holdout_result_advances_registry_to_splat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "videos.json"
            path.write_text(json.dumps(_registry()), encoding="utf-8")
            updated = apply_stage_d_to_registry(path, "scene", _backend_result())
        source = updated["videos"][0]
        self.assertEqual(source["evaluation_stage"], "splat")
        self.assertEqual(source["measurements"]["splat"]["holdout_psnr"], 20.5)
        self.assertEqual(source["measurements"]["splat"]["ply_sha256"], "a" * 64)
        self.assertNotIn("score", source)
        self.assertNotIn("rank", source)


if __name__ == "__main__":
    unittest.main()
