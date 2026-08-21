import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from processing.video_preflight import (
    PREFLIGHT_FIELDS,
    apply_preflight_to_registry,
    measure_frames,
)


class VideoPreflightTests(unittest.TestCase):
    def _image(self, path: Path, array: np.ndarray) -> None:
        Image.fromarray(array.astype(np.uint8), mode="L").save(path)

    def _registry(self, path: Path, *, verified: bool = True) -> None:
        registry = {
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
                    "evaluation_stage": "metadata",
                    "source_page": "https://example.test/source",
                    "media_url": "https://example.test/video.webm",
                    "author": "Example",
                    "duration_seconds": 120,
                    "resolution": [1920, 1080],
                    "license": {
                        "name": "CC0",
                        "status": "verified" if verified else "needs_review",
                        "url": "https://example.test/license" if verified else None,
                    },
                    "measurements": {
                        "preflight": None,
                        "colmap": None,
                        "splat": None,
                    },
                }
            ],
        }
        path.write_text(json.dumps(registry), encoding="utf-8")

    def test_identical_textured_frames_have_no_scene_cut_and_no_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            texture = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
            paths = []
            for index in range(3):
                path = root / f"frame-{index}.png"
                self._image(path, texture)
                paths.append(path)
            result = measure_frames(paths)

        self.assertEqual(result["metrics"]["scene_cut_count"], 0)
        self.assertLessEqual(result["metrics"]["dynamic_pixel_ratio"], 0.01)
        self.assertEqual(set(result["metrics"]), set(PREFLIGHT_FIELDS))
        self.assertNotIn("score", result)
        self.assertNotIn("rank", result)
        self.assertNotIn("expected_success", result)

    def test_discontinuous_frames_increment_scene_cut_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.png"
            second = root / "second.png"
            self._image(first, np.zeros((96, 96), dtype=np.uint8))
            self._image(second, np.full((96, 96), 255, dtype=np.uint8))
            result = measure_frames([first, second], scene_ssim_threshold=0.5)
        self.assertEqual(result["metrics"]["scene_cut_count"], 1)
        self.assertGreater(result["metrics"]["dynamic_pixel_ratio"], 0.9)

    def test_registry_update_advances_only_to_preflight_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "videos.json"
            self._registry(registry)
            result = {
                "status": "success",
                "metrics": {
                    "scene_cut_count": 0,
                    "sharp_frame_ratio": 0.9,
                    "adjacent_view_overlap": 0.5,
                    "camera_translation_proxy": 0.02,
                    "dynamic_pixel_ratio": 0.1,
                    "exposure_variation": 0.03,
                },
                "shot_evidence": {
                    "selected_shot_id": "shot-0002",
                    "selected_start_seconds": 10.0,
                    "selected_end_seconds": 40.0,
                    "selected_duration_seconds": 30.0,
                    "selection_basis": ["geometry_pair_count desc"],
                    "shots": [
                        {
                            "id": "shot-0002",
                            "start_seconds": 10.0,
                            "end_seconds": 40.0,
                        }
                    ],
                },
            }
            updated = apply_preflight_to_registry(registry, "scene", result)

        source = updated["videos"][0]
        self.assertEqual(source["evaluation_stage"], "preflight")
        self.assertEqual(source["measurements"]["preflight"], result["metrics"])
        self.assertEqual(source["preflight_evidence"]["selected_shot_id"], "shot-0002")
        self.assertFalse({"rank", "score", "expected_success"}.intersection(source))

    def test_registry_update_requires_completed_metadata_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "videos.json"
            self._registry(registry, verified=False)
            result = {
                "status": "success",
                "metrics": {field: 0 for field in PREFLIGHT_FIELDS},
            }
            with self.assertRaisesRegex(ValueError, "metadata gate"):
                apply_preflight_to_registry(registry, "scene", result)


if __name__ == "__main__":
    unittest.main()
