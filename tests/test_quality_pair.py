import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.quality_pair import run_quality_pair


class QualityPairTests(unittest.TestCase):
    def _scene(self, root: Path, scene: str) -> None:
        source = root / "input" / scene / "source.webm"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video")
        data = root / "output" / scene / "nerfstudio-data"
        data.mkdir(parents=True)
        (data / "transforms.json").write_text('{"frames": []}', encoding="utf-8")

    def test_pair_requires_explicit_distinct_roles(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            run_quality_pair("same", "same")

    def test_pair_attempts_both_scenes_and_records_decision_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._scene(root, "bad")
            self._scene(root, "good")
            calls = []

            def fake_sweep(data, source, sweep_root, **kwargs):
                scene = Path(data).parent.name
                calls.append(scene)
                sweep_root = Path(sweep_root)
                sweep_root.mkdir(parents=True, exist_ok=True)
                comparison = sweep_root / "comparison.json"
                comparison.write_text('{"results": []}', encoding="utf-8")
                manifest = {
                    "status": "success",
                    "dataset_id": f"dataset-{scene}",
                    "comparison_path": str(comparison),
                    "failed_variants": [],
                }
                (sweep_root / "quality-sweep.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                return manifest

            with patch("processing.quality_pair.run_quality_sweep", side_effect=fake_sweep):
                result = run_quality_pair(
                    "bad",
                    "good",
                    input_root=root / "input",
                    output_root=root / "output",
                    iterations=30000,
                )

            self.assertEqual(calls, ["bad", "good"])
            self.assertEqual(result["roles"], {"bad_scene": "bad", "good_control": "good"})
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["winner_decision_inputs"]), 2)
            self.assertIsNone(result["selected_winner"])
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_first_failure_does_not_prevent_good_control_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._scene(root, "bad")
            self._scene(root, "good")
            calls = []

            def fake_sweep(data, source, sweep_root, **kwargs):
                scene = Path(data).parent.name
                calls.append(scene)
                sweep_root = Path(sweep_root)
                sweep_root.mkdir(parents=True, exist_ok=True)
                status = "failed" if scene == "bad" else "success"
                comparison = sweep_root / "comparison.json"
                comparison.write_text('{"results": []}', encoding="utf-8")
                (sweep_root / "quality-sweep.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "dataset_id": f"dataset-{scene}",
                            "comparison_path": str(comparison),
                            "failed_variants": ["default"] if status == "failed" else [],
                        }
                    ),
                    encoding="utf-8",
                )
                if status == "failed":
                    raise RuntimeError("bad failed")
                return {"status": status}

            with patch("processing.quality_pair.run_quality_sweep", side_effect=fake_sweep):
                result = run_quality_pair(
                    "bad",
                    "good",
                    input_root=root / "input",
                    output_root=root / "output",
                )

            self.assertEqual(calls, ["bad", "good"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                [entry["status"] for entry in result["scene_runs"]],
                ["failed", "success"],
            )


if __name__ == "__main__":
    unittest.main()
