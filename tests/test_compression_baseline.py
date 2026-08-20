import json
import tempfile
import unittest
from pathlib import Path

from processing.compression_baseline import (
    build_timestamp_split,
    compression_command,
    write_named_split_transforms,
)


class CompressionBaselineTests(unittest.TestCase):
    def test_compression_command_changes_only_explicit_h264_condition(self):
        command = compression_command("source.webm", "out.mp4", crf=35)
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-crf") + 1], "35")
        self.assertEqual(command[command.index("-preset") + 1], "medium")
        self.assertIn("-an", command)

    def test_timestamp_split_is_independent_of_compressed_frame_bytes(self):
        names = [f"frame-{index:06d}.jpg" for index in range(1, 11)]
        first = build_timestamp_split("a" * 64, names, fps=1 / 3, holdout_count=2)
        second = build_timestamp_split("a" * 64, names, fps=1 / 3, holdout_count=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first["holdout_frame_names"]), 2)
        self.assertEqual(len(first["train_frame_names"]), 8)
        self.assertTrue(
            set(first["holdout_frame_names"]).isdisjoint(first["train_frame_names"])
        )

    def test_named_split_uses_same_frame_names_in_each_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [
                {
                    "file_path": f"images/frame-{index:06d}.jpg",
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
                for index in range(1, 6)
            ]
            transforms = root / "transforms.json"
            transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            split = build_timestamp_split(
                "b" * 64,
                [Path(frame["file_path"]).name for frame in frames],
                fps=1 / 3,
                holdout_count=1,
            )
            result = write_named_split_transforms(
                transforms,
                split,
                root / "evaluation-transforms.json",
            )
            output = json.loads(Path(result["transforms_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(output["train_filenames"]), 4)
            self.assertEqual(len(output["test_filenames"]), 1)
            self.assertEqual(output["val_filenames"], output["test_filenames"])
            self.assertEqual(
                {Path(path).name for path in output["test_filenames"]},
                set(split["holdout_frame_names"]),
            )

    def test_invalid_crf_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CRF"):
            compression_command("source.webm", "out.mp4", crf=60)


if __name__ == "__main__":
    unittest.main()
