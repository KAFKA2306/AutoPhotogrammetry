import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.backend_evaluation import build_nerfstudio_dataset_contract, dataset_identity
from processing.external_research import materialize_frozen_images, verify_checkout


class ExternalResearchTests(unittest.TestCase):
    def _dataset(self, root: Path):
        source = root / "source.webm"
        source.write_bytes(b"video")
        data = root / "data"
        images = data / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(3):
            image = images / f"frame-{index:03d}.jpg"
            image.write_bytes(f"image-{index}".encode())
            frames.append(
                {
                    "file_path": f"images/{image.name}",
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
            )
        transforms = data / "transforms.json"
        transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
        dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=1)
        return dataset, transforms

    def test_materialize_preserves_exact_frozen_bytes_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, transforms = self._dataset(root)
            result = materialize_frozen_images(dataset, transforms, root / "out")
            self.assertEqual(result["dataset_id"], dataset_identity(dataset))
            self.assertEqual(result["frame_count"], 3)
            self.assertEqual(
                {row["sha256"] for row in result["frames"]},
                {row["sha256"] for row in dataset["frames"]},
            )
            self.assertEqual(sum(row["split"] == "holdout" for row in result["frames"]), 1)

    def test_checkout_requires_exact_revision_and_required_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.py").write_text("", encoding="utf-8")
            with patch("processing.external_research.git_head", return_value="abc"):
                result = verify_checkout(root, expected_revision="abc", required_paths=("train.py",))
            self.assertEqual(result["revision"], "abc")
            with patch("processing.external_research.git_head", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "revision mismatch"):
                    verify_checkout(root, expected_revision="abc")


if __name__ == "__main__":
    unittest.main()
