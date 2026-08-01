import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from main import (
    calculate_similarity,
    cluster_images,
    extract_features,
    select_images,
)


class ImagePipelineTests(unittest.TestCase):
    def _write_image(self, path: Path, width: int, height: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)

    def test_feature_length_does_not_depend_on_source_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            self._write_image(first, 320, 240, 1)
            self._write_image(second, 800, 600, 2)
            self.assertEqual(
                extract_features(first).shape,
                extract_features(second).shape,
            )

    def test_similarity_handles_different_source_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            self._write_image(first, 320, 240, 3)
            self._write_image(second, 640, 480, 4)
            score = calculate_similarity(first, second)
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(score, -1)
            self.assertLessEqual(score, 1)

    def test_selection_copies_without_removing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            output = root / "selected"
            self._write_image(source, 320, 240, 5)
            selected = select_images(
                [source],
                sharpness_threshold=0,
                similarity_threshold=0.99,
                output_dir=output,
            )
            self.assertTrue(source.exists())
            self.assertEqual(len(selected), 1)
            self.assertTrue(selected[0].exists())

    def test_empty_cluster_input_returns_empty_labels(self) -> None:
        labels = cluster_images([])
        self.assertEqual(labels.size, 0)


if __name__ == "__main__":
    unittest.main()
