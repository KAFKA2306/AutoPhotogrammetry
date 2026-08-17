import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from processing.image_selection import (
    calculate_similarity,
    select_images,
    select_video_frames,
)


class ImageSelectionTests(unittest.TestCase):
    def _write_image(self, path: Path, width: int, height: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)

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

    def test_video_selection_is_ordered_and_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(4):
                path = root / f"frame-{index}.jpg"
                path.write_bytes(str(index).encode())
                frames.append(path)

            sharpness = {
                frames[0]: 1.0,
                frames[1]: 0.0,
                frames[2]: 1.0,
                frames[3]: 1.0,
            }
            similarity_calls = []

            def similarity(left, right):
                similarity_calls.append((Path(left), Path(right)))
                return 0.95 if Path(left) == frames[2] else 0.1

            result = select_video_frames(
                frames,
                root / "selected",
                sharpness_threshold=0.5,
                similarity_threshold=0.92,
                sharpness_fn=lambda path: sharpness[Path(path)],
                similarity_fn=similarity,
            )

            self.assertEqual(result["input"], 4)
            self.assertEqual(result["selected"], 2)
            self.assertEqual(result["rejected_blur"], 1)
            self.assertEqual(result["rejected_duplicate"], 1)
            self.assertEqual(len(similarity_calls), 2)


if __name__ == "__main__":
    unittest.main()
