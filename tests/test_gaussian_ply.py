import tempfile
import unittest
from pathlib import Path

import numpy as np

from processing.gaussian_ply import gaussian_ply_metrics


class GaussianPlyMetricsTests(unittest.TestCase):
    def _write_ply(self, path: Path) -> None:
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        vertices = np.zeros(2, dtype=dtype)
        vertices[0]["opacity"] = np.float32(-3.0)
        vertices[1]["opacity"] = np.float32(3.0)
        vertices[0]["scale_0"] = np.float32(0.0)
        vertices[0]["scale_1"] = np.float32(0.0)
        vertices[0]["scale_2"] = np.float32(0.0)
        vertices[1]["scale_0"] = np.float32(0.0)
        vertices[1]["scale_1"] = np.float32(0.0)
        vertices[1]["scale_2"] = np.float32(3.0)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.tobytes())

    def test_metrics_measure_low_opacity_and_spiky_gaussians(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splat.ply"
            self._write_ply(path)
            result = gaussian_ply_metrics(path)
            self.assertEqual(result["primitive_count"], 2)
            self.assertEqual(result["opacity"]["below_0_1_count"], 1)
            self.assertAlmostEqual(result["opacity"]["below_0_1_ratio"], 0.5)
            self.assertEqual(result["scale_anisotropy_ratio"]["above_10_count"], 1)
            self.assertAlmostEqual(result["scale_anisotropy_ratio"]["above_10_ratio"], 0.5)
            self.assertEqual(len(result["sha256"]), 64)

    def test_metrics_fail_closed_for_ascii_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ascii.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float opacity\nend_header\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "binary_little_endian"):
                gaussian_ply_metrics(path)


if __name__ == "__main__":
    unittest.main()
