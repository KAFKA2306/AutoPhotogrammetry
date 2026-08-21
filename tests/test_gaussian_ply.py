import tempfile
import unittest
from pathlib import Path

import numpy as np

from processing.gaussian_ply import (
    gaussian_ply_inspection,
    gaussian_ply_metrics,
    validate_gaussian_ply_backend,
)


class GaussianPlyMetricsTests(unittest.TestCase):
    def _write_ply(
        self,
        path: Path,
        *,
        include_rotation: bool = False,
        include_xyz: bool = True,
        nonfinite_x: bool = False,
    ) -> None:
        fields = []
        if include_xyz:
            fields.extend([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        fields.extend(
            [
                ("f_dc_0", "<f4"),
                ("f_dc_1", "<f4"),
                ("f_dc_2", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        if include_rotation:
            fields.extend((f"rot_{index}", "<f4") for index in range(4))
        dtype = np.dtype(fields)
        vertices = np.zeros(2, dtype=dtype)
        if include_xyz:
            vertices[0]["x"] = np.float32(np.nan if nonfinite_x else -1.0)
            vertices[1]["x"] = np.float32(3.0)
            vertices[1]["y"] = np.float32(2.0)
            vertices[1]["z"] = np.float32(1.0)
        vertices[0]["opacity"] = np.float32(-3.0)
        vertices[1]["opacity"] = np.float32(3.0)
        vertices[0]["scale_0"] = np.float32(0.0)
        vertices[0]["scale_1"] = np.float32(0.0)
        vertices[0]["scale_2"] = np.float32(0.0)
        vertices[1]["scale_0"] = np.float32(0.0)
        vertices[1]["scale_1"] = np.float32(0.0)
        vertices[1]["scale_2"] = np.float32(3.0)
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
        ]
        for name in dtype.names or ():
            header_lines.append(f"property float {name}")
        header_lines.append("end_header")
        header = ("\n".join(header_lines) + "\n").encode("ascii")
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

    def test_inspection_records_fields_bounds_hash_and_unknown_dialect(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splat.ply"
            self._write_ply(path, include_rotation=True)
            result = gaussian_ply_inspection(path)
            self.assertEqual(result["encoding"], "binary_little_endian")
            self.assertEqual(result["vertex_count"], 2)
            self.assertEqual(result["dialect"], "unknown")
            self.assertEqual(result["position"]["bbox_min"], [-1.0, 0.0, 0.0])
            self.assertEqual(result["position"]["bbox_max"], [3.0, 2.0, 1.0])
            self.assertEqual(result["position"]["centroid"], [1.0, 1.0, 0.5])
            self.assertTrue(result["gaussian_fields"]["rotation"])
            self.assertEqual(result["gaussian_fields"]["inferred_sh_degree"], 0)
            self.assertEqual(len(result["sha256"]), 64)

    def test_inspection_rejects_missing_xyz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-xyz.ply"
            self._write_ply(path, include_xyz=False)
            with self.assertRaisesRegex(ValueError, "position properties"):
                gaussian_ply_inspection(path)

    def test_inspection_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nan.ply"
            self._write_ply(path, nonfinite_x=True)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                gaussian_ply_inspection(path)

    def test_backend_validator_reports_missing_fields_without_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splat.ply"
            self._write_ply(path)
            inspection = gaussian_ply_inspection(path)
            poisson = validate_gaussian_ply_backend(inspection, "point-cloud-poisson")
            covariance = validate_gaussian_ply_backend(
                inspection, "gaussian-covariance-normal"
            )
            tsdf = validate_gaussian_ply_backend(inspection, "render-depth-tsdf")
            self.assertTrue(poisson["supported"])
            self.assertFalse(covariance["supported"])
            self.assertEqual(
                covariance["missing_fields"],
                ["rot_0", "rot_1", "rot_2", "rot_3"],
            )
            self.assertFalse(tsdf["supported"])
            self.assertIn("checkpoint", tsdf["reason"])

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
