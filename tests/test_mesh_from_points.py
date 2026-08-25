from __future__ import annotations

import unittest

import numpy as np

from processing.mesh_from_points import _point_samples


def _vertices(opacity: list[float] | None = None) -> np.ndarray:
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if opacity is not None:
        fields.append(("opacity", "f4"))
    values = np.zeros(len(opacity or [0, 0, 0, 0]), dtype=np.dtype(fields))
    values["x"] = np.arange(len(values), dtype=np.float32)
    values["y"] = 1
    values["z"] = 2
    if opacity is not None:
        values["opacity"] = opacity
    return values


class MeshPointSamplesTests(unittest.TestCase):
    def test_without_opacity_filter_keeps_all_points(self) -> None:
        points, filtering = _point_samples(_vertices([0.0, 0.0, 0.0, 0.0]), None)
        self.assertEqual(len(points), 4)
        self.assertEqual(
            filtering,
            {
                "input_point_count": 4,
                "kept_point_count": 4,
                "filtered_point_count": 0,
                "opacity_threshold": None,
            },
        )

    def test_filters_using_sigmoid_opacity(self) -> None:
        points, filtering = _point_samples(_vertices([-3.0, 0.0, 1.0, 2.0, 3.0]), 0.5)
        self.assertEqual(len(points), 4)
        self.assertEqual(points[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(filtering["filtered_point_count"], 1)
        self.assertEqual(filtering["opacity_threshold"], 0.5)

    def test_rejects_invalid_or_unsupported_filter(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            _point_samples(_vertices([0.0, 0.0, 0.0, 0.0]), 1.1)
        with self.assertRaisesRegex(ValueError, "requires an opacity property"):
            _point_samples(_vertices(), 0.5)

    def test_rejects_too_few_remaining_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "too few point samples"):
            _point_samples(_vertices([-10.0, -10.0, -10.0, 10.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
