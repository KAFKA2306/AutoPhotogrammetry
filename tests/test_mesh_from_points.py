from __future__ import annotations

import numpy as np
import pytest

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


def test_point_samples_without_opacity_filter_keeps_all_points() -> None:
    points, filtering = _point_samples(_vertices([0.0, 0.0, 0.0, 0.0]), None)
    assert len(points) == 4
    assert filtering == {
        "input_point_count": 4,
        "kept_point_count": 4,
        "filtered_point_count": 0,
        "opacity_threshold": None,
    }


def test_point_samples_filters_using_sigmoid_opacity() -> None:
    points, filtering = _point_samples(_vertices([-3.0, 0.0, 1.0, 2.0, 3.0]), 0.5)
    assert len(points) == 4
    assert points[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert filtering["filtered_point_count"] == 1
    assert filtering["opacity_threshold"] == 0.5


def test_point_samples_rejects_invalid_or_unsupported_filter() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _point_samples(_vertices([0.0, 0.0, 0.0, 0.0]), 1.1)
    with pytest.raises(ValueError, match="requires an opacity property"):
        _point_samples(_vertices(), 0.5)


def test_point_samples_rejects_too_few_remaining_points() -> None:
    with pytest.raises(ValueError, match="too few point samples"):
        _point_samples(_vertices([-10.0, -10.0, -10.0, 10.0]), 0.5)
