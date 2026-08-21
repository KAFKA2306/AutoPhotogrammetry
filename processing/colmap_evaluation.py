from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def parse_model_analyzer(text: str) -> dict[str, int | float]:
    """Parse the stable numeric summary emitted by COLMAP model_analyzer."""
    patterns: dict[str, tuple[str, type[int] | type[float]]] = {
        "cameras": (r"Cameras:\s*(\d+)", int),
        "images": (r"Images:\s*(\d+)", int),
        "registered_images": (r"Registered images:\s*(\d+)", int),
        "points": (r"Points:\s*(\d+)", int),
        "observations": (r"Observations:\s*(\d+)", int),
        "mean_track_length": (r"Mean track length:\s*([0-9.eE+-]+)", float),
        "mean_observations_per_image": (
            r"Mean observations per image:\s*([0-9.eE+-]+)",
            float,
        ),
        "mean_reprojection_error_px": (
            r"Mean reprojection error:\s*([0-9.eE+-]+)\s*px",
            float,
        ),
    }
    metrics: dict[str, int | float] = {}
    for key, (pattern, caster) in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[key] = caster(match.group(1))
    return metrics


def model_directories(sparse_root: str | Path) -> list[Path]:
    root = Path(sparse_root)
    if not root.is_dir():
        return []
    candidates = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if any((directory / name).is_file() for name in ("images.bin", "images.txt")):
            candidates.append(directory)
    return candidates


def aggregate_models(
    models: Sequence[Mapping[str, Any]],
    *,
    input_images: int,
) -> dict[str, Any]:
    """Aggregate fragmented COLMAP models while preserving largest-model evidence."""
    if input_images <= 0:
        raise ValueError("input_images must be positive")
    usable = [
        dict(model)
        for model in models
        if int((model.get("metrics") or {}).get("registered_images") or 0) > 0
    ]
    if not usable:
        raise ValueError("COLMAP produced no registered model")
    usable.sort(
        key=lambda model: (
            -int(model["metrics"].get("registered_images") or 0),
            str(model.get("model_path") or ""),
        )
    )
    largest = usable[0]
    largest_metrics = largest["metrics"]
    total_registered = sum(int(model["metrics"].get("registered_images") or 0) for model in usable)
    if total_registered > input_images:
        raise ValueError(
            f"COLMAP model aggregation registered {total_registered} images from {input_images} inputs"
        )
    largest_registered = int(largest_metrics.get("registered_images") or 0)
    return {
        "input_images": input_images,
        "registered_images": total_registered,
        "registration_ratio": total_registered / input_images,
        "largest_model_registered_images": largest_registered,
        "largest_model_ratio": largest_registered / total_registered,
        "submodel_count": len(usable),
        "sparse_points": int(largest_metrics.get("points") or 0),
        "observations": int(largest_metrics.get("observations") or 0),
        "mean_track_length": largest_metrics.get("mean_track_length"),
        "mean_observations_per_image": largest_metrics.get("mean_observations_per_image"),
        "mean_reprojection_error_px": largest_metrics.get("mean_reprojection_error_px"),
        "largest_model_path": str(largest["model_path"]),
        "models": usable,
        "aggregation": {
            "registration_ratio": "sum registered images across non-empty submodels / input images",
            "largest_model_ratio": "registered images in largest submodel / total registered images",
            "quality_metrics": "points/track/reprojection metrics are reported from the largest submodel",
        },
    }
