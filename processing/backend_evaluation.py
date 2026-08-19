from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from processing.gaussian_ply import gaussian_ply_metrics
from processing.provenance import image_records, sha256_file, write_json

METRIC_FIELDS = (
    "reconstruction_success",
    "input_frame_count",
    "train_frame_count",
    "holdout_frame_count",
    "psnr",
    "ssim",
    "lpips",
    "wall_clock_seconds",
    "peak_gpu_memory_bytes",
    "output_size_bytes",
    "primitive_count",
    "camera_pose_available",
    "low_opacity_primitive_count",
    "low_opacity_primitive_ratio",
    "scale_anisotropy_above_10_count",
    "scale_anisotropy_above_10_ratio",
    "cleanup_removed_primitive_count",
    "cleanup_removed_primitive_ratio",
)


def _stable_score(source_sha256: str, frame_sha256: str) -> str:
    return hashlib.sha256(f"{source_sha256}:{frame_sha256}".encode("ascii")).hexdigest()


def build_dataset_contract(
    source_video: str | Path,
    frame_dir: str | Path,
    *,
    holdout_count: int,
) -> dict:
    """Create a content-addressed, deterministic train/hold-out split."""
    video = Path(source_video).expanduser().resolve()
    frames = Path(frame_dir).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"source video does not exist: {video}")
    if not frames.is_dir():
        raise ValueError(f"frame directory does not exist: {frames}")

    records = image_records(frames)
    if len(records) < 2:
        raise ValueError("at least two frames are required")
    if holdout_count < 1 or holdout_count >= len(records):
        raise ValueError("holdout_count must be between 1 and frame_count - 1")

    source_sha256 = sha256_file(video)
    ranked = sorted(
        records,
        key=lambda record: (
            _stable_score(source_sha256, record["sha256"]),
            record["path"],
        ),
    )
    holdout_hashes = {record["sha256"] for record in ranked[:holdout_count]}
    annotated = [
        {**record, "split": "holdout" if record["sha256"] in holdout_hashes else "train"}
        for record in records
    ]

    return {
        "schema_version": 1,
        "source_video": {
            "path": video.name,
            "size_bytes": video.stat().st_size,
            "sha256": source_sha256,
        },
        "frames": annotated,
        "train_frame_sha256": [
            record["sha256"] for record in annotated if record["split"] == "train"
        ],
        "holdout_frame_sha256": [
            record["sha256"] for record in annotated if record["split"] == "holdout"
        ],
    }


def dataset_identity(contract: Mapping) -> str:
    """Return a stable identity for the source and exact train/hold-out frame sets."""
    payload = {
        "source_video_sha256": contract["source_video"]["sha256"],
        "train_frame_sha256": sorted(contract["train_frame_sha256"]),
        "holdout_frame_sha256": sorted(contract["holdout_frame_sha256"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def empty_metrics() -> dict:
    """Return the common metric shape without inventing unmeasured values."""
    return {field: None for field in METRIC_FIELDS}


def gaussian_artifact_metrics(path: str | Path) -> dict:
    """Return direct, representation-level artifact measurements for a Gaussian PLY.

    These values do not classify visual quality. They expose low-opacity primitives and
    highly anisotropic primitives so those counts can be compared alongside fixed-view
    renders and image-space quality metrics.
    """
    measured = gaussian_ply_metrics(path)
    return {
        "primitive_count": measured["primitive_count"],
        "output_size_bytes": measured["size_bytes"],
        "low_opacity_primitive_count": measured["opacity"]["below_0_1_count"],
        "low_opacity_primitive_ratio": measured["opacity"]["below_0_1_ratio"],
        "scale_anisotropy_above_10_count": measured["scale_anisotropy_ratio"]["above_10_count"],
        "scale_anisotropy_above_10_ratio": measured["scale_anisotropy_ratio"]["above_10_ratio"],
    }


def cleanup_metrics(before_path: str | Path, after_path: str | Path) -> dict:
    """Measure primitive removal by cleanup without claiming that removed means improved."""
    before = gaussian_ply_metrics(before_path)
    after = gaussian_ply_metrics(after_path)
    removed = before["primitive_count"] - after["primitive_count"]
    if removed < 0:
        raise ValueError(
            "cleanup output contains more primitives than its input; use backend-specific metrics instead"
        )
    return {
        "cleanup_removed_primitive_count": removed,
        "cleanup_removed_primitive_ratio": removed / before["primitive_count"],
    }


def artifact_record(path: str | Path, *, format: str) -> dict:
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise ValueError(f"output artifact does not exist: {artifact}")
    return {
        "path": str(artifact),
        "format": format,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def validate_backend_result(result: Mapping, dataset: Mapping) -> None:
    if result.get("dataset_id") != dataset_identity(dataset):
        raise ValueError("backend result does not match the dataset contract")

    backend = result.get("backend") or {}
    if not backend.get("name"):
        raise ValueError("backend name is required")
    if not backend.get("upstream_revision"):
        raise ValueError("backend upstream revision/version is required")
    if not isinstance(result.get("command"), list) or not result["command"]:
        raise ValueError("backend command must be a non-empty argv list")
    if "return_code" not in result:
        raise ValueError("backend return_code is required")

    status = result.get("status")
    if status not in {"success", "failed"}:
        raise ValueError("backend status must be success or failed")
    if status == "failed" and not result.get("failure_phase"):
        raise ValueError("failed backend result requires failure_phase")

    artifact = result.get("artifact")
    if status == "success":
        if not artifact:
            raise ValueError("successful backend result requires an artifact")
        if not artifact.get("format") or not artifact.get("sha256"):
            raise ValueError("successful artifact requires format and sha256")
        if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] <= 0:
            raise ValueError("successful artifact requires positive size_bytes")

    metrics = result.get("metrics") or {}
    unknown = set(metrics) - set(METRIC_FIELDS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    for key, value in metrics.items():
        if value is None:
            continue
        if key == "reconstruction_success" or key == "camera_pose_available":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be bool or null")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric or null")


def compare_backend_results(results: Sequence[Mapping], dataset: Mapping) -> list[dict]:
    """Generate table rows whose values remain traceable to each result manifest."""
    rows = []
    for result in results:
        validate_backend_result(result, dataset)
        metrics = {**empty_metrics(), **(result.get("metrics") or {})}
        rows.append(
            {
                "backend": result["backend"]["name"],
                "upstream_revision": result["backend"]["upstream_revision"],
                "status": result["status"],
                "failure_phase": result.get("failure_phase"),
                "artifact_format": (result.get("artifact") or {}).get("format"),
                "artifact_sha256": (result.get("artifact") or {}).get("sha256"),
                **metrics,
            }
        )
    return rows


def write_comparison(
    output_path: str | Path,
    results: Sequence[Mapping],
    dataset: Mapping,
) -> dict:
    value = {
        "schema_version": 1,
        "dataset_id": dataset_identity(dataset),
        "source_video_sha256": dataset["source_video"]["sha256"],
        "train_frame_sha256": list(dataset["train_frame_sha256"]),
        "holdout_frame_sha256": list(dataset["holdout_frame_sha256"]),
        "results": compare_backend_results(results, dataset),
    }
    write_json(output_path, value)
    return value
