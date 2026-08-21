from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from processing.video_sources import EVALUATION_STAGES, load_video_registry

REQUIRED_HOLDOUT_METRICS = ("psnr", "ssim", "lpips")


def stage_d_measurements(backend_result: Mapping[str, Any]) -> dict[str, Any]:
    """Map a successful quality-sweep backend result to the canonical Stage-D fields."""
    if backend_result.get("status") != "success":
        raise ValueError("Stage D requires a successful backend result")
    metrics = backend_result.get("metrics")
    artifact = backend_result.get("artifact")
    if not isinstance(metrics, Mapping):
        raise ValueError("Stage D backend result requires metrics")
    if metrics.get("reconstruction_success") is not True:
        raise ValueError("Stage D requires reconstruction_success=true")
    holdout_count = metrics.get("holdout_frame_count")
    if not isinstance(holdout_count, int) or holdout_count < 1:
        raise ValueError("Stage D requires at least one deterministic hold-out frame")
    missing = [
        key for key in REQUIRED_HOLDOUT_METRICS if not isinstance(metrics.get(key), (int, float))
    ]
    if missing:
        raise ValueError(f"Stage D is missing hold-out metrics: {missing}")
    if not isinstance(artifact, Mapping) or artifact.get("format") != "ply":
        raise ValueError("Stage D requires a PLY artifact")
    sha256 = artifact.get("sha256")
    size_bytes = artifact.get("size_bytes")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("Stage D PLY requires SHA-256")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("Stage D PLY requires positive size_bytes")
    return {
        "train_success": True,
        "export_success": True,
        "holdout_psnr": float(metrics["psnr"]),
        "holdout_ssim": float(metrics["ssim"]),
        "holdout_lpips": float(metrics["lpips"]),
        "ply_sha256": sha256,
        "ply_size_bytes": size_bytes,
    }


def apply_stage_d_to_registry(
    registry_path: str | Path,
    source_id: str,
    backend_result: Mapping[str, Any],
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = load_video_registry(registry_file)
    source = next((item for item in registry["videos"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"unknown video source: {source_id}")
    if source["measurements"].get("colmap") is None:
        raise ValueError(f"{source_id}: Stage D evidence requires Stage C evidence first")
    measurements = stage_d_measurements(backend_result)
    source["measurements"]["splat"] = measurements
    metrics = backend_result["metrics"]
    artifact = backend_result["artifact"]
    source["splat_evidence"] = {
        "dataset_id": backend_result.get("dataset_id"),
        "backend": backend_result.get("backend"),
        "holdout_frame_count": metrics.get("holdout_frame_count"),
        "train_frame_count": metrics.get("train_frame_count"),
        "artifact_path": artifact.get("path"),
        "training_manifest_path": backend_result.get("training_manifest_path"),
        "evaluation_manifest_path": backend_result.get("evaluation_manifest_path"),
    }
    current_stage = source["evaluation_stage"]
    if EVALUATION_STAGES.index(current_stage) < EVALUATION_STAGES.index("splat"):
        source["evaluation_stage"] = "splat"
    registry_file.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def load_and_apply_stage_d(
    registry_path: str | Path,
    source_id: str,
    backend_result_path: str | Path,
) -> dict[str, Any]:
    result = json.loads(Path(backend_result_path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("backend result must be a JSON object")
    return apply_stage_d_to_registry(registry_path, source_id, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Stage D only from a successful deterministic hold-out backend result."
    )
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--backend-result", required=True)
    args = parser.parse_args()
    registry = load_and_apply_stage_d(args.registry, args.source_id, args.backend_result)
    source = next(video for video in registry["videos"] if video["id"] == args.source_id)
    print(json.dumps(source["measurements"]["splat"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
