from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image
from skimage.feature import ORB, match_descriptors
from skimage.filters import laplace
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from processing.provenance import sha256_file, write_json
from processing.shot_preflight import (
    measure_shot_geometry,
    select_shot,
    shot_intervals,
)
from processing.video import extract_frames_command, probe_video, scene_cut_times
from processing.video_sources import EVALUATION_STAGES, load_video_registry

PREFLIGHT_FIELDS = (
    "scene_cut_count",
    "sharp_frame_ratio",
    "adjacent_view_overlap",
    "camera_translation_proxy",
    "dynamic_pixel_ratio",
    "exposure_variation",
)


def _gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if arr.ndim != 2 or min(arr.shape) < 16:
        raise ValueError(f"invalid preflight frame: {path}")
    return arr


def _feature_overlap(first: np.ndarray, second: np.ndarray, *, n_keypoints: int = 400) -> float:
    detectors = []
    for image in (first, second):
        detector = ORB(n_keypoints=n_keypoints, fast_threshold=0.07)
        try:
            detector.detect_and_extract(image)
        except RuntimeError:
            return 0.0
        if detector.descriptors is None or len(detector.descriptors) == 0:
            return 0.0
        detectors.append(detector)
    matches = match_descriptors(
        detectors[0].descriptors,
        detectors[1].descriptors,
        cross_check=True,
    )
    denominator = min(len(detectors[0].descriptors), len(detectors[1].descriptors))
    return 0.0 if denominator == 0 else float(len(matches) / denominator)


def _translation_and_dynamic(
    first: np.ndarray,
    second: np.ndarray,
    *,
    dynamic_threshold: float,
) -> tuple[float | None, float]:
    shift, _, _ = phase_cross_correlation(first, second, upsample_factor=4)
    if not np.all(np.isfinite(shift)):
        translation = None
        aligned = second
    else:
        diagonal = math.hypot(first.shape[0], first.shape[1])
        translation = float(np.linalg.norm(shift) / diagonal) if diagonal > 0 else None
        integer_shift = np.rint(shift).astype(int)
        aligned = np.roll(second, tuple(integer_shift), axis=(0, 1))

    residual = np.abs(first - aligned)
    return translation, float(np.mean(residual > dynamic_threshold))


def measure_frames(
    frame_paths: Sequence[str | Path],
    *,
    scene_ssim_threshold: float = 0.35,
    sharpness_threshold: float = 0.0015,
    dynamic_threshold: float = 0.12,
) -> dict:
    """Measure legacy Stage-B screening metrics from one continuous ordered shot."""
    paths = [Path(path).expanduser().resolve() for path in frame_paths]
    if len(paths) < 2:
        raise ValueError("preflight requires at least two sampled frames")
    if not 0 < scene_ssim_threshold < 1:
        raise ValueError("scene_ssim_threshold must be in (0, 1)")
    if sharpness_threshold < 0 or not 0 < dynamic_threshold < 1:
        raise ValueError("invalid preflight thresholds")

    frames = [_gray(path) for path in paths]
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("all preflight frames must have the same dimensions")

    sharpness = [float(np.var(laplace(frame))) for frame in frames]
    exposures = [float(np.mean(frame)) for frame in frames]
    overlaps: list[float] = []
    translations: list[float] = []
    dynamics: list[float] = []
    pair_ssim: list[float] = []

    for first, second in zip(frames, frames[1:], strict=False):
        score = float(structural_similarity(first, second, data_range=1.0))
        pair_ssim.append(score)
        overlaps.append(_feature_overlap(first, second))
        translation, dynamic = _translation_and_dynamic(
            first,
            second,
            dynamic_threshold=dynamic_threshold,
        )
        if translation is not None:
            translations.append(translation)
        dynamics.append(dynamic)

    mean_exposure = float(np.mean(exposures))
    exposure_variation = (
        None if mean_exposure <= 1e-12 else float(np.std(exposures, ddof=0) / mean_exposure)
    )
    metrics = {
        "scene_cut_count": int(sum(score < scene_ssim_threshold for score in pair_ssim)),
        "sharp_frame_ratio": float(np.mean(np.asarray(sharpness) >= sharpness_threshold)),
        "adjacent_view_overlap": float(median(overlaps)),
        "camera_translation_proxy": None if not translations else float(median(translations)),
        "dynamic_pixel_ratio": float(median(dynamics)),
        "exposure_variation": exposure_variation,
    }
    return {
        "metrics": metrics,
        "diagnostics": {
            "sampled_frame_count": len(paths),
            "sharpness_laplacian_variance": sharpness,
            "adjacent_ssim": pair_ssim,
            "adjacent_feature_match_ratio": overlaps,
            "adjacent_translation_diagonal_ratio": translations,
            "adjacent_translation_aligned_dynamic_ratio": dynamics,
            "mean_luma": exposures,
        },
        "method": {
            "scene_cut": f"adjacent grayscale SSIM < {scene_ssim_threshold}",
            "sharp_frame": f"variance(laplacian(gray)) >= {sharpness_threshold}",
            "adjacent_view_overlap": "median ORB cross-checked matches / min adjacent descriptor count",
            "camera_translation_proxy": "median phase-correlation translation magnitude / image diagonal",
            "dynamic_pixel_ratio": f"median translation-aligned grayscale residual fraction > {dynamic_threshold}",
            "exposure_variation": "population stddev(mean grayscale luma) / mean(mean grayscale luma)",
        },
    }


def _extract_shot_frames(
    source: Path,
    directory: Path,
    *,
    sample_fps: float,
    width: int,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[list[Path], list[str]]:
    directory.mkdir(parents=True, exist_ok=True)
    command = extract_frames_command(
        source,
        directory,
        fps=sample_fps,
        width=width,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    completed = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return sorted(directory.glob("frame-*.jpg")), command


def _compact_shot(shot: Mapping[str, Any]) -> dict[str, Any]:
    geometry = dict(shot.get("geometry") or {})
    geometry.pop("pairs", None)
    return {
        "id": shot["id"],
        "index": shot["index"],
        "start_seconds": shot["start_seconds"],
        "end_seconds": shot["end_seconds"],
        "duration_seconds": shot["duration_seconds"],
        "frame_count": shot.get("frame_count", 0),
        "metrics": shot.get("metrics"),
        "geometry": geometry or None,
        "error": shot.get("error"),
    }


def run_video_preflight(
    video: str | Path,
    output_path: str | Path,
    *,
    sample_fps: float = 2.0,
    width: int = 640,
    scene_threshold: float = 0.4,
    minimum_shot_seconds: float = 2.0,
    pair_strides: Sequence[int] = (1, 2, 4),
    scene_ssim_threshold: float = 0.35,
    sharpness_threshold: float = 0.0015,
    dynamic_threshold: float = 0.12,
) -> dict:
    """Measure Stage B per continuous shot and select one shot from measured geometry evidence."""
    source = Path(video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"video does not exist: {source}")
    if sample_fps <= 0 or width <= 0:
        raise ValueError("sample_fps and width must be positive")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    probe = probe_video(source)
    duration_raw = (probe.get("format") or {}).get("duration")
    try:
        duration_seconds = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("video probe did not provide a finite duration") from exc
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("video probe did not provide a finite duration")

    cuts = scene_cut_times(source, threshold=scene_threshold)
    intervals = shot_intervals(
        duration_seconds,
        cuts,
        minimum_seconds=minimum_shot_seconds,
    )
    if not intervals:
        raise RuntimeError("scene segmentation produced no usable shot interval")

    measured_shots: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    selected_frames: list[Path] = []
    selected_measured: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="autophotogrammetry-preflight-") as tmp:
        root = Path(tmp)
        shot_frames: dict[str, list[Path]] = {}
        for interval in intervals:
            shot = dict(interval)
            shot_id = str(shot["id"])
            try:
                frames, command = _extract_shot_frames(
                    source,
                    root / shot_id,
                    sample_fps=sample_fps,
                    width=width,
                    start_seconds=float(shot["start_seconds"]),
                    duration_seconds=float(shot["duration_seconds"]),
                )
                commands.append(command)
                if len(frames) < 2:
                    raise RuntimeError("shot extraction produced fewer than two frames")
                legacy = measure_frames(
                    frames,
                    scene_ssim_threshold=scene_ssim_threshold,
                    sharpness_threshold=sharpness_threshold,
                    dynamic_threshold=dynamic_threshold,
                )
                geometry = measure_shot_geometry(
                    frames,
                    sample_fps=sample_fps,
                    strides=pair_strides,
                )
                shot.update(
                    {
                        "frame_count": len(frames),
                        "metrics": legacy["metrics"],
                        "diagnostics": legacy["diagnostics"],
                        "legacy_method": legacy["method"],
                        "geometry": geometry,
                        "error": None,
                    }
                )
                shot_frames[shot_id] = frames
            except Exception as exc:
                shot.update(
                    {
                        "frame_count": 0,
                        "metrics": None,
                        "geometry": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            measured_shots.append(shot)

        selected_shot_id = select_shot(measured_shots)
        if selected_shot_id is None:
            raise RuntimeError("no shot produced valid two-view geometry evidence")
        selected_measured = next(shot for shot in measured_shots if shot["id"] == selected_shot_id)
        selected_frames = shot_frames[selected_shot_id]
        frame_records = [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in selected_frames
        ]

    assert selected_measured is not None
    result = {
        "schema_version": 2,
        "status": "success",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "probe": probe,
        },
        "sampling": {
            "fps": sample_fps,
            "width": width,
            "pair_strides": list(pair_strides),
            "pair_delta_seconds": [stride / sample_fps for stride in pair_strides],
            "commands": commands,
            "selected_frame_count": len(frame_records),
            "selected_frames": frame_records,
        },
        "metrics": dict(selected_measured["metrics"]),
        "diagnostics": dict(selected_measured["diagnostics"]),
        "method": dict(selected_measured["legacy_method"]),
        "shot_evidence": {
            "scene_cut_threshold": scene_threshold,
            "scene_cut_times_seconds": cuts,
            "minimum_shot_seconds": minimum_shot_seconds,
            "shot_count": len(measured_shots),
            "selected_shot_id": selected_measured["id"],
            "selected_start_seconds": selected_measured["start_seconds"],
            "selected_end_seconds": selected_measured["end_seconds"],
            "selected_duration_seconds": selected_measured["duration_seconds"],
            "selection_basis": [
                "geometry_pair_count desc",
                "essential_inlier_ratio_median desc",
                "triangulation_angle_degrees_median desc",
                "feature_overlap_ratio_median desc",
                "shot duration desc",
                "shot id asc",
            ],
            "shots": [_compact_shot(shot) for shot in measured_shots],
            "selected_geometry": selected_measured["geometry"],
        },
    }
    write_json(destination, result)
    result["manifest_path"] = str(destination)
    return result


def _metadata_gate(source: Mapping) -> None:
    required = ("source_page", "media_url", "author", "duration_seconds", "resolution")
    missing = [field for field in required if not source.get(field)]
    license_record = source.get("license")
    if not isinstance(license_record, Mapping) or license_record.get("status") != "verified":
        missing.append("verified license")
    if missing:
        raise ValueError(
            f"{source.get('id')}: metadata gate is incomplete; missing={sorted(set(missing))}"
        )


def apply_preflight_to_registry(
    registry_path: str | Path,
    source_id: str,
    preflight_result: Mapping,
) -> dict:
    registry_file = Path(registry_path).expanduser().resolve()
    registry = load_video_registry(registry_file)
    if preflight_result.get("status") != "success":
        raise ValueError("only successful preflight evidence can update the registry")
    metrics = preflight_result.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(PREFLIGHT_FIELDS):
        raise ValueError("preflight result does not contain the canonical metric set")

    source = next((item for item in registry["videos"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"unknown video source: {source_id}")
    _metadata_gate(source)

    source["measurements"]["preflight"] = dict(metrics)
    shot_evidence = preflight_result.get("shot_evidence")
    if isinstance(shot_evidence, Mapping):
        source["preflight_evidence"] = {
            "selected_shot_id": shot_evidence.get("selected_shot_id"),
            "selected_start_seconds": shot_evidence.get("selected_start_seconds"),
            "selected_end_seconds": shot_evidence.get("selected_end_seconds"),
            "selected_duration_seconds": shot_evidence.get("selected_duration_seconds"),
            "selection_basis": shot_evidence.get("selection_basis"),
            "shots": shot_evidence.get("shots"),
        }
    current_stage = source["evaluation_stage"]
    if EVALUATION_STAGES.index(current_stage) < EVALUATION_STAGES.index("preflight"):
        source["evaluation_stage"] = "preflight"
    registry_file.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure shot-level CPU preflight evidence for #23 without producing a heuristic "
            "reconstruction score."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--scene-threshold", type=float, default=0.4)
    parser.add_argument("--minimum-shot-seconds", type=float, default=2.0)
    parser.add_argument("--scene-ssim-threshold", type=float, default=0.35)
    parser.add_argument("--sharpness-threshold", type=float, default=0.0015)
    parser.add_argument("--dynamic-threshold", type=float, default=0.12)
    parser.add_argument("--registry")
    parser.add_argument("--source-id")
    parser.add_argument("--update-registry", action="store_true")
    args = parser.parse_args()

    result = run_video_preflight(
        args.video,
        args.output,
        sample_fps=args.sample_fps,
        width=args.width,
        scene_threshold=args.scene_threshold,
        minimum_shot_seconds=args.minimum_shot_seconds,
        scene_ssim_threshold=args.scene_ssim_threshold,
        sharpness_threshold=args.sharpness_threshold,
        dynamic_threshold=args.dynamic_threshold,
    )
    if args.update_registry:
        if not args.registry or not args.source_id:
            parser.error("--update-registry requires --registry and --source-id")
        apply_preflight_to_registry(args.registry, args.source_id, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
