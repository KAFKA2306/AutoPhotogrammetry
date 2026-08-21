from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image
from skimage.feature import ORB, match_descriptors
from skimage.transform import EssentialMatrixTransform, ProjectiveTransform


def shot_intervals(
    duration_seconds: float,
    cut_times: Sequence[float],
    *,
    minimum_seconds: float = 2.0,
) -> list[dict[str, float | int | str]]:
    """Convert scene-cut timestamps into deterministic, non-overlapping shot intervals."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if minimum_seconds <= 0:
        raise ValueError("minimum_seconds must be positive")
    cuts = sorted(
        {
            float(value)
            for value in cut_times
            if math.isfinite(float(value)) and 0 < float(value) < duration_seconds
        }
    )
    boundaries = [0.0, *cuts, float(duration_seconds)]
    shots: list[dict[str, float | int | str]] = []
    for start, end in zip(boundaries, boundaries[1:], strict=True):
        duration = end - start
        if duration + 1e-9 < minimum_seconds:
            continue
        index = len(shots)
        shots.append(
            {
                "id": f"shot-{index:04d}",
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": duration,
            }
        )
    return shots


def _gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2 or min(array.shape) < 16:
        raise ValueError(f"invalid shot frame: {path}")
    return array


def _orb_correspondences(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_keypoints: int = 800,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    keypoints: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    for image in (first, second):
        detector = ORB(n_keypoints=n_keypoints, fast_threshold=0.07)
        try:
            detector.detect_and_extract(image)
        except RuntimeError:
            return np.empty((0, 2)), np.empty((0, 2)), 0, 0, 0
        if detector.descriptors is None or detector.keypoints is None:
            return np.empty((0, 2)), np.empty((0, 2)), 0, 0, 0
        keypoints.append(detector.keypoints)
        descriptors.append(detector.descriptors)
    matches = match_descriptors(descriptors[0], descriptors[1], cross_check=True)
    if len(matches) == 0:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            len(keypoints[0]),
            len(keypoints[1]),
            0,
        )
    first_rc = keypoints[0][matches[:, 0]]
    second_rc = keypoints[1][matches[:, 1]]
    first_xy = first_rc[:, ::-1].astype(np.float64, copy=False)
    second_xy = second_rc[:, ::-1].astype(np.float64, copy=False)
    return first_xy, second_xy, len(keypoints[0]), len(keypoints[1]), len(matches)


def _normalize_points(points: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    height, width = shape
    focal = float(max(height, width))
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    return (points - center) / focal, focal


def _sampson_residuals(matrix: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_h = np.column_stack((first, np.ones(len(first))))
    second_h = np.column_stack((second, np.ones(len(second))))
    e_first = (matrix @ first_h.T).T
    et_second = (matrix.T @ second_h.T).T
    numerator = np.sum(second_h * e_first, axis=1) ** 2
    denominator = (
        e_first[:, 0] ** 2
        + e_first[:, 1] ** 2
        + et_second[:, 0] ** 2
        + et_second[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _fit_essential_ransac(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float,
    max_trials: int = 160,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(first) < 8:
        return None, np.zeros(len(first), dtype=bool)
    rng = np.random.default_rng(0)
    best_model: np.ndarray | None = None
    best_inliers = np.zeros(len(first), dtype=bool)
    best_residual = math.inf
    for _ in range(max_trials):
        sample = rng.choice(len(first), size=8, replace=False)
        model = EssentialMatrixTransform()
        try:
            if not model.estimate(first[sample], second[sample]):
                continue
        except (ValueError, np.linalg.LinAlgError):
            continue
        params = np.asarray(model.params, dtype=np.float64)
        residuals = _sampson_residuals(params, first, second)
        inliers = residuals <= threshold
        count = int(np.sum(inliers))
        if count < 8:
            continue
        inlier_residual = float(np.median(residuals[inliers]))
        if count > int(np.sum(best_inliers)) or (
            count == int(np.sum(best_inliers)) and inlier_residual < best_residual
        ):
            best_model = params
            best_inliers = inliers
            best_residual = inlier_residual
    if best_model is None:
        return None, best_inliers
    refined = EssentialMatrixTransform()
    try:
        if refined.estimate(first[best_inliers], second[best_inliers]):
            best_model = np.asarray(refined.params, dtype=np.float64)
            best_inliers = _sampson_residuals(best_model, first, second) <= threshold
    except (ValueError, np.linalg.LinAlgError):
        pass
    return best_model, best_inliers


def _projective_transfer_error(
    matrix: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first_h = np.column_stack((first, np.ones(len(first))))
    projected = (matrix @ first_h.T).T
    valid = np.abs(projected[:, 2]) > 1e-12
    result = np.full(len(first), np.inf, dtype=np.float64)
    projected_xy = projected[valid, :2] / projected[valid, 2:3]
    result[valid] = np.linalg.norm(projected_xy - second[valid], axis=1)
    return result


def _fit_homography_ransac(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold_px: float = 2.0,
    max_trials: int = 120,
) -> np.ndarray:
    if len(first) < 4:
        return np.zeros(len(first), dtype=bool)
    rng = np.random.default_rng(1)
    best = np.zeros(len(first), dtype=bool)
    best_residual = math.inf
    for _ in range(max_trials):
        sample = rng.choice(len(first), size=4, replace=False)
        model = ProjectiveTransform()
        try:
            if not model.estimate(first[sample], second[sample]):
                continue
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = _projective_transfer_error(np.asarray(model.params), first, second)
        inliers = residuals <= threshold_px
        count = int(np.sum(inliers))
        if count < 4:
            continue
        inlier_residual = float(np.median(residuals[inliers]))
        if count > int(np.sum(best)) or (
            count == int(np.sum(best)) and inlier_residual < best_residual
        ):
            best = inliers
            best_residual = inlier_residual
    return best


def _triangulate(
    first: np.ndarray,
    second: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, list[float]]:
    first_projection = np.column_stack((np.eye(3), np.zeros(3)))
    second_projection = np.column_stack((rotation, translation))
    camera_two = -rotation.T @ translation
    positive = 0
    angles: list[float] = []
    for first_point, second_point in zip(first, second, strict=True):
        matrix = np.stack(
            (
                first_point[0] * first_projection[2] - first_projection[0],
                first_point[1] * first_projection[2] - first_projection[1],
                second_point[0] * second_projection[2] - second_projection[0],
                second_point[1] * second_projection[2] - second_projection[1],
            )
        )
        try:
            _, _, vh = np.linalg.svd(matrix)
        except np.linalg.LinAlgError:
            continue
        homogeneous = vh[-1]
        if abs(homogeneous[3]) <= 1e-12:
            continue
        point = homogeneous[:3] / homogeneous[3]
        depth_one = point[2]
        depth_two = (rotation @ point + translation)[2]
        if depth_one > 0 and depth_two > 0:
            positive += 1
        ray_one = point
        ray_two = point - camera_two
        norm = np.linalg.norm(ray_one) * np.linalg.norm(ray_two)
        if norm > 1e-12:
            cosine = float(np.clip(np.dot(ray_one, ray_two) / norm, -1.0, 1.0))
            angles.append(math.degrees(math.acos(cosine)))
    denominator = max(1, len(first))
    return positive / denominator, angles


def _essential_pose(
    essential: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any] | None:
    if len(first) < 8:
        return None
    try:
        u, _, vh = np.linalg.svd(essential)
    except np.linalg.LinAlgError:
        return None
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vh) < 0:
        vh[-1, :] *= -1
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = (u @ w @ vh, u @ w.T @ vh)
    translation = u[:, 2]
    candidates: list[tuple[float, list[float], np.ndarray, np.ndarray]] = []
    for rotation in rotations:
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        for direction in (translation, -translation):
            cheirality, angles = _triangulate(first, second, rotation, direction)
            candidates.append((cheirality, angles, rotation, direction))
    if not candidates:
        return None
    cheirality, angles, rotation, direction = max(
        candidates,
        key=lambda item: (item[0], median(item[1]) if item[1] else 0.0),
    )
    trace = float(np.trace(rotation))
    rotation_degrees = math.degrees(math.acos(float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))
    norm = float(np.linalg.norm(direction))
    unit_direction = direction / norm if norm > 1e-12 else direction
    return {
        "pose_cheirality_ratio": cheirality,
        "rotation_degrees": rotation_degrees,
        "translation_direction": [float(value) for value in unit_direction],
        "triangulation_angle_degrees": None if not angles else float(median(angles)),
    }


def measure_pair_geometry(first_path: str | Path, second_path: str | Path) -> dict[str, Any]:
    """Measure robust two-view geometry without pretending approximate intrinsics are calibration."""
    first = _gray(first_path)
    second = _gray(second_path)
    if first.shape != second.shape:
        raise ValueError("pair frames must have equal dimensions")
    first_xy, second_xy, first_features, second_features, match_count = _orb_correspondences(
        first, second
    )
    denominator = min(first_features, second_features)
    overlap = 0.0 if denominator == 0 else match_count / denominator
    diagonal = math.hypot(first.shape[0], first.shape[1])
    parallax = (
        None
        if match_count == 0 or diagonal <= 0
        else float(median(np.linalg.norm(second_xy - first_xy, axis=1)) / diagonal)
    )
    result: dict[str, Any] = {
        "match_count": match_count,
        "feature_overlap_ratio": overlap,
        "parallax_diagonal_ratio": parallax,
        "essential_inlier_ratio": None,
        "homography_inlier_ratio": None,
        "pose_cheirality_ratio": None,
        "rotation_degrees": None,
        "translation_direction": None,
        "triangulation_angle_degrees": None,
        "geometry_valid": False,
    }
    if match_count < 8:
        return result

    homography_inliers = _fit_homography_ransac(first_xy, second_xy)
    result["homography_inlier_ratio"] = float(np.mean(homography_inliers))
    first_normalized, focal = _normalize_points(first_xy, first.shape)
    second_normalized, _ = _normalize_points(second_xy, second.shape)
    essential, essential_inliers = _fit_essential_ransac(
        first_normalized,
        second_normalized,
        threshold=2.0 / focal,
    )
    if essential is None or int(np.sum(essential_inliers)) < 8:
        return result
    result["essential_inlier_ratio"] = float(np.mean(essential_inliers))
    pose = _essential_pose(
        essential,
        first_normalized[essential_inliers],
        second_normalized[essential_inliers],
    )
    if pose is None:
        return result
    result.update(pose)
    result["geometry_valid"] = True
    return result


def _median_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return None if not values else float(median(values))


def measure_shot_geometry(
    frame_paths: Sequence[str | Path],
    *,
    sample_fps: float,
    strides: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    paths = sorted(map(Path, frame_paths))
    if len(paths) < 2:
        raise ValueError("shot geometry requires at least two frames")
    normalized_strides = sorted({int(value) for value in strides if int(value) > 0})
    if not normalized_strides:
        raise ValueError("at least one positive pair stride is required")
    pairs: list[dict[str, Any]] = []
    for stride in normalized_strides:
        for first_index in range(0, len(paths) - stride, stride):
            second_index = first_index + stride
            geometry = measure_pair_geometry(paths[first_index], paths[second_index])
            pairs.append(
                {
                    "first_frame": paths[first_index].name,
                    "second_frame": paths[second_index].name,
                    "stride": stride,
                    "delta_seconds": stride / sample_fps,
                    **geometry,
                }
            )
    valid_pairs = [row for row in pairs if row["geometry_valid"]]
    return {
        "frame_count": len(paths),
        "pair_count": len(pairs),
        "geometry_pair_count": len(valid_pairs),
        "feature_overlap_ratio_median": _median_present(pairs, "feature_overlap_ratio"),
        "parallax_diagonal_ratio_median": _median_present(pairs, "parallax_diagonal_ratio"),
        "essential_inlier_ratio_median": _median_present(valid_pairs, "essential_inlier_ratio"),
        "homography_inlier_ratio_median": _median_present(pairs, "homography_inlier_ratio"),
        "pose_cheirality_ratio_median": _median_present(valid_pairs, "pose_cheirality_ratio"),
        "rotation_degrees_median": _median_present(valid_pairs, "rotation_degrees"),
        "triangulation_angle_degrees_median": _median_present(
            valid_pairs, "triangulation_angle_degrees"
        ),
        "pairs": pairs,
        "method": {
            "features": "ORB cross-checked correspondences",
            "essential": (
                "deterministic 8-point EssentialMatrixTransform RANSAC after approximate "
                "normalization with focal=max(width,height); this is a pose proxy, not camera calibration"
            ),
            "homography": "deterministic 4-point projective RANSAC in pixel coordinates",
            "pose": "essential decomposition; cheirality selects R,t; translation magnitude is undefined",
            "triangulation": "median ray angle for positive/finite two-view triangulations",
            "multi_baseline": [stride / sample_fps for stride in normalized_strides],
        },
    }


def select_shot(shots: Sequence[Mapping[str, Any]]) -> str | None:
    """Select one measured shot lexicographically from geometry evidence, never metadata score."""
    eligible = [
        shot
        for shot in shots
        if isinstance(shot.get("geometry"), Mapping)
        and int(shot["geometry"].get("geometry_pair_count") or 0) > 0
    ]
    if not eligible:
        return None

    def evidence_key(shot: Mapping[str, Any]) -> tuple[Any, ...]:
        geometry = shot["geometry"]
        essential = geometry.get("essential_inlier_ratio_median")
        triangulation = geometry.get("triangulation_angle_degrees_median")
        overlap = geometry.get("feature_overlap_ratio_median")
        return (
            -int(geometry.get("geometry_pair_count") or 0),
            -float(essential or 0.0),
            -float(triangulation or 0.0),
            -float(overlap or 0.0),
            -float(shot.get("duration_seconds") or 0.0),
            str(shot["id"]),
        )

    return str(min(eligible, key=evidence_key)["id"])
