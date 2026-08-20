#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": ("i1", 1), "int8": ("i1", 1),
    "uchar": ("u1", 1), "uint8": ("u1", 1),
    "short": ("<i2", 2), "int16": ("<i2", 2),
    "ushort": ("<u2", 2), "uint16": ("<u2", 2),
    "int": ("<i4", 4), "int32": ("<i4", 4),
    "uint": ("<u4", 4), "uint32": ("<u4", 4),
    "float": ("<f4", 4), "float32": ("<f4", 4),
    "double": ("<f8", 8), "float64": ("<f8", 8),
}


def parse_header(path: Path):
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not PLY")
        fmt = None
        vertex_count = None
        props = []
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: truncated header")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                offset = f.tell()
                break
            p = line.split()
            if not p or p[0] in {"comment", "obj_info"}:
                continue
            if p[0] == "format":
                fmt = p[1]
            elif p[0] == "element":
                in_vertex = p[1] == "vertex"
                if in_vertex:
                    vertex_count = int(p[2])
            elif p[0] == "property" and in_vertex:
                if p[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element unsupported")
                typ, name = p[1], p[2]
                if typ not in PLY_TYPES:
                    raise ValueError(f"{path}: unsupported type {typ}")
                props.append((name, typ))
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: expected binary_little_endian, got {fmt}")
    if vertex_count is None or vertex_count < 3:
        raise ValueError(f"{path}: invalid vertex count")
    if not {"x", "y", "z"}.issubset({n for n, _ in props}):
        raise ValueError(f"{path}: x/y/z missing")
    return vertex_count, props, offset


def sample_xyz(path: Path, max_points: int):
    count, props, offset = parse_header(path)
    dtype = np.dtype([(name, code) for name, typ in props for code in [PLY_TYPES[typ][0]]], align=False)
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    n = min(count, max_points)
    idx = np.linspace(0, count - 1, n, dtype=np.int64)
    pts = np.column_stack((data["x"][idx], data["y"][idx], data["z"][idx])).astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        raise ValueError(f"{path}: insufficient finite points")
    return pts, count


def robust_core(pts: np.ndarray, quantile: float = 0.985):
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    cutoff = np.quantile(dist, quantile)
    return pts[dist <= cutoff]


def norm(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def axis_angles_deg(v):
    v = norm(v)
    if v is None:
        raise ValueError("zero vector")
    return {
        "x": math.degrees(math.acos(float(np.clip(abs(v[0]), 0.0, 1.0)))),
        "y": math.degrees(math.acos(float(np.clip(abs(v[1]), 0.0, 1.0)))),
        "z": math.degrees(math.acos(float(np.clip(abs(v[2]), 0.0, 1.0)))),
    }


def angle_between_planes_deg(a, b):
    a = norm(a); b = norm(b)
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def pca_thin_axis(pts: np.ndarray):
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    axis = vecs[:, order[0]]
    return axis, vals[order]


def dominant_plane(pts: np.ndarray, seed: int, iterations: int = 700, distance_ratio: float = 0.006):
    rng = np.random.default_rng(seed)
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    threshold = max(diag * distance_ratio, 1e-8)
    best_mask = None
    best_count = -1
    npts = len(pts)
    for _ in range(iterations):
        ids = rng.choice(npts, size=3, replace=False)
        a, b, c = pts[ids]
        n = norm(np.cross(b - a, c - a))
        if n is None:
            continue
        d = -float(np.dot(n, a))
        mask = np.abs(pts @ n + d) <= threshold
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
    if best_mask is None or best_count < 3:
        raise ValueError("RANSAC plane failed")
    inliers = pts[best_mask]
    center = inliers.mean(axis=0)
    centered = inliers - center
    cov = centered.T @ centered / max(1, len(inliers) - 1)
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, int(np.argmin(vals))]
    residual = np.abs(centered @ normal)
    return {
        "normal": normal,
        "center": center,
        "inliers": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / len(pts)),
        "rms": float(np.sqrt(np.mean(residual ** 2))),
        "threshold": threshold,
        "bbox_diag": diag,
        "eigenvalues": np.sort(vals),
    }


def analyze_file(path: Path, max_points: int):
    pts, total = sample_xyz(path, max_points)
    pts = robust_core(pts)
    thin, global_evals = pca_thin_axis(pts)
    seed = int.from_bytes(path.as_posix().encode("utf-8")[:8].ljust(8, b"\0"), "little")
    plane = dominant_plane(pts, seed)
    pa = axis_angles_deg(plane["normal"])
    ga = axis_angles_deg(thin)
    nearest_plane = min(pa, key=pa.get)
    nearest_global = min(ga, key=ga.get)
    result = {
        "scene": path.parts[path.parts.index("output") + 1],
        "path": path.as_posix(),
        "vertex_count": total,
        "sample_count": int(len(pts)),
        "dominant_plane": {
            "normal": plane["normal"].tolist(),
            "center": plane["center"].tolist(),
            "axis_angles_deg": pa,
            "nearest_axis": nearest_plane,
            "nearest_axis_tilt_deg": pa[nearest_plane],
            "y_up_tilt_deg": pa["y"],
            "inlier_ratio": plane["inlier_ratio"],
            "rms_residual": plane["rms"],
            "distance_threshold": plane["threshold"],
            "bbox_diag": plane["bbox_diag"],
        },
        "global_pca": {
            "thin_axis": thin.tolist(),
            "axis_angles_deg": ga,
            "nearest_axis": nearest_global,
            "nearest_axis_tilt_deg": ga[nearest_global],
            "y_up_tilt_deg": ga["y"],
            "eigenvalues": global_evals.tolist(),
        },
        "plane_vs_global_thin_axis_deg": angle_between_planes_deg(plane["normal"], thin),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--max-points", type=int, default=16000)
    ap.add_argument("--output", default="orientation-results.json")
    args = ap.parse_args()

    files = sorted(Path(args.root).glob("**/splat.ply"))
    if not files:
        raise SystemExit("no splat.ply files found")
    results = []
    for path in files:
        r = analyze_file(path, args.max_points)
        results.append(r)
        p = r["dominant_plane"]
        g = r["global_pca"]
        print(
            f'{r["scene"]}: plane nearest={p["nearest_axis"]} tilt={p["nearest_axis_tilt_deg"]:.3f}° '
            f'Y-up={p["y_up_tilt_deg"]:.3f}° inliers={p["inlier_ratio"]:.3f}; '
            f'global-thin nearest={g["nearest_axis"]} tilt={g["nearest_axis_tilt_deg"]:.3f}° '
            f'Y-up={g["y_up_tilt_deg"]:.3f}°; consistency={r["plane_vs_global_thin_axis_deg"]:.3f}°'
        )
    payload = {
        "schema_version": 1,
        "source_commit": "1fa7f35b1fc9fce669f97d5ec7c7f46ce4601206",
        "method": "uniform-sample + robust-core + dominant-plane-RANSAC/eigh + global-PCA",
        "count": len(results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"measured={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
