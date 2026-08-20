#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def parse_header(path: Path):
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not PLY")
        fmt = None
        count = None
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
                    count = int(p[2])
            elif p[0] == "property" and in_vertex:
                if p[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element unsupported")
                typ, name = p[1], p[2]
                if typ not in PLY_TYPES:
                    raise ValueError(f"{path}: unsupported property type {typ}")
                props.append((name, PLY_TYPES[typ]))
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: expected binary_little_endian, got {fmt}")
    if count is None or count < 3:
        raise ValueError(f"{path}: invalid vertex count")
    if not {"x", "y", "z"}.issubset({n for n, _ in props}):
        raise ValueError(f"{path}: x/y/z missing")
    return count, np.dtype(props, align=False), offset


def sample_xyz(path: Path, max_points: int):
    count, dtype, offset = parse_header(path)
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    n = min(count, max_points)
    idx = np.linspace(0, count - 1, n, dtype=np.int64)
    pts = np.column_stack((data["x"][idx], data["y"][idx], data["z"][idx])).astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        raise ValueError(f"{path}: insufficient finite points")
    return pts, count


def robust_core(pts, q=0.985):
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    return pts[d <= np.quantile(d, q)]


def unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def axis_angles(v):
    v = unit(v)
    if v is None:
        raise ValueError("zero vector")
    return {axis: math.degrees(math.acos(float(np.clip(abs(v[i]), 0.0, 1.0)))) for i, axis in enumerate("xyz")}


def plane_angle(a, b):
    a = unit(a); b = unit(b)
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def pca_thin_axis(pts):
    c = pts - pts.mean(axis=0)
    cov = c.T @ c / max(1, len(c) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    return vecs[:, order[0]], vals[order]


def dominant_plane(pts, seed, iterations=700, distance_ratio=0.006):
    rng = np.random.default_rng(seed)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    threshold = max(diag * distance_ratio, 1e-8)
    best = None
    best_count = -1
    for _ in range(iterations):
        ids = rng.choice(len(pts), size=3, replace=False)
        a, b, c = pts[ids]
        n = unit(np.cross(b - a, c - a))
        if n is None:
            continue
        d = -float(np.dot(n, a))
        mask = np.abs(pts @ n + d) <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best = mask
    if best is None or best_count < 3:
        raise ValueError("RANSAC plane failed")
    inliers = pts[best]
    center = inliers.mean(axis=0)
    c = inliers - center
    cov = c.T @ c / max(1, len(c) - 1)
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, int(np.argmin(vals))]
    residual = np.abs(c @ normal)
    return normal, center, float(len(inliers) / len(pts)), float(np.sqrt(np.mean(residual ** 2))), threshold


def analyze(path: Path, max_points: int):
    pts, total = sample_xyz(path, max_points)
    pts = robust_core(pts)
    thin, evals = pca_thin_axis(pts)
    seed = int.from_bytes(path.as_posix().encode()[:8].ljust(8, b"\0"), "little")
    normal, center, inlier_ratio, rms, threshold = dominant_plane(pts, seed)
    pa = axis_angles(normal)
    ga = axis_angles(thin)
    pn = min(pa, key=pa.get)
    gn = min(ga, key=ga.get)
    parts = path.parts
    scene = parts[parts.index("output") + 1]
    return {
        "scene": scene,
        "path": path.as_posix(),
        "vertex_count": total,
        "sample_count": int(len(pts)),
        "dominant_plane": {
            "normal": normal.tolist(), "center": center.tolist(), "axis_angles_deg": pa,
            "nearest_axis": pn, "nearest_axis_tilt_deg": pa[pn], "y_up_tilt_deg": pa["y"],
            "inlier_ratio": inlier_ratio, "rms_residual": rms, "distance_threshold": threshold,
        },
        "global_pca": {
            "thin_axis": thin.tolist(), "axis_angles_deg": ga, "nearest_axis": gn,
            "nearest_axis_tilt_deg": ga[gn], "y_up_tilt_deg": ga["y"], "eigenvalues": evals.tolist(),
        },
        "plane_vs_global_thin_axis_deg": plane_angle(normal, thin),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--max-points", type=int, default=16000)
    ap.add_argument("--output", default="orientation-results.json")
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()
    files = sorted(Path(args.root).glob("**/splat.ply"))
    if not files:
        raise SystemExit("no splat.ply files found")
    results = []
    for path in files:
        r = analyze(path, args.max_points)
        results.append(r)
        p, g = r["dominant_plane"], r["global_pca"]
        print(f'{r["scene"]}: plane={p["nearest_axis"]}+{p["nearest_axis_tilt_deg"]:.3f}deg Y-up={p["y_up_tilt_deg"]:.3f}deg inliers={p["inlier_ratio"]:.3f}; global={g["nearest_axis"]}+{g["nearest_axis_tilt_deg"]:.3f}deg Y-up={g["y_up_tilt_deg"]:.3f}deg consistency={r["plane_vs_global_thin_axis_deg"]:.3f}deg')
    payload = {"schema_version": 1, "source_commit": args.source_commit, "method": "uniform-sample+robust-core+RANSAC-plane+global-PCA", "count": len(results), "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"measured={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
