from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from processing.provenance import sha256_file, write_json

_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

_BACKEND_REQUIREMENTS = {
    "point-cloud-poisson": {"x", "y", "z"},
    "gaussian-covariance-normal": {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    },
}


def _read_vertex_layout(handle) -> tuple[int, np.dtype]:
    first = handle.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("not a PLY file")

    format_name = None
    vertex_count = None
    properties: list[tuple[str, str]] = []
    current_element = None
    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError("PLY header ended before end_header")
        line = raw.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue
        parts = line.split()
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("list properties are not supported in the vertex element")
            ply_type, name = parts[1], parts[2]
            if ply_type not in _PLY_TYPES:
                raise ValueError(f"unsupported PLY property type: {ply_type}")
            properties.append((name, _PLY_TYPES[ply_type]))

    if format_name != "binary_little_endian":
        raise ValueError(f"expected binary_little_endian PLY, got {format_name!r}")
    if vertex_count is None or vertex_count < 0:
        raise ValueError("PLY vertex count is missing or invalid")
    if not properties:
        raise ValueError("PLY vertex properties are missing")
    return vertex_count, np.dtype(properties)


def _read_vertices(path: str | Path) -> tuple[Path, np.ndarray]:
    ply = Path(path).expanduser().resolve()
    if not ply.is_file():
        raise ValueError(f"PLY does not exist: {ply}")

    with ply.open("rb") as handle:
        vertex_count, dtype = _read_vertex_layout(handle)
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)

    if len(vertices) != vertex_count:
        raise ValueError(
            f"PLY vertex payload is truncated: expected {vertex_count}, got {len(vertices)}"
        )
    if vertex_count == 0:
        raise ValueError("PLY has zero Gaussian primitives")
    return ply, vertices


def _infer_sh_degree(names: set[str]) -> int | None:
    rest_count = sum(name.startswith("f_rest_") for name in names)
    if rest_count == 0:
        return 0 if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names) else None
    if rest_count % 3 != 0:
        return None
    coefficients_per_channel = rest_count // 3 + 1
    root = int(math.isqrt(coefficients_per_channel))
    if root * root != coefficients_per_channel:
        return None
    return root - 1


def gaussian_ply_inspection(path: str | Path) -> dict:
    """Inspect a Gaussian PLY without modifying it.

    Unknown field layouts remain unknown. The function validates payload completeness and
    finite numeric values instead of guessing missing Gaussian attributes or provenance.
    """
    ply, vertices = _read_vertices(path)
    names = set(vertices.dtype.names or ())
    required_xyz = {"x", "y", "z"}
    missing_xyz = required_xyz - names
    if missing_xyz:
        raise ValueError(f"PLY is missing position properties: {sorted(missing_xyz)}")

    floating_names = [
        name
        for name in vertices.dtype.names or ()
        if np.issubdtype(vertices.dtype[name], np.floating)
    ]
    nonfinite_by_property = {
        name: int(np.count_nonzero(~np.isfinite(vertices[name]))) for name in floating_names
    }
    nonfinite_by_property = {
        name: count for name, count in nonfinite_by_property.items() if count > 0
    }
    if nonfinite_by_property:
        raise ValueError(f"PLY contains non-finite values: {nonfinite_by_property}")

    xyz = np.column_stack(
        [
            np.asarray(vertices["x"], dtype=np.float64),
            np.asarray(vertices["y"], dtype=np.float64),
            np.asarray(vertices["z"], dtype=np.float64),
        ]
    )
    minimum = np.min(xyz, axis=0)
    maximum = np.max(xyz, axis=0)
    centroid = np.mean(xyz, axis=0)

    rotation_fields = [f"rot_{index}" for index in range(4)]
    scale_fields = [f"scale_{index}" for index in range(3)]
    dc_fields = sorted(name for name in names if name.startswith("f_dc_"))
    rest_fields = sorted(name for name in names if name.startswith("f_rest_"))

    return {
        "schema_version": 1,
        "path": str(ply),
        "encoding": "binary_little_endian",
        "size_bytes": ply.stat().st_size,
        "sha256": sha256_file(ply),
        "vertex_count": int(len(vertices)),
        "properties": list(vertices.dtype.names or ()),
        "dialect": "unknown",
        "position": {
            "fields": ["x", "y", "z"],
            "bbox_min": [float(value) for value in minimum],
            "bbox_max": [float(value) for value in maximum],
            "centroid": [float(value) for value in centroid],
        },
        "gaussian_fields": {
            "opacity": "opacity" in names,
            "scale": all(field in names for field in scale_fields),
            "rotation": all(field in names for field in rotation_fields),
            "dc_fields": dc_fields,
            "rest_field_count": len(rest_fields),
            "inferred_sh_degree": _infer_sh_degree(names),
        },
        "finite_values": True,
    }


def validate_gaussian_ply_backend(inspection: dict, backend: str) -> dict:
    """Validate whether an inspected PLY satisfies a mesh backend's PLY-side inputs."""
    if backend == "render-depth-tsdf":
        return {
            "backend": backend,
            "supported": False,
            "missing_fields": [],
            "reason": "requires training config/checkpoint/cameras; PLY alone is insufficient",
        }
    if backend not in _BACKEND_REQUIREMENTS:
        raise ValueError(f"unsupported Gaussian PLY backend: {backend}")

    names = set(inspection.get("properties", []))
    missing = sorted(_BACKEND_REQUIREMENTS[backend] - names)
    return {
        "backend": backend,
        "supported": not missing,
        "missing_fields": missing,
        "reason": None if not missing else "required PLY properties are missing",
    }


def gaussian_ply_metrics(path: str | Path) -> dict:
    """Measure primitive, opacity and scale-anisotropy statistics from a standard 3DGS PLY.

    The exporter representation stores opacity as logits and scale components as log-scales.
    Unsupported PLY layouts fail closed instead of producing guessed metrics.
    """
    ply, vertices = _read_vertices(path)
    required = {"opacity", "scale_0", "scale_1", "scale_2"}
    missing = required - set(vertices.dtype.names or ())
    if missing:
        raise ValueError(f"PLY is missing Gaussian properties: {sorted(missing)}")

    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float64)
    if not np.all(np.isfinite(opacity_logits)):
        raise ValueError("PLY opacity contains non-finite values")
    opacity = np.empty_like(opacity_logits)
    positive = opacity_logits >= 0
    opacity[positive] = 1.0 / (1.0 + np.exp(-opacity_logits[positive]))
    exp_logits = np.exp(opacity_logits[~positive])
    opacity[~positive] = exp_logits / (1.0 + exp_logits)

    log_scales = np.column_stack(
        [
            np.asarray(vertices["scale_0"], dtype=np.float64),
            np.asarray(vertices["scale_1"], dtype=np.float64),
            np.asarray(vertices["scale_2"], dtype=np.float64),
        ]
    )
    if not np.all(np.isfinite(log_scales)):
        raise ValueError("PLY scales contain non-finite values")
    log_ratio = np.max(log_scales, axis=1) - np.min(log_scales, axis=1)
    ratio = np.exp(np.minimum(log_ratio, math.log(np.finfo(np.float64).max)))

    def quantiles(values: np.ndarray) -> dict:
        p50, p95, p99 = np.quantile(values, [0.50, 0.95, 0.99])
        return {
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": float(np.max(values)),
        }

    low_opacity = opacity < 0.1
    spiky = ratio > 10.0
    return {
        "schema_version": 1,
        "path": str(ply),
        "size_bytes": ply.stat().st_size,
        "sha256": sha256_file(ply),
        "primitive_count": int(len(vertices)),
        "opacity": {
            "quantiles": quantiles(opacity),
            "below_0_1_count": int(np.count_nonzero(low_opacity)),
            "below_0_1_ratio": float(np.mean(low_opacity)),
        },
        "scale_anisotropy_ratio": {
            "quantiles": quantiles(ratio),
            "above_10_count": int(np.count_nonzero(spiky)),
            "above_10_ratio": float(np.mean(spiky)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or measure a Gaussian PLY.")
    parser.add_argument("ply")
    parser.add_argument("--output")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument(
        "--backend",
        choices=[*sorted(_BACKEND_REQUIREMENTS), "render-depth-tsdf"],
        help="Validate PLY-side requirements for a mesh backend; implies --inspect.",
    )
    args = parser.parse_args()

    if args.inspect or args.backend:
        result = gaussian_ply_inspection(args.ply)
        if args.backend:
            result["backend_validation"] = validate_gaussian_ply_backend(result, args.backend)
    else:
        result = gaussian_ply_metrics(args.ply)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
