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


def gaussian_ply_metrics(path: str | Path) -> dict:
    """Measure primitive, opacity and scale-anisotropy statistics from a standard 3DGS PLY.

    The exporter representation stores opacity as logits and scale components as log-scales.
    Unsupported PLY layouts fail closed instead of producing guessed metrics.
    """
    ply = Path(path).expanduser().resolve()
    if not ply.is_file():
        raise ValueError(f"PLY does not exist: {ply}")

    with ply.open("rb") as handle:
        vertex_count, dtype = _read_vertex_layout(handle)
        required = {"opacity", "scale_0", "scale_1", "scale_2"}
        missing = required - set(dtype.names or ())
        if missing:
            raise ValueError(f"PLY is missing Gaussian properties: {sorted(missing)}")
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)

    if len(vertices) != vertex_count:
        raise ValueError(f"PLY vertex payload is truncated: expected {vertex_count}, got {len(vertices)}")
    if vertex_count == 0:
        raise ValueError("PLY has zero Gaussian primitives")

    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float64)
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
        "primitive_count": int(vertex_count),
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
    parser = argparse.ArgumentParser(description="Measure Gaussian PLY primitive/opacity/scale statistics.")
    parser.add_argument("ply")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = gaussian_ply_metrics(args.ply)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
