from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from processing.provenance import sha256_file, write_json

SUPPORTED_FORMATS = {"glb", "obj", "stl"}


def _open3d():
    try:
        return importlib.import_module("open3d")
    except ImportError as exc:
        raise RuntimeError("Open3D is required for mesh export") from exc


def _format_from_output(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported mesh export format: {suffix or 'missing'}")
    return suffix


def _legacy_mesh_stats(mesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("input mesh must contain vertices and triangles")
    if not np.isfinite(vertices).all():
        raise ValueError("input mesh contains non-finite vertices")
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(triangles)),
        "bbox_min": [float(value) for value in vertices.min(axis=0)],
        "bbox_max": [float(value) for value in vertices.max(axis=0)],
    }


def _belongs_to_obj_group(path: Path, output: Path) -> bool:
    return path == output or path.name.startswith(f"{output.stem}_") or path.stem == output.stem


def _artifact_files(output: Path) -> list[dict]:
    if output.suffix.lower() == ".obj":
        candidates = sorted(
            path
            for path in output.parent.iterdir()
            if path.is_file() and _belongs_to_obj_group(path, output)
        )
    else:
        candidates = [output]
    return [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in candidates
    ]


def export_mesh(
    input_mesh: str | Path,
    output_mesh: str | Path,
    *,
    source_manifest: str | Path | None = None,
) -> dict:
    o3d = _open3d()
    input_path = Path(input_mesh).expanduser().resolve()
    output_path = Path(output_mesh).expanduser().resolve()
    output_format = _format_from_output(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input mesh does not exist: {input_path}")

    legacy = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=False)
    source_stats = _legacy_mesh_stats(legacy)
    if not legacy.has_vertex_normals():
        legacy.compute_vertex_normals()

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.t.io.write_triangle_mesh(str(output_path), tensor_mesh):
        raise RuntimeError(f"failed to export mesh: {output_path}")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"mesh exporter produced no artifact: {output_path}")

    readback = o3d.io.read_triangle_mesh(str(output_path), enable_post_processing=False)
    readback_stats = _legacy_mesh_stats(readback)

    lineage = None
    if source_manifest is not None:
        manifest_path = Path(source_manifest).expanduser().resolve()
        lineage = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    return {
        "schema_version": 1,
        "status": "success",
        "implementation": {
            "library": "Open3D",
            "version": importlib.metadata.version("open3d"),
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
            **source_stats,
        },
        "output": {
            "format": output_format,
            "primary_path": str(output_path),
            "files": _artifact_files(output_path),
            "readback": readback_stats,
        },
        "transform": {
            "applied": False,
            "coordinate_frame": "source-mesh-unmodified",
            "metric_scale": {"status": "unverified", "unit": None},
        },
        "source_lineage": lineage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a raw triangle mesh to GLB, OBJ, or STL.")
    parser.add_argument("input_mesh")
    parser.add_argument("output_mesh")
    parser.add_argument("--source-manifest")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    result = export_mesh(
        args.input_mesh,
        args.output_mesh,
        source_manifest=args.source_manifest,
    )
    if args.manifest:
        write_json(args.manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
