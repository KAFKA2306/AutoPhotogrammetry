from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from processing.provenance import sha256_file, write_json


def _open3d():
    try:
        return importlib.import_module("open3d")
    except ImportError as exc:
        raise RuntimeError("Open3D is required for mesh post-processing") from exc


def _triangle_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    return np.sort(edges, axis=1)


def _duplicate_count(rows: np.ndarray) -> int:
    if len(rows) == 0:
        return 0
    return int(len(rows) - len(np.unique(rows, axis=0)))


def _mesh_stats(mesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("mesh must contain vertices and triangles")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh contains non-finite vertices")

    edges = _triangle_edges(triangles)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    repeated_vertex = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    double_area = np.linalg.norm(np.cross(a, b), axis=1)
    degenerate = repeated_vertex | (double_area <= 1e-15)

    _, component_sizes, component_areas = mesh.cluster_connected_triangles()
    normals_finite = True
    if mesh.has_vertex_normals():
        normals_finite = bool(np.isfinite(np.asarray(mesh.vertex_normals)).all())

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(triangles)),
        "bbox_min": [float(value) for value in vertices.min(axis=0)],
        "bbox_max": [float(value) for value in vertices.max(axis=0)],
        "connected_component_count": int(len(component_sizes)),
        "largest_component_face_count": int(max(component_sizes, default=0)),
        "largest_component_area": float(max(component_areas, default=0.0)),
        "duplicate_vertex_count": _duplicate_count(vertices),
        "duplicate_face_count": _duplicate_count(triangles),
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "boundary_edge_count": int(np.count_nonzero(edge_counts == 1)),
        "non_manifold_edge_count": int(np.count_nonzero(edge_counts > 2)),
        "non_manifold_vertex_count": int(len(mesh.get_non_manifold_vertices())),
        "edge_manifold_allow_boundary": bool(mesh.is_edge_manifold(True)),
        "edge_manifold_closed": bool(mesh.is_edge_manifold(False)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
        "has_vertex_normals": bool(mesh.has_vertex_normals()),
        "vertex_normals_finite": normals_finite,
    }


def inspect_mesh(path: str | Path) -> dict:
    o3d = _open3d()
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh does not exist: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    return {
        "schema_version": 1,
        "path": str(mesh_path),
        "sha256": sha256_file(mesh_path),
        "size_bytes": mesh_path.stat().st_size,
        "implementation": {
            "library": "Open3D",
            "version": importlib.metadata.version("open3d"),
        },
        **_mesh_stats(mesh),
    }


def _remove_small_components(mesh, minimum_faces: int) -> int:
    if minimum_faces <= 0:
        return 0
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels_array = np.asarray(labels)
    remove_clusters = {index for index, size in enumerate(sizes) if size < minimum_faces}
    if not remove_clusters:
        return 0
    mask = np.isin(labels_array, list(remove_clusters))
    removed = int(np.count_nonzero(mask))
    mesh.remove_triangles_by_mask(mask.tolist())
    mesh.remove_unreferenced_vertices()
    return removed


def _write_mesh(mesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not _open3d().io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"failed to write mesh: {output_path}")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"mesh writer produced no artifact: {output_path}")


def repair_mesh(
    input_mesh: str | Path,
    output_mesh: str | Path,
    *,
    minimum_component_faces: int = 0,
    remove_non_manifold_edges: bool = False,
) -> dict:
    if minimum_component_faces < 0:
        raise ValueError("minimum_component_faces must be >= 0")
    o3d = _open3d()
    input_path = Path(input_mesh).expanduser().resolve()
    output_path = Path(output_mesh).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("repair output must not overwrite the raw mesh")
    if not input_path.is_file():
        raise FileNotFoundError(f"mesh does not exist: {input_path}")

    raw = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=False)
    before = _mesh_stats(raw)
    mesh = copy.deepcopy(raw)
    operations: list[dict] = []

    initial_vertices = len(mesh.vertices)
    mesh.remove_duplicated_vertices()
    operations.append(
        {"operation": "remove_duplicated_vertices", "removed": initial_vertices - len(mesh.vertices)}
    )
    initial_faces = len(mesh.triangles)
    mesh.remove_duplicated_triangles()
    operations.append(
        {"operation": "remove_duplicated_triangles", "removed": initial_faces - len(mesh.triangles)}
    )
    initial_faces = len(mesh.triangles)
    mesh.remove_degenerate_triangles()
    operations.append(
        {"operation": "remove_degenerate_triangles", "removed": initial_faces - len(mesh.triangles)}
    )
    removed_small = _remove_small_components(mesh, minimum_component_faces)
    operations.append(
        {
            "operation": "remove_small_components",
            "minimum_faces": minimum_component_faces,
            "removed_faces": removed_small,
        }
    )
    if remove_non_manifold_edges:
        initial_faces = len(mesh.triangles)
        mesh.remove_non_manifold_edges()
        operations.append(
            {"operation": "remove_non_manifold_edges", "removed_faces": initial_faces - len(mesh.triangles)}
        )
    mesh.remove_unreferenced_vertices()
    oriented = bool(mesh.orient_triangles()) if mesh.is_orientable() else False
    operations.append({"operation": "orient_triangles", "applied": oriented})
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    operations.append({"operation": "recompute_normals", "applied": True})

    after = _mesh_stats(mesh)
    _write_mesh(mesh, output_path)
    return {
        "schema_version": 1,
        "status": "success",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "inspection": before,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "inspection": after,
        },
        "operations": operations,
    }


def _surface_deviation(source, target) -> dict:
    o3d = _open3d()

    def distances(points: np.ndarray, surface) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface))
        tensor = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        return scene.compute_distance(tensor).numpy()

    source_vertices = np.asarray(source.vertices)
    target_vertices = np.asarray(target.vertices)
    source_to_target = distances(source_vertices, target)
    target_to_source = distances(target_vertices, source)
    return {
        "unit": "source_coordinate_unit",
        "source_vertices_to_decimated_surface_mean": float(source_to_target.mean()),
        "source_vertices_to_decimated_surface_max": float(source_to_target.max()),
        "decimated_vertices_to_source_surface_mean": float(target_to_source.mean()),
        "decimated_vertices_to_source_surface_max": float(target_to_source.max()),
        "symmetric_max": float(max(source_to_target.max(), target_to_source.max())),
    }


def decimate_mesh(
    input_mesh: str | Path,
    output_mesh: str | Path,
    *,
    target_faces: int,
    maximum_error: float = float("inf"),
    boundary_weight: float = 1.0,
) -> dict:
    if target_faces <= 0:
        raise ValueError("target_faces must be positive")
    if boundary_weight <= 0:
        raise ValueError("boundary_weight must be positive")
    o3d = _open3d()
    input_path = Path(input_mesh).expanduser().resolve()
    output_path = Path(output_mesh).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("decimation output must not overwrite its input")
    source = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=False)
    before = _mesh_stats(source)
    if target_faces >= before["face_count"]:
        raise ValueError("target_faces must be smaller than the input face count")

    decimated = source.simplify_quadric_decimation(
        target_number_of_triangles=target_faces,
        maximum_error=maximum_error,
        boundary_weight=boundary_weight,
    )
    decimated.remove_degenerate_triangles()
    decimated.remove_duplicated_triangles()
    decimated.remove_unreferenced_vertices()
    decimated.compute_triangle_normals()
    decimated.compute_vertex_normals()
    after = _mesh_stats(decimated)
    deviation = _surface_deviation(source, decimated)
    _write_mesh(decimated, output_path)
    return {
        "schema_version": 1,
        "status": "success",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "inspection": before,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "inspection": after,
        },
        "algorithm": {
            "name": "Open3D simplify_quadric_decimation",
            "target_faces": target_faces,
            "maximum_error": None if np.isinf(maximum_error) else maximum_error,
            "boundary_weight": boundary_weight,
        },
        "geometry_deviation": deviation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect, repair, or decimate a triangle mesh.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("input_mesh")
    inspect_parser.add_argument("--manifest")

    repair_parser = subparsers.add_parser("repair")
    repair_parser.add_argument("input_mesh")
    repair_parser.add_argument("output_mesh")
    repair_parser.add_argument("--minimum-component-faces", type=int, default=0)
    repair_parser.add_argument("--remove-non-manifold-edges", action="store_true")
    repair_parser.add_argument("--manifest")

    decimate_parser = subparsers.add_parser("decimate")
    decimate_parser.add_argument("input_mesh")
    decimate_parser.add_argument("output_mesh")
    decimate_parser.add_argument("--target-faces", type=int, required=True)
    decimate_parser.add_argument("--maximum-error", type=float, default=float("inf"))
    decimate_parser.add_argument("--boundary-weight", type=float, default=1.0)
    decimate_parser.add_argument("--manifest")

    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_mesh(args.input_mesh)
    elif args.command == "repair":
        result = repair_mesh(
            args.input_mesh,
            args.output_mesh,
            minimum_component_faces=args.minimum_component_faces,
            remove_non_manifold_edges=args.remove_non_manifold_edges,
        )
    else:
        result = decimate_mesh(
            args.input_mesh,
            args.output_mesh,
            target_faces=args.target_faces,
            maximum_error=args.maximum_error,
            boundary_weight=args.boundary_weight,
        )
    if args.manifest:
        write_json(args.manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
