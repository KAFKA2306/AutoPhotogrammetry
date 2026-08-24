from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import numpy as np

from processing.gaussian_ply import (
    _read_vertices,
    gaussian_ply_inspection,
    validate_gaussian_ply_backend,
)
from processing.provenance import sha256_file, write_json


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for PLY-only mesh reconstruction") from exc
    return o3d


def _point_cloud(path: str | Path):
    o3d = _open3d()
    _, vertices = _read_vertices(path)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return o3d, cloud


def _mesh_stats(mesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("surface reconstruction produced an empty mesh")
    labels, counts, _ = mesh.cluster_connected_triangles()
    del labels
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(triangles)),
        "connected_components": int(len(counts)),
        "bbox_min": [float(value) for value in vertices.min(axis=0)],
        "bbox_max": [float(value) for value in vertices.max(axis=0)],
    }


def reconstruct_mesh(
    input_ply: str | Path,
    output_ply: str | Path,
    *,
    method: str,
    alpha: float | None = None,
    radii: list[float] | None = None,
    depth: int | None = None,
    normal_radius: float | None = None,
    normal_max_nn: int = 30,
    max_faces: int | None = None,
) -> dict:
    inspection = gaussian_ply_inspection(input_ply)
    backend = validate_gaussian_ply_backend(inspection, "point-cloud-poisson")
    if not backend["supported"]:
        raise ValueError(f"input PLY is not usable as a point cloud: {backend['missing_fields']}")

    o3d, cloud = _point_cloud(input_ply)
    parameters: dict[str, object] = {}
    started = time.perf_counter()

    if method == "alpha":
        if alpha is None or alpha <= 0:
            raise ValueError("alpha must be positive")
        parameters["alpha"] = alpha
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
    elif method in {"ball-pivoting", "poisson"}:
        if normal_radius is None or normal_radius <= 0:
            raise ValueError("normal_radius must be positive for normal-dependent methods")
        if normal_max_nn <= 0:
            raise ValueError("normal_max_nn must be positive")
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        cloud.orient_normals_consistent_tangent_plane(min(normal_max_nn, len(cloud.points) - 1))
        parameters.update({"normal_radius": normal_radius, "normal_max_nn": normal_max_nn})
        if method == "ball-pivoting":
            if not radii or any(radius <= 0 for radius in radii):
                raise ValueError("ball-pivoting radii must contain positive values")
            parameters["radii"] = radii
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                cloud, o3d.utility.DoubleVector(radii)
            )
        else:
            if depth is None or depth <= 0:
                raise ValueError("poisson depth must be positive")
            parameters["depth"] = depth
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=depth
            )
            parameters["density_min"] = float(np.min(densities)) if len(densities) else None
    else:
        raise ValueError(f"unsupported mesh method: {method}")

    wall_seconds = time.perf_counter() - started
    stats = _mesh_stats(mesh)
    if max_faces is not None and stats["face_count"] > max_faces:
        raise ValueError(
            f"surface reconstruction exceeded max_faces: {stats['face_count']} > {max_faces}"
        )

    output = Path(output_ply).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write triangle mesh: {output}")

    return {
        "schema_version": 1,
        "status": "success",
        "method": method,
        "parameters": parameters,
        "implementation": {"library": "Open3D", "version": importlib.metadata.version("open3d")},
        "input": {
            "path": inspection["path"],
            "sha256": inspection["sha256"],
            "size_bytes": inspection["size_bytes"],
            "point_count": inspection["vertex_count"],
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            **stats,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a raw triangle mesh from Gaussian PLY centers using Open3D."
    )
    parser.add_argument("input_ply")
    parser.add_argument("output_ply")
    parser.add_argument("--method", required=True, choices=["alpha", "ball-pivoting", "poisson"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--radii", type=float, nargs="+")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--normal-radius", type=float)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--max-faces", type=int)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    result = reconstruct_mesh(
        args.input_ply,
        args.output_ply,
        method=args.method,
        alpha=args.alpha,
        radii=args.radii,
        depth=args.depth,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        max_faces=args.max_faces,
    )
    if args.manifest:
        write_json(args.manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
