from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from processing.backend_evaluation import (
    artifact_record,
    dataset_identity,
    empty_metrics,
    gaussian_artifact_metrics,
    validate_backend_result,
)
from processing.external_research import (
    materialize_frozen_images,
    read_json_object,
    require_success,
    run_recorded_gpu_step,
    verify_checkout,
    write_tree_manifest,
)
from processing.provenance import write_json

EFA_REPOSITORY = "https://github.com/jcwang-gh/EFA-GS"
EFA_REVISION = "57f330f6f9b12d2684c6df0f0359ffec8f60976d"
EFA_LICENSE_STATE = "research-evaluation-only-commercial-consent-required"


def verify_efa_checkout(checkout: str | Path) -> dict:
    info = verify_checkout(
        checkout,
        expected_revision=EFA_REVISION,
        required_paths=(
            "3DGS/train.py",
            "3DGS/render.py",
            "3DGS/metrics.py",
            "LICENSE.md",
        ),
    )
    return {
        **info,
        "repository": EFA_REPOSITORY,
        "license_state": EFA_LICENSE_STATE,
        "production_eligible": False,
        "implementation_root": str(Path(info["checkout"]) / "3DGS"),
    }


def _copy_colmap_model(source: Path, destination: Path) -> dict:
    required_groups = (
        ("cameras.bin", "cameras.txt"),
        ("images.bin", "images.txt"),
        ("points3D.bin", "points3D.txt"),
    )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied = []
    for options in required_groups:
        candidate = next((source / name for name in options if (source / name).is_file()), None)
        if candidate is None:
            raise ValueError(f"COLMAP model is missing one of {options}: {source}")
        target = destination / candidate.name
        shutil.copy2(candidate, target)
        copied.append(str(target))
    return write_tree_manifest(destination, destination.parent / "colmap-model.json")


def _native_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_efa_gs(
    dataset_json: str | Path,
    transforms_json: str | Path,
    colmap_model: str | Path,
    checkout: str | Path,
    output_root: str | Path,
    *,
    python_executable: str = "python",
    iterations: int = 30000,
    resolution: int = 1,
    port: int = 12109,
    init_scaling_multiplier_max: float = 1.5,
    init_scaling_multiplier_min: float = 1.0,
    interval_times: int = 2,
    diffscale: bool = True,
    tolerance: float = 1e-5,
    timeout: float | None = None,
) -> dict:
    """Run pinned EFA-GS 3DGS on the exact frozen input frame bytes.

    The official method's own `--eval` split is retained as native evidence and is not
    relabeled as the #26 deterministic hold-out.
    """
    if iterations <= 0 or resolution <= 0 or interval_times <= 0:
        raise ValueError("iterations, resolution and interval_times must be positive")
    dataset = read_json_object(dataset_json)
    upstream = verify_efa_checkout(checkout)
    impl = Path(upstream["implementation_root"])
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = root / "input-scene"
    images_manifest = materialize_frozen_images(
        dataset,
        transforms_json,
        scene / "images",
        preserve_basename=True,
    )
    write_json(scene / "input-manifest.json", images_manifest)
    colmap_manifest = _copy_colmap_model(
        Path(colmap_model).expanduser().resolve(),
        scene / "sparse" / "0",
    )

    model = root / "model"
    if model.exists():
        shutil.rmtree(model)
    logs = root / "logs"
    train_command = [
        python_executable,
        str(impl / "train.py"),
        "-s",
        str(scene),
        "-m",
        str(model),
        "--eval",
        "-r",
        str(resolution),
        "--port",
        str(port),
        "--iterations",
        str(iterations),
        "--init_scaling_multiplier_max",
        str(init_scaling_multiplier_max),
        "--init_scaling_multiplier_min",
        str(init_scaling_multiplier_min),
        "--interval_times",
        str(interval_times),
        "--tolerance",
        str(tolerance),
        "--diffscale",
        "True" if diffscale else "False",
    ]
    steps = []
    train = run_recorded_gpu_step(
        train_command,
        cwd=impl,
        log_root=logs,
        name="train",
        timeout=timeout,
        env={"OMP_NUM_THREADS": "4"},
    )
    steps.append(train)
    require_success(train)

    render = run_recorded_gpu_step(
        [python_executable, str(impl / "render.py"), "-m", str(model), "--skip_train"],
        cwd=impl,
        log_root=logs,
        name="render",
        timeout=timeout,
        env={"OMP_NUM_THREADS": "4"},
    )
    steps.append(render)
    require_success(render)

    metrics_step = run_recorded_gpu_step(
        [python_executable, str(impl / "metrics.py"), "-m", str(model)],
        cwd=impl,
        log_root=logs,
        name="metrics",
        timeout=timeout,
        env={"OMP_NUM_THREADS": "4"},
    )
    steps.append(metrics_step)
    require_success(metrics_step)

    ply = model / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    artifact_metrics = gaussian_artifact_metrics(ply)
    common = empty_metrics()
    common.update(
        {
            "reconstruction_success": True,
            "input_frame_count": len(dataset.get("frames") or []),
            "train_frame_count": len(dataset.get("train_frame_sha256") or []),
            "holdout_frame_count": len(dataset.get("holdout_frame_sha256") or []),
            "wall_clock_seconds": sum(float(step["wall_clock_seconds"]) for step in steps),
            "peak_gpu_memory_bytes": max(
                (int(step["peak_gpu_memory_bytes"]) for step in steps if step.get("peak_gpu_memory_bytes") is not None),
                default=None,
            ),
            "camera_pose_available": True,
            **artifact_metrics,
        }
    )
    result = {
        "schema_version": 2,
        "dataset_id": dataset_identity(dataset),
        "backend": {"name": "efa-gs-3dgs-research", "upstream_revision": EFA_REVISION},
        "command": train_command,
        "config": {
            "license_state": EFA_LICENSE_STATE,
            "production_eligible": False,
            "iterations": iterations,
            "resolution": resolution,
            "init_scaling_multiplier_max": init_scaling_multiplier_max,
            "init_scaling_multiplier_min": init_scaling_multiplier_min,
            "interval_times": interval_times,
            "diffscale": diffscale,
            "tolerance": tolerance,
            "native_eval_semantics": "EFA-GS/3DGS --eval; not relabeled as #26 deterministic hold-out",
            "colmap_model_manifest": colmap_manifest["manifest_path"],
        },
        "return_code": 0,
        "status": "success",
        "failure_phase": None,
        "artifact": artifact_record(ply, format="ply"),
        "metrics": common,
        "native_metrics": _native_metrics(model / "results.json"),
        "input_manifest": str(scene / "input-manifest.json"),
        "steps": steps,
        "upstream": upstream,
    }
    validate_backend_result(result, dataset)
    path = root / "backend-result.json"
    write_json(path, result)
    return {**result, "manifest_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned EFA-GS/3DGS as a non-commercial research backend.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--colmap-model", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--port", type=int, default=12109)
    parser.add_argument("--init-scaling-multiplier-max", type=float, default=1.5)
    parser.add_argument("--init-scaling-multiplier-min", type=float, default=1.0)
    parser.add_argument("--interval-times", type=int, default=2)
    parser.add_argument("--diffscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_efa_gs(
        args.dataset,
        args.transforms,
        args.colmap_model,
        args.checkout,
        args.output_root,
        python_executable=args.python,
        iterations=args.iterations,
        resolution=args.resolution,
        port=args.port,
        init_scaling_multiplier_max=args.init_scaling_multiplier_max,
        init_scaling_multiplier_min=args.init_scaling_multiplier_min,
        interval_times=args.interval_times,
        diffscale=args.diffscale,
        tolerance=args.tolerance,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
