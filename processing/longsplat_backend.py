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

LONGSPLAT_REPOSITORY = "https://github.com/NVlabs/LongSplat"
LONGSPLAT_REVISION = "19750775a9d19f30aa05a8333c4c6c231b2d5f4a"
LONGSPLAT_LICENSE_STATE = "research-evaluation-only-noncommercial"


def verify_longsplat_checkout(checkout: str | Path) -> dict:
    info = verify_checkout(
        checkout,
        expected_revision=LONGSPLAT_REVISION,
        required_paths=(
            "train.py",
            "render.py",
            "metrics.py",
            "convert_3dgs.py",
            "scripts/train_custom.sh",
            "LICENSE.md",
        ),
    )
    return {
        **info,
        "repository": LONGSPLAT_REPOSITORY,
        "license_state": LONGSPLAT_LICENSE_STATE,
        "production_eligible": False,
    }


def _native_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_longsplat(
    dataset_json: str | Path,
    transforms_json: str | Path,
    checkout: str | Path,
    output_root: str | Path,
    *,
    python_executable: str = "python",
    resolution: int = 4,
    pose_iterations: int = 100,
    local_iterations: int = 200,
    global_iterations: int = 500,
    conversion_iterations: int = 10000,
    prune_ratio: float = 0.6,
    port: int = 12009,
    timeout: float | None = None,
) -> dict:
    """Run pinned LongSplat on the exact frozen input frames and export standard 3DGS.

    LongSplat's own `--eval` split is method-native and is not relabeled as the #26
    deterministic hold-out. Native metrics are retained separately; common PSNR/SSIM/
    LPIPS remain null unless a future evaluator proves identical hold-out semantics.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if any(value <= 0 for value in (pose_iterations, local_iterations, global_iterations, conversion_iterations)):
        raise ValueError("iteration counts must be positive")
    if not 0 <= prune_ratio < 1:
        raise ValueError("prune_ratio must be in [0, 1)")

    dataset = read_json_object(dataset_json)
    upstream = verify_longsplat_checkout(checkout)
    source_root = Path(upstream["checkout"])
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = root / "input-scene"
    input_manifest = materialize_frozen_images(
        dataset,
        transforms_json,
        scene / "images",
        preserve_basename=True,
    )
    write_json(scene / "input-manifest.json", input_manifest)

    model = root / "model"
    if model.exists():
        shutil.rmtree(model)
    logs = root / "logs"
    steps = []

    train_command = [
        python_executable,
        str(source_root / "train.py"),
        "--eval",
        "-s",
        str(scene),
        "-m",
        str(model),
        "--port",
        str(port),
        "--images",
        "images",
        "--mode",
        "custom",
        "-r",
        str(resolution),
        "--pose_iteration",
        str(pose_iterations),
        "--local_iter",
        str(local_iterations),
        "--global_iter",
        str(global_iterations),
    ]
    train = run_recorded_gpu_step(
        train_command,
        cwd=source_root,
        log_root=logs,
        name="train",
        timeout=timeout,
    )
    steps.append(train)
    require_success(train)

    render = run_recorded_gpu_step(
        [python_executable, str(source_root / "render.py"), "-m", str(model)],
        cwd=source_root,
        log_root=logs,
        name="render",
        timeout=timeout,
    )
    steps.append(render)
    require_success(render)

    metrics_step = run_recorded_gpu_step(
        [python_executable, str(source_root / "metrics.py"), "-m", str(model)],
        cwd=source_root,
        log_root=logs,
        name="metrics",
        timeout=timeout,
    )
    steps.append(metrics_step)
    require_success(metrics_step)
    native_metrics = _native_metrics(model / "results.json")

    convert = run_recorded_gpu_step(
        [
            python_executable,
            str(source_root / "convert_3dgs.py"),
            "-m",
            str(model),
            "--iteration",
            str(conversion_iterations),
            "--prune_ratio",
            str(prune_ratio),
        ],
        cwd=source_root,
        log_root=logs,
        name="convert-3dgs",
        timeout=timeout,
    )
    steps.append(convert)
    require_success(convert)

    ply = model / "converted_3dgs" / "point_cloud.ply"
    artifact_metrics = gaussian_artifact_metrics(ply)
    native_manifest = write_tree_manifest(
        model,
        root / "native-model.json",
        suffixes=(".json", ".ply", ".pth"),
    )
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
        "backend": {
            "name": "longsplat-research-converted-3dgs",
            "upstream_revision": LONGSPLAT_REVISION,
        },
        "command": train_command,
        "config": {
            "license_state": LONGSPLAT_LICENSE_STATE,
            "production_eligible": False,
            "resolution": resolution,
            "pose_iterations": pose_iterations,
            "local_iterations": local_iterations,
            "global_iterations": global_iterations,
            "conversion_iterations": conversion_iterations,
            "prune_ratio": prune_ratio,
            "native_eval_semantics": "LongSplat --eval; not relabeled as #26 deterministic hold-out",
        },
        "return_code": 0,
        "status": "success",
        "failure_phase": None,
        "artifact": artifact_record(ply, format="ply"),
        "metrics": common,
        "native_metrics": native_metrics,
        "native_model_manifest": native_manifest["manifest_path"],
        "input_manifest": str(scene / "input-manifest.json"),
        "steps": steps,
        "upstream": upstream,
    }
    validate_backend_result(result, dataset)
    path = root / "backend-result.json"
    write_json(path, result)
    return {**result, "manifest_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned LongSplat as a non-commercial research backend and export standard 3DGS."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--pose-iterations", type=int, default=100)
    parser.add_argument("--local-iterations", type=int, default=200)
    parser.add_argument("--global-iterations", type=int, default=500)
    parser.add_argument("--conversion-iterations", type=int, default=10000)
    parser.add_argument("--prune-ratio", type=float, default=0.6)
    parser.add_argument("--port", type=int, default=12009)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_longsplat(
        args.dataset,
        args.transforms,
        args.checkout,
        args.output_root,
        python_executable=args.python,
        resolution=args.resolution,
        pose_iterations=args.pose_iterations,
        local_iterations=args.local_iterations,
        global_iterations=args.global_iterations,
        conversion_iterations=args.conversion_iterations,
        prune_ratio=args.prune_ratio,
        port=args.port,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
