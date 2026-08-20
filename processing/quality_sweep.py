from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from processing.backend_evaluation import (
    artifact_record,
    build_nerfstudio_dataset_contract,
    dataset_identity,
    empty_metrics,
    gaussian_artifact_metrics,
    validate_backend_result,
    write_comparison,
    write_nerfstudio_split_transforms,
)
from processing.gpu_memory import ComputeMemoryMonitor
from processing.mcmc_quality import DEFAULT_NERFSTUDIO_SOURCE, verify_research_environment
from processing.nerfstudio import run_nerfstudio_eval, run_splatfacto_export
from processing.provenance import write_json


def verify_gpu_runtime() -> dict:
    """Fail before training unless the single quality container has the expected GPU runtime."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the GPU execution image") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the quality container")
    capability = tuple(torch.cuda.get_device_capability())
    if capability[0] != 12:
        raise RuntimeError(
            f"Expected compute capability 12.x for the pinned sm_120 image, got {capability}"
        )
    return {
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "container_image_ref": os.environ.get("AUTOPHOTOGRAMMETRY_IMAGE_REF"),
        "container_image_id": os.environ.get("AUTOPHOTOGRAMMETRY_IMAGE_ID"),
    }


def quality_sweep_train_args(*, iterations: int, variant: str) -> tuple[str, ...]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    args = [
        "--max-num-iterations",
        str(iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    if variant == "default":
        pass
    elif variant == "scale-regularized":
        args.extend(("--pipeline.model.use-scale-regularization", "True"))
    elif variant == "mcmc":
        args.extend(("--pipeline.model.strategy", "mcmc"))
    else:
        raise ValueError(f"unknown quality sweep variant: {variant}")
    return tuple(args)


def winner_train_args(*, iterations: int, winner: str) -> tuple[str, ...]:
    """Return the exact #43 strategy args so follow-up experiments can freeze it."""
    return quality_sweep_train_args(iterations=iterations, variant=winner)


def _latest_splat_manifest(root: Path) -> dict | None:
    manifests = sorted(
        root.rglob("manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )
    if not manifests:
        return None
    try:
        return json.loads(manifests[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _failed_result_details(
    experiment_root: Path,
    exc: Exception,
    fallback_command: Sequence[str],
    *,
    default_phase: str,
) -> tuple[list[str], int, str]:
    manifest = _latest_splat_manifest(experiment_root)
    if manifest:
        training = manifest.get("training") or {}
        export = manifest.get("export") or {}
        command = training.get("command") or export.get("command") or list(fallback_command)
        if manifest.get("failed_phase") == "export" and export.get("command"):
            command = export["command"]
        return_code = training.get("return_code")
        if manifest.get("failed_phase") == "export":
            return_code = export.get("return_code")
        if isinstance(return_code, int):
            return list(map(str, command)), return_code, str(manifest.get("failed_phase") or default_phase)
    if isinstance(exc, subprocess.CalledProcessError):
        command = exc.cmd if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        return list(map(str, command)), int(exc.returncode), default_phase
    return list(map(str, fallback_command)), 1, default_phase


def run_splatfacto_experiment(
    *,
    name: str,
    dataset: Mapping,
    split_transforms: str | Path,
    output_root: str | Path,
    environment: Mapping,
    train_args: Sequence[str],
    config: Mapping,
    timeout: float | None = None,
) -> tuple[dict, dict]:
    """Run one independent Splatfacto experiment and always emit a common result row.

    A failed experiment remains failed and does not prevent later experiments from
    running.  Peak GPU memory is measured from nvidia-smi compute-process memory as a
    baseline delta; when that measurement cannot be obtained it remains null.
    """
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    split_path = Path(split_transforms).expanduser().resolve()
    dataset_id = dataset_identity(dataset)
    fallback_command = ["ns-train", "splatfacto", "--data", str(split_path), *map(str, train_args)]
    started = time.perf_counter()
    phase = "training-export"
    training_result: dict | None = None
    evaluation: dict | None = None
    train_export_seconds: float | None = None
    monitor = ComputeMemoryMonitor()

    try:
        with monitor:
            training_result = run_splatfacto_export(
                split_path,
                root,
                train_extra_args=train_args,
                timeout=timeout,
                env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
            )
        train_export_seconds = time.perf_counter() - started
        memory = monitor.measurement()
        manifest_path = Path(training_result["manifest_path"])
        run_dir = manifest_path.parent
        ply_path = run_dir / training_result["output"]["ply_path"]
        config_path = run_dir / training_result["training"]["config_path"]
        phase = "evaluation"
        evaluation = run_nerfstudio_eval(
            config_path,
            run_dir / "evaluation",
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        metrics = {
            "reconstruction_success": True,
            "input_frame_count": len(dataset["frames"]),
            "train_frame_count": len(dataset["train_frame_sha256"]),
            "holdout_frame_count": len(dataset["holdout_frame_sha256"]),
            "psnr": evaluation["metrics"].get("psnr"),
            "ssim": evaluation["metrics"].get("ssim"),
            "lpips": evaluation["metrics"].get("lpips"),
            "wall_clock_seconds": train_export_seconds,
            "peak_gpu_memory_bytes": memory.peak_delta_bytes,
            "camera_pose_available": True,
            **gaussian_artifact_metrics(ply_path),
        }
        backend_result = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "backend": {
                "name": name,
                "upstream_revision": environment["nerfstudio_revision"],
            },
            "command": list(training_result["training"]["command"]),
            "config": dict(config),
            "return_code": 0,
            "status": "success",
            "failure_phase": None,
            "artifact": artifact_record(ply_path, format="ply"),
            "metrics": metrics,
            "training_manifest_path": str(manifest_path),
            "evaluation_manifest_path": evaluation["manifest_path"],
            "gpu_memory_measurement": memory.as_dict(),
        }
        validate_backend_result(backend_result, dataset)
        result_path = root / "backend-result.json"
        write_json(result_path, backend_result)
        entry = {
            "name": name,
            "status": "success",
            "manifest_path": str(manifest_path),
            "evaluation_manifest_path": evaluation["manifest_path"],
            "backend_result_path": str(result_path),
            "ply_path": str(ply_path),
            "ply_sha256": training_result["output"]["sha256"],
            "ply_size_bytes": training_result["output"]["size_bytes"],
            "metrics": metrics,
            "gpu_memory_measurement": memory.as_dict(),
        }
        return entry, backend_result
    except Exception as exc:
        if monitor.measurement().samples == 0:
            memory = monitor.measurement()
        else:
            memory = monitor.measurement()
        elapsed = train_export_seconds if train_export_seconds is not None else time.perf_counter() - started
        command, return_code, failure_phase = _failed_result_details(
            root,
            exc,
            fallback_command,
            default_phase=phase,
        )
        metrics = empty_metrics()
        metrics.update(
            {
                "reconstruction_success": False,
                "input_frame_count": len(dataset["frames"]),
                "train_frame_count": len(dataset["train_frame_sha256"]),
                "holdout_frame_count": len(dataset["holdout_frame_sha256"]),
                "wall_clock_seconds": elapsed,
                "peak_gpu_memory_bytes": memory.peak_delta_bytes,
                "camera_pose_available": True,
            }
        )
        artifact = None
        training_manifest_path = None
        if training_result is not None:
            training_manifest_path = training_result.get("manifest_path")
            try:
                run_dir = Path(training_manifest_path).parent
                ply_path = run_dir / training_result["output"]["ply_path"]
                if ply_path.is_file():
                    artifact = artifact_record(ply_path, format="ply")
            except (KeyError, TypeError, ValueError):
                artifact = None
        backend_result = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "backend": {
                "name": name,
                "upstream_revision": environment["nerfstudio_revision"],
            },
            "command": command,
            "config": dict(config),
            "return_code": return_code,
            "status": "failed",
            "failure_phase": failure_phase,
            "artifact": artifact,
            "metrics": metrics,
            "training_manifest_path": training_manifest_path,
            "evaluation_manifest_path": None if evaluation is None else evaluation.get("manifest_path"),
            "gpu_memory_measurement": memory.as_dict(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        validate_backend_result(backend_result, dataset)
        result_path = root / "backend-result.json"
        write_json(result_path, backend_result)
        return (
            {
                "name": name,
                "status": "failed",
                "backend_result_path": str(result_path),
                "failure_phase": failure_phase,
                "metrics": metrics,
                "gpu_memory_measurement": memory.as_dict(),
                "error": backend_result["error"],
            },
            backend_result,
        )


def run_quality_sweep(
    data_dir: str | Path,
    source_video: str | Path,
    output_root: str | Path,
    *,
    nerfstudio_source: str | Path = DEFAULT_NERFSTUDIO_SOURCE,
    iterations: int = 30000,
    holdout_count: int | None = None,
    timeout: float | None = None,
) -> dict:
    """Run and evaluate default, scale-regularized and MCMC Splatfacto variants."""
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"Nerfstudio data directory does not exist: {data}")
    transforms = data / "transforms.json"
    if not transforms.is_file():
        raise ValueError(f"Nerfstudio transforms.json does not exist: {transforms}")
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source video does not exist: {source}")

    runtime = verify_gpu_runtime()
    environment = verify_research_environment(nerfstudio_source)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=holdout_count)
    dataset_path = root / "evaluation-dataset.json"
    write_json(dataset_path, dataset)
    split = write_nerfstudio_split_transforms(
        transforms,
        dataset,
        root / "evaluation-transforms.json",
    )
    split_transforms = Path(split["transforms_path"])
    dataset_id = dataset_identity(dataset)

    summary_path = root / "quality-sweep.json"
    comparison_path = root / "comparison.json"
    summary = {
        "schema_version": 4,
        "data_dir": str(data),
        "source_video": str(source),
        "dataset_id": dataset_id,
        "dataset_manifest_path": str(dataset_path),
        "split_transforms_path": str(split_transforms),
        "train_frame_sha256": list(dataset["train_frame_sha256"]),
        "holdout_frame_sha256": list(dataset["holdout_frame_sha256"]),
        "runtime": runtime,
        "environment": environment,
        "iterations": iterations,
        "variants": [],
    }
    backend_results = []

    for variant in ("default", "scale-regularized", "mcmc"):
        args = quality_sweep_train_args(iterations=iterations, variant=variant)
        entry, backend_result = run_splatfacto_experiment(
            name=f"splatfacto-{variant}",
            dataset=dataset,
            split_transforms=split_transforms,
            output_root=root / variant,
            environment=environment,
            train_args=args,
            config={"variant": variant, "iterations": iterations},
            timeout=timeout,
        )
        summary["variants"].append({"variant": variant, **entry})
        backend_results.append(backend_result)
        write_json(summary_path, summary)

    comparison = write_comparison(comparison_path, backend_results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["comparison"] = comparison
    summary["all_variants_succeeded"] = all(
        entry["status"] == "success" for entry in summary["variants"]
    )
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run default, scale-regularized and MCMC Splatfacto on one deterministic "
            "train/hold-out split in the pinned GPU environment."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nerfstudio-source", default=str(DEFAULT_NERFSTUDIO_SOURCE))
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_quality_sweep(
        args.data,
        args.source_video,
        args.output_root,
        nerfstudio_source=args.nerfstudio_source,
        iterations=args.iterations,
        holdout_count=args.holdout_count,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_variants_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
