from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping

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


def _read_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_training_manifest(variant_root: Path) -> Path | None:
    candidates = list(variant_root.rglob("manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def _base_metrics(dataset: Mapping) -> dict:
    metrics = empty_metrics()
    metrics.update(
        input_frame_count=len(dataset["frames"]),
        train_frame_count=len(dataset["train_frame_sha256"]),
        holdout_frame_count=len(dataset["holdout_frame_sha256"]),
        camera_pose_available=True,
    )
    return metrics


def _failed_command_and_code(
    exc: Exception,
    manifest: Mapping | None,
    phase: str,
) -> tuple[list[str], int]:
    if isinstance(exc, subprocess.CalledProcessError):
        command = list(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        return command or ["quality-sweep"], int(exc.returncode)

    if manifest:
        phase_record = manifest.get("export") if phase == "export" else manifest.get("training")
        if isinstance(phase_record, Mapping):
            command = phase_record.get("command")
            return_code = phase_record.get("return_code")
            if isinstance(command, list) and command:
                return list(map(str, command)), int(return_code) if isinstance(return_code, int) else 0
    return ["quality-sweep", phase], 0


def _best_effort_artifact(result: Mapping | None, manifest_path: Path | None) -> tuple[Path | None, dict | None]:
    if not result or manifest_path is None:
        return None, None
    output = result.get("output")
    if not isinstance(output, Mapping) or not output.get("ply_path"):
        return None, None
    ply_path = manifest_path.parent / str(output["ply_path"])
    if not ply_path.is_file():
        return None, None
    try:
        return ply_path, artifact_record(ply_path, format="ply")
    except Exception:
        return ply_path, None


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
    """Run and evaluate default, scale-regularized and MCMC Splatfacto variants.

    One deterministic source-hash/frame-hash contract is written before training. The
    generated transforms JSON uses Nerfstudio's explicit train/val/test filename lists,
    so all variants train and evaluate on the same exact images without modifying the
    prepared dataset. Each variant is attempted independently; one failure does not
    erase or prevent evidence from the remaining variants. The overall command still
    fails after all attempts if any variant failed.
    """
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

    dataset = build_nerfstudio_dataset_contract(
        source,
        transforms,
        holdout_count=holdout_count,
    )
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
        "status": "running",
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
    backend_results: list[dict] = []
    failed_variants: list[str] = []

    for variant in ("default", "scale-regularized", "mcmc"):
        variant_root = root / variant
        started = time.perf_counter()
        result: dict | None = None
        manifest_path: Path | None = None
        evaluation: dict | None = None
        phase = "training"

        try:
            result = run_splatfacto_export(
                split_transforms,
                variant_root,
                train_extra_args=quality_sweep_train_args(iterations=iterations, variant=variant),
                timeout=timeout,
                env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
            )
            train_export_seconds = time.perf_counter() - started
            manifest_path = Path(result["manifest_path"])
            run_dir = manifest_path.parent
            ply_path = run_dir / result["output"]["ply_path"]
            config_path = run_dir / result["training"]["config_path"]

            phase = "evaluation"
            evaluation = run_nerfstudio_eval(
                config_path,
                run_dir / "evaluation",
                timeout=timeout,
                env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
            )

            phase = "artifact-metrics"
            metrics = _base_metrics(dataset)
            metrics.update(
                reconstruction_success=True,
                psnr=evaluation["metrics"].get("psnr"),
                ssim=evaluation["metrics"].get("ssim"),
                lpips=evaluation["metrics"].get("lpips"),
                wall_clock_seconds=train_export_seconds,
                peak_gpu_memory_bytes=(result.get("training") or {}).get("peak_gpu_memory_bytes"),
                **gaussian_artifact_metrics(ply_path),
            )
            backend_result = {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "backend": {
                    "name": f"splatfacto-{variant}",
                    "upstream_revision": environment["nerfstudio_revision"],
                },
                "command": list(result["training"]["command"]),
                "config": {"variant": variant, "iterations": iterations},
                "return_code": 0,
                "status": "success",
                "failure_phase": None,
                "artifact": artifact_record(ply_path, format="ply"),
                "metrics": metrics,
                "training_manifest_path": str(manifest_path),
                "evaluation_manifest_path": evaluation["manifest_path"],
            }
            validate_backend_result(backend_result, dataset)
            result_path = variant_root / "backend-result.json"
            write_json(result_path, backend_result)
            backend_results.append(backend_result)
            summary["variants"].append(
                {
                    "variant": variant,
                    "status": "success",
                    "manifest_path": str(manifest_path),
                    "evaluation_manifest_path": evaluation["manifest_path"],
                    "backend_result_path": str(result_path),
                    "ply_path": str(ply_path),
                    "ply_sha256": result["output"]["sha256"],
                    "ply_size_bytes": result["output"]["size_bytes"],
                    "metrics": metrics,
                    "render_count": evaluation.get("render_count"),
                    "renders": evaluation.get("renders"),
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failed_variants.append(variant)
            if manifest_path is None:
                manifest_path = _latest_training_manifest(variant_root)
            training_manifest = _read_json(manifest_path)
            failure_phase = phase
            if training_manifest and training_manifest.get("status") == "failed":
                failure_phase = str(training_manifest.get("failed_phase") or failure_phase)

            evaluation_manifest_path = None
            evaluation_manifest = None
            if manifest_path is not None:
                candidate = manifest_path.parent / "evaluation" / "eval-manifest.json"
                if candidate.is_file():
                    evaluation_manifest_path = candidate
                    evaluation_manifest = _read_json(candidate)
            if failure_phase == "evaluation" and evaluation_manifest:
                command_record = evaluation_manifest
                command = list(map(str, command_record.get("command") or ["ns-eval"]))
                return_code = int(command_record.get("return_code") or 0)
            else:
                command, return_code = _failed_command_and_code(exc, training_manifest, failure_phase)

            metrics = _base_metrics(dataset)
            training = (training_manifest or {}).get("training") or {}
            metrics.update(
                reconstruction_success=result is not None,
                wall_clock_seconds=elapsed,
                peak_gpu_memory_bytes=(training.get("peak_gpu_memory_bytes") if isinstance(training, Mapping) else None),
            )
            ply_path, artifact = _best_effort_artifact(result, manifest_path)
            if ply_path is not None:
                try:
                    metrics.update(gaussian_artifact_metrics(ply_path))
                except Exception:
                    pass

            backend_result = {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "backend": {
                    "name": f"splatfacto-{variant}",
                    "upstream_revision": environment["nerfstudio_revision"],
                },
                "command": command,
                "config": {"variant": variant, "iterations": iterations},
                "return_code": return_code,
                "status": "failed",
                "failure_phase": failure_phase,
                "artifact": artifact,
                "metrics": metrics,
                "training_manifest_path": str(manifest_path) if manifest_path else None,
                "evaluation_manifest_path": (
                    str(evaluation_manifest_path) if evaluation_manifest_path else None
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
            validate_backend_result(backend_result, dataset)
            result_path = variant_root / "backend-result.json"
            write_json(result_path, backend_result)
            backend_results.append(backend_result)
            summary["variants"].append(
                {
                    "variant": variant,
                    "status": "failed",
                    "failure_phase": failure_phase,
                    "error": backend_result["error"],
                    "manifest_path": str(manifest_path) if manifest_path else None,
                    "evaluation_manifest_path": (
                        str(evaluation_manifest_path) if evaluation_manifest_path else None
                    ),
                    "backend_result_path": str(result_path),
                    "ply_path": str(ply_path) if ply_path else None,
                    "metrics": metrics,
                }
            )

        write_json(summary_path, summary)

    comparison = write_comparison(comparison_path, backend_results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["comparison"] = comparison
    summary["failed_variants"] = failed_variants
    summary["status"] = "failed" if failed_variants else "success"
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)

    if failed_variants:
        raise RuntimeError(
            "quality sweep attempted all variants but failed: " + ", ".join(failed_variants)
        )
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


if __name__ == "__main__":
    main()
