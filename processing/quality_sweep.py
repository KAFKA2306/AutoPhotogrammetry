from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from processing.backend_evaluation import (
    artifact_record,
    build_nerfstudio_dataset_contract,
    dataset_identity,
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
    prepared dataset. ns-eval records PSNR/SSIM/LPIPS and saves the hold-out renders.
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
        "schema_version": 3,
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
        started = time.perf_counter()
        result = run_splatfacto_export(
            split_transforms,
            root / variant,
            train_extra_args=quality_sweep_train_args(iterations=iterations, variant=variant),
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        train_export_seconds = time.perf_counter() - started
        manifest_path = Path(result["manifest_path"])
        run_dir = manifest_path.parent
        ply_path = run_dir / result["output"]["ply_path"]
        config_path = run_dir / result["training"]["config_path"]
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
            "peak_gpu_memory_bytes": None,
            "camera_pose_available": True,
            **gaussian_artifact_metrics(ply_path),
        }
        backend_result = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "backend": {
                "name": f"splatfacto-{variant}",
                "upstream_revision": environment["nerfstudio_revision"],
            },
            "command": list(result["training"]["command"]),
            "config": {
                "variant": variant,
                "iterations": iterations,
            },
            "return_code": 0,
            "status": "success",
            "failure_phase": None,
            "artifact": artifact_record(ply_path, format="ply"),
            "metrics": metrics,
            "training_manifest_path": str(manifest_path),
            "evaluation_manifest_path": evaluation["manifest_path"],
        }
        validate_backend_result(backend_result, dataset)
        result_path = root / variant / "backend-result.json"
        write_json(result_path, backend_result)
        backend_results.append(backend_result)
        summary["variants"].append(
            {
                "variant": variant,
                "manifest_path": str(manifest_path),
                "evaluation_manifest_path": evaluation["manifest_path"],
                "backend_result_path": str(result_path),
                "ply_path": str(ply_path),
                "ply_sha256": result["output"]["sha256"],
                "ply_size_bytes": result["output"]["size_bytes"],
                "metrics": metrics,
            }
        )
        write_json(summary_path, summary)

    comparison = write_comparison(comparison_path, backend_results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["comparison"] = comparison
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


if __name__ == "__main__":
    main()
