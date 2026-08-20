from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

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
from processing.nerfstudio import (
    nerfstudio_process_images_command,
    run_nerfstudio_eval,
    run_splatfacto_export,
)
from processing.provenance import sha256_file, write_json
from processing.quality_sweep import quality_sweep_train_args, verify_gpu_runtime

REQUIRED_COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def _read_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def _hash_files(root: Path) -> dict[str, Path]:
    by_hash: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest in by_hash:
            raise ValueError(f"duplicate frame SHA-256 in VGGT image directory: {digest}")
        by_hash[digest] = path
    return by_hash


def _materialize_exact_images(dataset: Mapping, source_images: Path, destination: Path) -> list[dict]:
    """Copy the frozen #26 image bytes into Nerfstudio's required `images/` directory."""
    expected = {str(record["sha256"]): dict(record) for record in dataset.get("frames") or []}
    actual = _hash_files(source_images)
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"VGGT images do not match the frozen dataset; missing={missing}, extra={extra}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    records = []
    for digest, source in sorted(actual.items()):
        target = destination / source.name
        shutil.copy2(source, target)
        copied = sha256_file(target)
        if copied != digest:
            raise RuntimeError(f"copied frame hash mismatch: expected {digest}, got {copied}")
        records.append({"path": str(target), "sha256": digest, "size_bytes": target.stat().st_size})
    return records


def _verify_colmap_sparse(sparse: Path) -> list[dict]:
    files = []
    for name in REQUIRED_COLMAP_FILES:
        path = sparse / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"required VGGT COLMAP file is missing: {path}")
        files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _run_recorded(command: Sequence[str], *, cwd: Path, stdout: Path, stderr: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(map(str, command)),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    stdout.write_text(completed.stdout or "", encoding="utf-8")
    stderr.write_text(completed.stderr or "", encoding="utf-8")
    return completed


def prepare_vggt_nerfstudio_data(
    dataset: Mapping,
    vggt_scene: str | Path,
    output_dir: str | Path,
    *,
    process_executable: str = "ns-process-data",
) -> dict:
    """Convert a VGGT COLMAP model without re-running COLMAP or changing frame bytes."""
    scene = Path(vggt_scene).expanduser().resolve()
    source_images = scene / "images"
    sparse = scene / "sparse"
    if not source_images.is_dir():
        raise ValueError(f"VGGT image directory does not exist: {source_images}")
    sparse_files = _verify_colmap_sparse(sparse)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    image_records = _materialize_exact_images(dataset, source_images, images)

    command = nerfstudio_process_images_command(
        images,
        output,
        executable=process_executable,
        extra_args=(
            "--skip-colmap",
            "--skip-image-processing",
            "--colmap-model-path",
            str(sparse),
        ),
    )
    completed = _run_recorded(
        command,
        cwd=output,
        stdout=output / "ns-process-data.stdout.log",
        stderr=output / "ns-process-data.stderr.log",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    transforms = output / "transforms.json"
    if not transforms.is_file():
        raise RuntimeError("ns-process-data succeeded but transforms.json is missing")

    return {
        "command": command,
        "return_code": completed.returncode,
        "transforms_path": str(transforms),
        "images": image_records,
        "colmap_sparse": sparse_files,
        "stdout_log": str(output / "ns-process-data.stdout.log"),
        "stderr_log": str(output / "ns-process-data.stderr.log"),
    }


def run_vggt_splatfacto(
    dataset_json: str | Path,
    source_video: str | Path,
    vggt_scene: str | Path,
    output_root: str | Path,
    *,
    baseline_result_json: str | Path | None = None,
    iterations: int = 30000,
    nerfstudio_source: str | Path = DEFAULT_NERFSTUDIO_SOURCE,
    timeout: float | None = None,
) -> dict:
    """Run VGGT camera geometry through the same default Splatfacto/eval contract as baseline."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    dataset = _read_json(dataset_json)
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source video does not exist: {source}")

    runtime = verify_gpu_runtime()
    environment = verify_research_environment(nerfstudio_source)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    prepared = prepare_vggt_nerfstudio_data(dataset, vggt_scene, root / "nerfstudio-data")
    rebuilt = build_nerfstudio_dataset_contract(
        source,
        prepared["transforms_path"],
        holdout_count=len(dataset.get("holdout_frame_sha256") or []),
    )
    expected_id = dataset_identity(dataset)
    actual_id = dataset_identity(rebuilt)
    if actual_id != expected_id:
        raise RuntimeError(
            f"VGGT-derived Nerfstudio data changed the frozen dataset identity: expected {expected_id}, got {actual_id}"
        )

    split = write_nerfstudio_split_transforms(
        prepared["transforms_path"],
        rebuilt,
        root / "evaluation-transforms.json",
    )
    split_path = Path(split["transforms_path"])

    started = time.perf_counter()
    result = run_splatfacto_export(
        split_path,
        root / "run",
        train_extra_args=quality_sweep_train_args(iterations=iterations, variant="default"),
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
        "input_frame_count": len(rebuilt["frames"]),
        "train_frame_count": len(rebuilt["train_frame_sha256"]),
        "holdout_frame_count": len(rebuilt["holdout_frame_sha256"]),
        "psnr": evaluation["metrics"].get("psnr"),
        "ssim": evaluation["metrics"].get("ssim"),
        "lpips": evaluation["metrics"].get("lpips"),
        "wall_clock_seconds": train_export_seconds,
        "peak_gpu_memory_bytes": (result.get("training") or {}).get("peak_gpu_memory_bytes"),
        "camera_pose_available": True,
        **gaussian_artifact_metrics(ply_path),
    }
    backend_result = {
        "schema_version": 2,
        "dataset_id": expected_id,
        "backend": {
            "name": "vggt-colmap-splatfacto",
            "upstream_revision": f"vggt:a288dd0f14786c93483e45524328726ab7b1b4ce;nerfstudio:{environment['nerfstudio_revision']}",
        },
        "command": list(result["training"]["command"]),
        "config": {
            "iterations": iterations,
            "camera_backend": "vggt-colmap-research",
            "splatfacto_variant": "default",
            "prepared_data": prepared,
        },
        "return_code": 0,
        "status": "success",
        "failure_phase": None,
        "artifact": artifact_record(ply_path, format="ply"),
        "metrics": metrics,
        "training_manifest_path": str(manifest_path),
        "evaluation_manifest_path": evaluation["manifest_path"],
    }
    validate_backend_result(backend_result, rebuilt)
    backend_path = root / "backend-result.json"
    write_json(backend_path, backend_result)

    comparison_path = None
    if baseline_result_json is not None:
        baseline = _read_json(baseline_result_json)
        validate_backend_result(baseline, rebuilt)
        comparison_path = root / "comparison.json"
        write_comparison(comparison_path, [baseline, backend_result], rebuilt)

    summary = {
        "schema_version": 1,
        "status": "success",
        "dataset_id": expected_id,
        "runtime": runtime,
        "environment": environment,
        "prepared": prepared,
        "backend_result_path": str(backend_path),
        "comparison_path": None if comparison_path is None else str(comparison_path),
        "ply_path": str(ply_path),
        "ply_sha256": result["output"]["sha256"],
        "ply_size_bytes": result["output"]["size_bytes"],
        "metrics": metrics,
        "render_count": evaluation.get("render_count"),
        "renders": evaluation.get("renders"),
    }
    summary_path = root / "vggt-splatfacto.json"
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the pinned VGGT COLMAP result to Nerfstudio, run default Splatfacto, and evaluate it."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--vggt-scene", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-result")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--nerfstudio-source", default=str(DEFAULT_NERFSTUDIO_SOURCE))
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_vggt_splatfacto(
        args.dataset,
        args.source_video,
        args.vggt_scene,
        args.output_root,
        baseline_result_json=args.baseline_result,
        iterations=args.iterations,
        nerfstudio_source=args.nerfstudio_source,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
