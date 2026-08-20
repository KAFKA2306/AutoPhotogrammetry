from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from processing.backend_evaluation import (
    artifact_record,
    dataset_identity,
    empty_metrics,
    validate_backend_result,
)
from processing.provenance import sha256_file, write_json

VGGT_REPOSITORY = "https://github.com/facebookresearch/vggt"
VGGT_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_DEMO_CHECKPOINT_ID = "facebook/VGGT-1B"
VGGT_DEMO_CHECKPOINT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_DEMO_CHECKPOINT_USAGE = "non-commercial-research-only"
VGGT_COMMERCIAL_CHECKPOINT_ID = "facebook/VGGT-1B-Commercial"
REQUIRED_COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"VGGT checkout is not a readable Git checkout: {checkout}")
    return completed.stdout.strip()


def verify_vggt_checkout(checkout: str | Path) -> dict:
    root = Path(checkout).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"VGGT checkout does not exist: {root}")
    revision = _git_head(root)
    if revision != VGGT_REVISION:
        raise ValueError(f"VGGT revision mismatch: expected {VGGT_REVISION}, got {revision}")
    demo = root / "demo_colmap.py"
    if not demo.is_file():
        raise ValueError(f"pinned VGGT checkout is missing demo_colmap.py: {demo}")
    source = demo.read_text(encoding="utf-8")
    if VGGT_DEMO_CHECKPOINT_URL not in source:
        raise ValueError(
            "pinned demo_colmap.py no longer contains the expected research checkpoint URL; "
            "re-audit checkpoint semantics before execution"
        )
    return {
        "repository": VGGT_REPOSITORY,
        "revision": revision,
        "checkout": str(root),
        "demo_script": str(demo),
        "checkpoint": {
            "id": VGGT_DEMO_CHECKPOINT_ID,
            "url": VGGT_DEMO_CHECKPOINT_URL,
            "usage": VGGT_DEMO_CHECKPOINT_USAGE,
            "production_eligible": False,
            "commercial_checkpoint_id": VGGT_COMMERCIAL_CHECKPOINT_ID,
            "note": (
                "The pinned official demo hard-codes VGGT-1B. It is used only for research "
                "comparison; commercial checkpoint access is not inferred or fabricated."
            ),
        },
    }


def _read_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def _frame_sources(transforms_json: Path) -> dict[str, Path]:
    meta = _read_json(transforms_json)
    frames = meta.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Nerfstudio transforms.json requires frames")
    by_hash: dict[str, Path] = {}
    for frame in frames:
        declared = frame.get("file_path")
        if not declared:
            raise ValueError("Nerfstudio frame is missing file_path")
        candidate = Path(str(declared))
        source = (
            candidate if candidate.is_absolute() else transforms_json.parent / candidate
        ).resolve()
        if not source.is_file():
            raise ValueError(f"Nerfstudio frame does not exist: {source}")
        digest = sha256_file(source)
        if digest in by_hash:
            raise ValueError(f"duplicate frame SHA-256 in transforms: {digest}")
        by_hash[digest] = source
    return by_hash


def materialize_vggt_scene(
    dataset: Mapping,
    transforms_json: str | Path,
    scene_dir: str | Path,
) -> dict:
    """Copy the exact #26 frame set into the official VGGT `<scene>/images` layout."""
    transforms = Path(transforms_json).expanduser().resolve()
    sources = _frame_sources(transforms)
    scene = Path(scene_dir).expanduser().resolve()
    if scene.exists():
        shutil.rmtree(scene)
    images = scene / "images"
    images.mkdir(parents=True)

    expected = {str(record["sha256"]): dict(record) for record in dataset.get("frames") or []}
    if not expected:
        raise ValueError("dataset contract has no frames")
    if set(expected) != set(sources):
        missing = sorted(set(expected) - set(sources))
        extra = sorted(set(sources) - set(expected))
        raise ValueError(f"dataset/transforms frame mismatch; missing={missing}, extra={extra}")

    used_names: set[str] = set()
    records = []
    for index, record in enumerate(dataset["frames"], start=1):
        digest = str(record["sha256"])
        source = sources[digest]
        suffix = source.suffix.lower() or ".jpg"
        name = f"{index:04d}-{digest[:12]}{suffix}"
        if name in used_names:
            raise ValueError(f"duplicate materialized VGGT frame name: {name}")
        used_names.add(name)
        destination = images / name
        shutil.copy2(source, destination)
        actual = sha256_file(destination)
        if actual != digest:
            raise RuntimeError(
                f"materialized VGGT frame hash mismatch: expected {digest}, got {actual}"
            )
        records.append(
            {
                "name": name,
                "source_path": str(source),
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
                "split": record.get("split"),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_identity(dataset),
        "scene_dir": str(scene),
        "images_dir": str(images),
        "frame_count": len(records),
        "frames": records,
    }
    write_json(scene / "input-manifest.json", manifest)
    return manifest


def _run_command(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(map(str, command)),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _colmap_artifact(scene: Path) -> dict:
    sparse = scene / "sparse"
    files = []
    for name in REQUIRED_COLMAP_FILES:
        path = sparse / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"VGGT did not produce required COLMAP artifact: {path}")
        files.append(
            {
                "path": str(path),
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    points = sparse / "points.ply"
    if points.is_file() and points.stat().st_size > 0:
        files.append(
            {
                "path": str(points),
                "name": points.name,
                "size_bytes": points.stat().st_size,
                "sha256": sha256_file(points),
            }
        )
    manifest = {
        "schema_version": 1,
        "format": "colmap-sparse",
        "files": files,
    }
    manifest_path = scene / "colmap-artifact.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def run_vggt_colmap(
    dataset_json: str | Path,
    transforms_json: str | Path,
    checkout: str | Path,
    output_root: str | Path,
    *,
    python_executable: str = "python",
    use_ba: bool = False,
) -> dict:
    """Run the pinned official VGGT COLMAP exporter as a research-only backend."""
    dataset = _read_json(dataset_json)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = root / "scene"
    checkout_info = verify_vggt_checkout(checkout)
    input_manifest = materialize_vggt_scene(dataset, transforms_json, scene)

    command = [
        python_executable,
        checkout_info["demo_script"],
        "--scene_dir",
        str(scene),
    ]
    if use_ba:
        command.append("--use_ba")

    started = time.perf_counter()
    completed = _run_command(command, cwd=Path(checkout_info["checkout"]))
    elapsed = time.perf_counter() - started
    (root / "vggt.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (root / "vggt.stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    metrics = empty_metrics()
    metrics.update(
        {
            "input_frame_count": len(dataset.get("frames") or []),
            "train_frame_count": len(dataset.get("train_frame_sha256") or []),
            "holdout_frame_count": len(dataset.get("holdout_frame_sha256") or []),
            "wall_clock_seconds": elapsed,
            "peak_gpu_memory_bytes": None,
            "camera_pose_available": completed.returncode == 0,
            "reconstruction_success": completed.returncode == 0,
        }
    )

    artifact = None
    failure_phase = None
    error = None
    status = "success"
    if completed.returncode != 0:
        status = "failed"
        failure_phase = "vggt-colmap"
        error = f"VGGT demo_colmap.py exited with code {completed.returncode}"
    else:
        try:
            artifact_manifest = _colmap_artifact(scene)
            artifact = artifact_record(
                artifact_manifest["manifest_path"],
                format="colmap-sparse-manifest",
            )
        except Exception as exc:
            status = "failed"
            failure_phase = "artifact-validation"
            metrics["reconstruction_success"] = False
            metrics["camera_pose_available"] = False
            error = f"{type(exc).__name__}: {exc}"

    result = {
        "schema_version": 1,
        "dataset_id": dataset_identity(dataset),
        "backend": {
            "name": "vggt-colmap-research",
            "upstream_revision": checkout_info["revision"],
        },
        "command": command,
        "config": {
            "use_ba": use_ba,
            "checkpoint": checkout_info["checkpoint"],
            "input_manifest": str(scene / "input-manifest.json"),
            "production_eligible": False,
        },
        "return_code": completed.returncode,
        "status": status,
        "failure_phase": failure_phase,
        "artifact": artifact,
        "metrics": metrics,
        "error": error,
        "upstream": checkout_info,
        "input": input_manifest,
        "stdout_log": str(root / "vggt.stdout.log"),
        "stderr_log": str(root / "vggt.stderr.log"),
    }
    validate_backend_result(result, dataset)
    result_path = root / "backend-result.json"
    write_json(result_path, result)
    return {**result, "manifest_path": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned official VGGT demo_colmap.py as a research-only camera/geometry backend."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--use-ba", action="store_true")
    args = parser.parse_args()
    result = run_vggt_colmap(
        args.dataset,
        args.transforms,
        args.checkout,
        args.output_root,
        python_executable=args.python,
        use_ba=args.use_ba,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
