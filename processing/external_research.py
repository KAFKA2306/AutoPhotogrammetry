from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from processing.backend_evaluation import dataset_identity
from processing.nerfstudio import _run_recorded_command_with_peak_gpu_memory
from processing.provenance import sha256_file, write_json


def read_json_object(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def git_head(checkout: str | Path) -> str:
    root = Path(checkout).expanduser().resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"not a readable Git checkout: {root}")
    return completed.stdout.strip()


def verify_checkout(
    checkout: str | Path,
    *,
    expected_revision: str,
    required_paths: Sequence[str] = (),
) -> dict:
    root = Path(checkout).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkout does not exist: {root}")
    revision = git_head(root)
    if revision != expected_revision:
        raise ValueError(
            f"checkout revision mismatch: expected {expected_revision}, got {revision}"
        )
    missing = [path for path in required_paths if not (root / path).exists()]
    if missing:
        raise ValueError(f"pinned checkout is missing required paths: {missing}")
    return {"checkout": str(root), "revision": revision}


def _frame_sources(transforms_json: Path) -> dict[str, Path]:
    meta = read_json_object(transforms_json)
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Nerfstudio transforms.json requires one or more frames")
    by_hash: dict[str, Path] = {}
    for frame in frames:
        if not isinstance(frame, Mapping) or not frame.get("file_path"):
            raise ValueError("Nerfstudio frame is missing file_path")
        declared = Path(str(frame["file_path"]))
        source = (
            declared if declared.is_absolute() else transforms_json.parent / declared
        ).resolve()
        if not source.is_file():
            raise ValueError(f"Nerfstudio frame does not exist: {source}")
        digest = sha256_file(source)
        if digest in by_hash:
            raise ValueError(f"duplicate frame SHA-256 in transforms: {digest}")
        by_hash[digest] = source
    return by_hash


def materialize_frozen_images(
    dataset: Mapping,
    transforms_json: str | Path,
    images_dir: str | Path,
    *,
    preserve_basename: bool = True,
) -> dict:
    """Materialize exactly the frozen #26 frame bytes without inventing a new split."""
    transforms = Path(transforms_json).expanduser().resolve()
    sources = _frame_sources(transforms)
    records = dataset.get("frames") or []
    expected = {str(record["sha256"]): dict(record) for record in records}
    if not expected:
        raise ValueError("dataset contract has no frames")
    if set(expected) != set(sources):
        missing = sorted(set(expected) - set(sources))
        extra = sorted(set(sources) - set(expected))
        raise ValueError(
            f"dataset/transforms frame mismatch; missing={missing}, extra={extra}"
        )

    destination = Path(images_dir).expanduser().resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    used_names: set[str] = set()
    materialized = []
    for index, record in enumerate(records, start=1):
        digest = str(record["sha256"])
        source = sources[digest]
        if preserve_basename:
            name = source.name
        else:
            name = f"{index:04d}-{digest[:12]}{source.suffix.lower() or '.jpg'}"
        if name in used_names:
            raise ValueError(f"duplicate materialized frame name: {name}")
        used_names.add(name)
        target = destination / name
        shutil.copy2(source, target)
        actual = sha256_file(target)
        if actual != digest:
            raise RuntimeError(
                f"materialized frame hash mismatch: expected {digest}, got {actual}"
            )
        materialized.append(
            {
                "name": name,
                "path": str(target),
                "sha256": digest,
                "size_bytes": target.stat().st_size,
                "split": record.get("split"),
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": dataset_identity(dataset),
        "images_dir": str(destination),
        "frame_count": len(materialized),
        "frames": materialized,
    }


def run_recorded_gpu_step(
    command: Sequence[str],
    *,
    cwd: str | Path,
    log_root: str | Path,
    name: str,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run one external argv with logs and best-effort process-tree peak GPU memory."""
    root = Path(log_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(cwd).expanduser().resolve()
    started = time.perf_counter()
    completed, peak_gpu_memory = _run_recorded_command_with_peak_gpu_memory(
        list(map(str, command)),
        cwd=workdir,
        timeout=timeout,
        env=None if env is None else {**os.environ, **dict(env)},
    )
    elapsed = time.perf_counter() - started
    stdout = root / f"{name}.stdout.log"
    stderr = root / f"{name}.stderr.log"
    stdout.write_text(completed.stdout or "", encoding="utf-8")
    stderr.write_text(completed.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "command": list(map(str, command)),
        "cwd": str(workdir),
        "return_code": completed.returncode,
        "wall_clock_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "stdout_log": str(stdout),
        "stderr_log": str(stderr),
    }


def require_success(step: Mapping) -> None:
    return_code = step.get("return_code")
    if return_code != 0:
        raise subprocess.CalledProcessError(
            int(return_code if isinstance(return_code, int) else 1),
            list(step.get("command") or [str(step.get("name") or "external-step")]),
        )


def file_record(path: str | Path) -> dict:
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise ValueError(f"artifact is missing or empty: {artifact}")
    return {
        "path": str(artifact),
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def write_tree_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    suffixes: Sequence[str] | None = None,
) -> dict:
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"artifact directory does not exist: {source}")
    allowed = None if suffixes is None else {suffix.lower() for suffix in suffixes}
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if allowed is not None and path.suffix.lower() not in allowed:
            continue
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"artifact directory contains no selected files: {source}")
    manifest = {
        "schema_version": 1,
        "root": str(source),
        "file_count": len(files),
        "files": files,
    }
    write_json(output_path, manifest)
    return {**manifest, "manifest_path": str(Path(output_path).expanduser().resolve())}
