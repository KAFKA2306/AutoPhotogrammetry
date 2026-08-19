from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from processing.gaussian_ply import gaussian_ply_metrics
from processing.provenance import sha256_file, write_json


def clean_gs_command(
    script_path: str | Path,
    *,
    scene: str,
    masks_dir: str | Path,
    input_ply: str | Path,
    output_ply: str | Path,
    python_executable: str = "python",
    color_threshold: float | None = None,
    k_neighbors: int | None = None,
    neighbor_threshold: float | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(Path(script_path)),
        "--scene",
        scene,
        "--masks_dir",
        str(Path(masks_dir)),
        "--input_ply",
        str(Path(input_ply)),
        "--output_ply",
        str(Path(output_ply)),
    ]
    if color_threshold is not None:
        command.extend(("--color_threshold", str(color_threshold)))
    if k_neighbors is not None:
        command.extend(("--k_neighbors", str(k_neighbors)))
    if neighbor_threshold is not None:
        command.extend(("--neighbor_threshold", str(neighbor_threshold)))
    return command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask_records(masks_dir: Path) -> list[dict]:
    masks = sorted(path for path in masks_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if not masks:
        raise ValueError(f"no PNG semantic masks found: {masks_dir}")
    return [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in masks
    ]


def run_clean_gs(
    *,
    script_path: str | Path,
    upstream_revision: str,
    scene: str,
    masks_dir: str | Path,
    input_ply: str | Path,
    output_ply: str | Path,
    manifest_path: str | Path,
    python_executable: str = "python",
    color_threshold: float | None = None,
    k_neighbors: int | None = None,
    neighbor_threshold: float | None = None,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> dict:
    """Run an explicit Clean-GS checkout and record all inputs, argv and before/after artifacts."""
    script = Path(script_path).expanduser().resolve()
    masks = Path(masks_dir).expanduser().resolve()
    source = Path(input_ply).expanduser().resolve()
    output = Path(output_ply).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not script.is_file():
        raise ValueError(f"Clean-GS script does not exist: {script}")
    if not upstream_revision.strip():
        raise ValueError("upstream_revision is required")
    if not masks.is_dir():
        raise ValueError(f"mask directory does not exist: {masks}")
    if not source.is_file():
        raise ValueError(f"input PLY does not exist: {source}")

    mask_records = _mask_records(masks)
    before = gaussian_ply_metrics(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = clean_gs_command(
        script,
        scene=scene,
        masks_dir=masks,
        input_ply=source,
        output_ply=output,
        python_executable=python_executable,
        color_threshold=color_threshold,
        k_neighbors=k_neighbors,
        neighbor_threshold=neighbor_threshold,
    ) + list(map(str, extra_args))
    started = _utc_now()
    result = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        cwd=script.parent,
        timeout=timeout,
    )
    finished = _utc_now()
    manifest = {
        "schema_version": 1,
        "backend": "Clean-GS",
        "upstream_repository": "https://github.com/smlab-niser/clean-gs",
        "upstream_revision": upstream_revision,
        "scene": scene,
        "command": command,
        "started_at": started,
        "finished_at": finished,
        "return_code": result.returncode,
        "masks": mask_records,
        "input": before,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "status": "running",
    }
    if result.returncode != 0:
        manifest.update(status="failed", failure_phase="clean-gs")
        write_json(manifest_file, manifest)
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    if not output.is_file() or output.stat().st_size <= 0:
        manifest.update(status="failed", failure_phase="output-validation")
        write_json(manifest_file, manifest)
        raise RuntimeError("Clean-GS returned success but did not create a non-empty output PLY")

    after = gaussian_ply_metrics(output)
    removed = before["primitive_count"] - after["primitive_count"]
    manifest.update(
        status="success",
        output=after,
        removed_primitive_count=removed,
        removed_primitive_ratio=removed / before["primitive_count"],
    )
    write_json(manifest_file, manifest)
    manifest["manifest_path"] = str(manifest_file)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Clean-GS with auditable input/output lineage.")
    parser.add_argument("--script", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--color-threshold", type=float)
    parser.add_argument("--k-neighbors", type=int)
    parser.add_argument("--neighbor-threshold", type=float)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_clean_gs(
        script_path=args.script,
        upstream_revision=args.upstream_revision,
        scene=args.scene,
        masks_dir=args.masks_dir,
        input_ply=args.input_ply,
        output_ply=args.output_ply,
        manifest_path=args.manifest,
        python_executable=args.python,
        color_threshold=args.color_threshold,
        k_neighbors=args.k_neighbors,
        neighbor_threshold=args.neighbor_threshold,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
