from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

from processing.provenance import image_records, sha256_file, write_json


class NerfstudioConfigurationError(RuntimeError):
    """Raised when the external Nerfstudio CLI is not available."""


def nerfstudio_process_images_command(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-process-data",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "images",
        "--data", str(Path(image_dir)),
        "--output-dir", str(Path(output_dir)),
        *map(str, extra_args),
    ]


def splatfacto_train_command(
    data_dir: str | Path,
    *,
    executable: str = "ns-train",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "splatfacto",
        "--data", str(Path(data_dir)),
        *map(str, extra_args),
    ]


def gaussian_splat_export_command(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-export",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "gaussian-splat",
        "--load-config", str(Path(config_path)),
        "--output-dir", str(Path(output_dir)),
        *map(str, extra_args),
    ]


def _resolve_cli(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    else:
        found = shutil.which(str(candidate))
        if found:
            return Path(found).resolve()
    raise NerfstudioConfigurationError(
        f"{executable} was not found. Install Nerfstudio in the execution environment and ensure "
        "its CLI is on PATH, or pass an explicit executable path."
    )


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_recorded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None,
    env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    run_env = None if env is None else {**os.environ, **env}
    try:
        return subprocess.run(
            list(map(str, command)),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout} seconds."
        return subprocess.CompletedProcess(list(command), 124, stdout, stderr)


def run_splatfacto_export(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    train_executable: str = "ns-train",
    export_executable: str = "ns-export",
    train_extra_args: Sequence[str] = (),
    export_extra_args: Sequence[str] = (),
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run external Nerfstudio Splatfacto training and export one auditable PLY."""
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"Nerfstudio data directory does not exist: {data}")

    train_cli = _resolve_cli(train_executable)
    export_cli = _resolve_cli(export_executable)
    nerfstudio_version = _package_version("nerfstudio")
    gsplat_version = _package_version("gsplat")
    if nerfstudio_version is None or gsplat_version is None:
        missing = [
            name
            for name, version in (
                ("nerfstudio", nerfstudio_version),
                ("gsplat", gsplat_version),
            )
            if version is None
        ]
        raise NerfstudioConfigurationError(
            "Installed package version could not be resolved for: "
            + ", ".join(missing)
        )

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).expanduser().resolve() / "splatfacto" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    train_stdout = run_dir / "train.stdout.log"
    train_stderr = run_dir / "train.stderr.log"
    export_stdout = run_dir / "export.stdout.log"
    export_stderr = run_dir / "export.stderr.log"
    input_images = image_records(data)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "input": {
            "data_dir": str(data),
            "image_count": len(input_images),
            "images": input_images,
        },
        "versions": {
            "nerfstudio": nerfstudio_version,
            "gsplat": gsplat_version,
        },
        "training": None,
        "export": None,
        "output": None,
    }

    train_command = splatfacto_train_command(
        data,
        executable=str(train_cli),
        extra_args=train_extra_args,
    )
    train_started = _utc_now()
    train_result = _run_recorded_command(
        train_command,
        cwd=run_dir,
        timeout=timeout,
        env=env,
    )
    train_finished = _utc_now()
    train_stdout.write_text(train_result.stdout or "", encoding="utf-8")
    train_stderr.write_text(train_result.stderr or "", encoding="utf-8")
    manifest["training"] = {
        "command": train_command,
        "started_at": train_started,
        "finished_at": train_finished,
        "return_code": train_result.returncode,
        "stdout_log": train_stdout.name,
        "stderr_log": train_stderr.name,
        "config_path": None,
        "checkpoint_path": None,
    }
    if train_result.returncode != 0:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "training"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(
            train_result.returncode,
            train_command,
            output=train_result.stdout,
            stderr=train_result.stderr,
        )

    configs = sorted({*run_dir.rglob("config.yml"), *run_dir.rglob("config.yaml")})
    if len(configs) != 1:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "config_discovery"
        manifest["config_candidates"] = [
            str(path.relative_to(run_dir)) for path in configs
        ]
        write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Expected exactly one Nerfstudio config, found {len(configs)}"
        )
    config_path = configs[0]

    checkpoints = sorted(
        run_dir.rglob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )
    if not checkpoints:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "checkpoint_discovery"
        write_json(manifest_path, manifest)
        raise RuntimeError(
            "Nerfstudio training succeeded but no checkpoint was found"
        )
    checkpoint_path = checkpoints[-1]
    manifest["training"]["config_path"] = str(config_path.relative_to(run_dir))
    manifest["training"]["checkpoint_path"] = str(
        checkpoint_path.relative_to(run_dir)
    )

    export_dir = run_dir / "export"
    export_dir.mkdir()
    export_command = gaussian_splat_export_command(
        config_path,
        export_dir,
        executable=str(export_cli),
        extra_args=export_extra_args,
    )
    export_started = _utc_now()
    export_result = _run_recorded_command(
        export_command,
        cwd=run_dir,
        timeout=timeout,
        env=env,
    )
    export_finished = _utc_now()
    export_stdout.write_text(export_result.stdout or "", encoding="utf-8")
    export_stderr.write_text(export_result.stderr or "", encoding="utf-8")
    manifest["export"] = {
        "command": export_command,
        "started_at": export_started,
        "finished_at": export_finished,
        "return_code": export_result.returncode,
        "stdout_log": export_stdout.name,
        "stderr_log": export_stderr.name,
    }
    if export_result.returncode != 0:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "export"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(
            export_result.returncode,
            export_command,
            output=export_result.stdout,
            stderr=export_result.stderr,
        )

    ply_files = sorted(export_dir.rglob("*.ply"))
    if len(ply_files) != 1:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "ply_discovery"
        manifest["ply_candidates"] = [
            str(path.relative_to(run_dir)) for path in ply_files
        ]
        write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Expected exactly one exported PLY, found {len(ply_files)}"
        )

    ply_path = ply_files[0]
    manifest["status"] = "success"
    manifest["output"] = {
        "ply_path": str(ply_path.relative_to(run_dir)),
        "size_bytes": ply_path.stat().st_size,
        "sha256": sha256_file(ply_path),
    }
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
