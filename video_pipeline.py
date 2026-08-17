from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class NerfstudioConfigurationError(RuntimeError):
    """Raised when the external Nerfstudio CLI is not available."""


@dataclass(frozen=True)
class VideoSource:
    title: str
    source_page: str
    media_url: str
    author: str
    license: str
    license_url: str
    target: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(map(str, command)),
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> dict:
    completed = run([
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration,size,format_name:stream=codec_name,width,height",
        "-of", "json",
        str(Path(path)),
    ])
    return json.loads(completed.stdout)


def scene_cut_times(
    path: str | Path,
    *,
    threshold: float = 0.4,
    ffmpeg: str = "ffmpeg",
) -> list[float]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    completed = run([
        ffmpeg,
        "-hide_banner",
        "-i", str(Path(path)),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f", "null",
        "-",
    ])
    return [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)]


def extract_frames_command(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 3.0,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    output = Path(output_dir)
    return [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i", str(Path(video_path)),
        "-vf", f"fps={fps:g}",
        "-q:v", "2",
        str(output / "frame-%06d.jpg"),
    ]


def frame_timestamp_records(
    frame_paths: Sequence[str | Path],
    *,
    fps: float,
) -> list[dict]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return [
        {"frame": Path(path).name, "source_time_seconds": index / fps}
        for index, path in enumerate(sorted(map(Path, frame_paths)))
    ]


def select_video_frames(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    sharpness_threshold: float = 0.0001,
    similarity_threshold: float = 0.92,
    sharpness_fn: Callable[[str | Path], float] | None = None,
    similarity_fn: Callable[[str | Path, str | Path], float] | None = None,
) -> dict:
    """Keep sharp frames and remove only near-duplicates of the last accepted frame."""
    if sharpness_threshold < 0:
        raise ValueError("sharpness_threshold must be non-negative")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    if sharpness_fn is None or similarity_fn is None:
        from main import calculate_sharpness, calculate_similarity

        sharpness_fn = sharpness_fn or calculate_sharpness
        similarity_fn = similarity_fn or calculate_similarity

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    rejected_blur = 0
    rejected_duplicate = 0

    for raw_path in sorted(map(Path, frame_paths)):
        if sharpness_fn(raw_path) < sharpness_threshold:
            rejected_blur += 1
            continue
        if selected and similarity_fn(raw_path, selected[-1]) >= similarity_threshold:
            rejected_duplicate += 1
            continue
        target = destination / raw_path.name
        shutil.copy2(raw_path, target)
        selected.append(target)

    return {
        "input": len(frame_paths),
        "selected": len(selected),
        "rejected_blur": rejected_blur,
        "rejected_duplicate": rejected_duplicate,
        "selected_paths": [str(path) for path in selected],
    }


def nerfstudio_process_images_command(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-process-data",
) -> list[str]:
    return [
        executable,
        "images",
        "--data", str(Path(image_dir)),
        "--output-dir", str(Path(output_dir)),
    ]


def splatfacto_train_command(
    data_dir: str | Path,
    *,
    executable: str = "ns-train",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [executable, "splatfacto", "--data", str(Path(data_dir)), *map(str, extra_args)]


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


def _image_records(data_dir: Path) -> list[dict]:
    records = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            records.append({
                "path": path.relative_to(data_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return records


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


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            name for name, version in (("nerfstudio", nerfstudio_version), ("gsplat", gsplat_version))
            if version is None
        ]
        raise NerfstudioConfigurationError(
            "Installed package version could not be resolved for: " + ", ".join(missing)
        )

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).expanduser().resolve() / "splatfacto" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    train_stdout = run_dir / "train.stdout.log"
    train_stderr = run_dir / "train.stderr.log"
    export_stdout = run_dir / "export.stdout.log"
    export_stderr = run_dir / "export.stderr.log"
    input_images = _image_records(data)

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
        _write_json(manifest_path, manifest)
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
        manifest["config_candidates"] = [str(path.relative_to(run_dir)) for path in configs]
        _write_json(manifest_path, manifest)
        raise RuntimeError(f"Expected exactly one Nerfstudio config, found {len(configs)}")
    config_path = configs[0]
    checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))
    if not checkpoints:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "checkpoint_discovery"
        _write_json(manifest_path, manifest)
        raise RuntimeError("Nerfstudio training succeeded but no checkpoint was found")
    checkpoint_path = checkpoints[-1]
    manifest["training"]["config_path"] = str(config_path.relative_to(run_dir))
    manifest["training"]["checkpoint_path"] = str(checkpoint_path.relative_to(run_dir))

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
        _write_json(manifest_path, manifest)
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
        manifest["ply_candidates"] = [str(path.relative_to(run_dir)) for path in ply_files]
        _write_json(manifest_path, manifest)
        raise RuntimeError(f"Expected exactly one exported PLY, found {len(ply_files)}")

    ply_path = ply_files[0]
    manifest["status"] = "success"
    manifest["output"] = {
        "ply_path": str(ply_path.relative_to(run_dir)),
        "size_bytes": ply_path.stat().st_size,
        "sha256": sha256_file(ply_path),
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def write_source_manifest(
    video_path: str | Path,
    source: VideoSource,
    probe: Mapping,
    output_path: str | Path,
    *,
    downloaded_at: str | None = None,
) -> dict:
    path = Path(video_path)
    manifest = {
        "schema_version": 1,
        "source": asdict(source),
        "video": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "downloaded_at": downloaded_at or datetime.now(timezone.utc).isoformat(),
            "probe": dict(probe),
        },
    }
    Path(output_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
