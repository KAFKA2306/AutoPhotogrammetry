from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class VideoSource:
    title: str
    source_page: str
    media_url: str
    author: str
    license: str
    license_url: str
    target: str


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
        "-show_entries", "format=duration,size:stream=codec_name,width,height",
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
) -> list[str]:
    return [executable, "splatfacto", "--data", str(Path(data_dir))]


def gaussian_splat_export_command(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-export",
) -> list[str]:
    return [
        executable,
        "gaussian-splat",
        "--load-config", str(Path(config_path)),
        "--output-dir", str(Path(output_dir)),
    ]


def write_source_manifest(
    video_path: str | Path,
    source: VideoSource,
    probe: Mapping,
    output_path: str | Path,
) -> dict:
    path = Path(video_path)
    manifest = {
        "schema_version": 1,
        "source": asdict(source),
        "video": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "probe": dict(probe),
        },
    }
    Path(output_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
