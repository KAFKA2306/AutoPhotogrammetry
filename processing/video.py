from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(map(str, command)),
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> dict:
    completed = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=codec_name,width,height",
            "-of",
            "json",
            str(Path(path)),
        ]
    )
    return json.loads(completed.stdout)


def scene_cut_times(
    path: str | Path,
    *,
    threshold: float = 0.4,
    ffmpeg: str = "ffmpeg",
) -> list[float]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    completed = run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(Path(path)),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)]


def extract_frames_command(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 3.0,
    width: int | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width is not None and width <= 0:
        raise ValueError("width must be positive")
    filters = [f"fps={fps:g}"]
    if width is not None:
        filters.append(f"scale={width}:-2")
    output = Path(output_dir)
    return [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(Path(video_path)),
        "-vf",
        ",".join(filters),
        "-q:v",
        "2",
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
