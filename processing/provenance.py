from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VideoSource:
    title: str
    source_page: str
    media_url: str
    author: str
    license: str
    license_url: str
    target: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_revision(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Return the exact AutoPhotogrammetry revision executing this run."""
    revision = os.environ.get("AUTOPHOTOGRAMMETRY_SOURCE_REVISION", "").strip()
    if not revision:
        try:
            completed = runner(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "AUTOPHOTOGRAMMETRY_SOURCE_REVISION is required when the runtime does not contain .git"
            ) from exc
        revision = completed.stdout.strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError(
            "AutoPhotogrammetry source revision must be a full lowercase 40-character Git commit SHA"
        )
    return revision


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Mapping | list) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def image_records(data_dir: str | Path) -> list[dict]:
    root = Path(data_dir)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


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
            "downloaded_at": downloaded_at or utc_now(),
            "probe": dict(probe),
        },
    }
    write_json(output_path, manifest)
    return manifest
