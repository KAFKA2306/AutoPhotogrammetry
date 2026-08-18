from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources" / "videos.json"


def load_video_registry(path: str | Path = REGISTRY_PATH) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = registry.get("videos")
    if registry.get("schema_version") != 1 or not isinstance(videos, list) or not videos:
        raise ValueError("Invalid video source registry")

    ids = [video.get("id") for video in videos]
    if any(not source_id for source_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Video source ids must be non-empty and unique")
    if registry.get("default") not in set(ids):
        raise ValueError("Registry default must reference an existing video id")
    return registry


def get_video_source(source_id: str | None = None, path: str | Path = REGISTRY_PATH) -> dict:
    registry = load_video_registry(path)
    wanted = source_id or registry["default"]
    for video in registry["videos"]:
        if video["id"] == wanted:
            return video
    raise KeyError(f"Unknown video source: {wanted}")


def ranked_video_sources(path: str | Path = REGISTRY_PATH) -> list[dict]:
    registry = load_video_registry(path)
    return sorted(registry["videos"], key=lambda video: (video["rank"], video["id"]))
