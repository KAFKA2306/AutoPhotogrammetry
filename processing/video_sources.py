from __future__ import annotations

from pathlib import Path
from typing import Any

from processing.schemas import VideoRegistryModel

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources" / "videos.json"
EVALUATION_STAGES = ("metadata", "preflight", "colmap", "splat")
DEPRECATED_HEURISTIC_FIELDS = {"rank", "score", "expected_success"}


def load_video_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = VideoRegistryModel.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    ).model_dump(mode="python")
    videos = registry["videos"]
    policy = registry["evaluation_policy"]
    if registry["schema_version"] != 2:
        raise ValueError("Invalid video source registry schema version")
    if set(policy["stages"]) != set(EVALUATION_STAGES):
        raise ValueError("Registry must define metadata/preflight/colmap/splat evaluation stages")

    ids = [video["id"] for video in videos]
    if len(ids) != len(set(ids)):
        raise ValueError("Video source ids must be unique")
    if registry["default"] not in set(ids):
        raise ValueError("Registry default must reference an existing video id")

    for video in videos:
        deprecated = DEPRECATED_HEURISTIC_FIELDS.intersection(video)
        if deprecated:
            raise ValueError(f"Heuristic ranking fields are forbidden: {sorted(deprecated)}")
        if video["evaluation_stage"] not in EVALUATION_STAGES:
            raise ValueError(f"Invalid evaluation stage for {video['id']}")

    return registry


def get_video_source(
    source_id: str | None = None,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, Any]:
    registry = load_video_registry(path)
    wanted = source_id or registry["default"]
    for video in registry["videos"]:
        if video["id"] == wanted:
            return video
    raise KeyError(f"Unknown video source: {wanted}")


def video_sources(path: str | Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    return list(load_video_registry(path)["videos"])
