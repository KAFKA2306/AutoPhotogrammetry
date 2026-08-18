from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources" / "videos.json"
EVALUATION_STAGES = ("metadata", "preflight", "colmap", "splat")
DEPRECATED_HEURISTIC_FIELDS = {"rank", "score", "expected_success"}


def load_video_registry(path: str | Path = REGISTRY_PATH) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = registry.get("videos")
    policy = registry.get("evaluation_policy")
    if registry.get("schema_version") != 2 or not isinstance(videos, list) or not videos:
        raise ValueError("Invalid video source registry")
    if not isinstance(policy, dict) or set(policy.get("stages", {})) != set(EVALUATION_STAGES):
        raise ValueError("Registry must define metadata/preflight/colmap/splat evaluation stages")

    ids = [video.get("id") for video in videos]
    if any(not source_id for source_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Video source ids must be non-empty and unique")
    if registry.get("default") not in set(ids):
        raise ValueError("Registry default must reference an existing video id")

    for video in videos:
        deprecated = DEPRECATED_HEURISTIC_FIELDS.intersection(video)
        if deprecated:
            raise ValueError(f"Heuristic ranking fields are forbidden: {sorted(deprecated)}")
        if video.get("evaluation_stage") not in EVALUATION_STAGES:
            raise ValueError(f"Invalid evaluation stage for {video['id']}")
        measurements = video.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError(f"Missing measurements for {video['id']}")
        if set(measurements) != {"preflight", "colmap", "splat"}:
            raise ValueError(f"Measurements must contain preflight/colmap/splat for {video['id']}")

    return registry


def get_video_source(source_id: str | None = None, path: str | Path = REGISTRY_PATH) -> dict:
    registry = load_video_registry(path)
    wanted = source_id or registry["default"]
    for video in registry["videos"]:
        if video["id"] == wanted:
            return video
    raise KeyError(f"Unknown video source: {wanted}")


def video_sources(path: str | Path = REGISTRY_PATH) -> list[dict]:
    return list(load_video_registry(path)["videos"])
