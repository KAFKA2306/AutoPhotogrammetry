from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from processing.batch import ensure_source
from processing.nordic_pool import (
    DEFAULT_POOL,
    DEFAULT_QUEUE,
    find_candidate,
    load_pool,
    validate_queue,
)
from processing.provenance import sha256_file
from processing.video_preflight import apply_preflight_to_registry, run_video_preflight
from processing.video_sources import load_video_registry

DEFAULT_CANONICAL_REGISTRY = Path("sources/videos.json")
DEFAULT_INPUT_ROOT = Path("input/nordic")
DEFAULT_OUTPUT_ROOT = Path("output/nordic-preflight")


def load_queue(
    queue_path: str | Path = DEFAULT_QUEUE,
    *,
    pool_path: str | Path = DEFAULT_POOL,
) -> dict[str, Any]:
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    validate_queue(queue, load_pool(pool_path))
    return queue


def _evaluation_source(candidate: Mapping[str, Any], sha256: str, size: int) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "evaluation_stage": "metadata",
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "status": "candidate",
        "title": candidate["title"],
        "provider": "Wikimedia Commons",
        "source_page": candidate["source_page"],
        "media_url": candidate["media_url"],
        "sha256": sha256,
        "author": candidate["author"],
        "license": candidate["license"],
        "duration_seconds": candidate["duration_seconds"],
        "resolution": candidate["resolution"],
        "target": candidate["title"],
        "metadata_evidence": {
            "authority": candidate["metadata_authority"],
            "resolved_via": "nordic-stage-a-queue",
            "source_sha1": candidate["source_sha1"],
            "source_size_bytes": candidate["source_size_bytes"],
            "mime": candidate["mime"],
            "download_url_available": candidate["downloadable"],
            "license_verified": candidate["license"].get("status") == "verified",
            "downloaded_size_bytes": size,
            "sha256_verified_from_downloaded_bytes": True,
            "commons_sha1_verified_from_downloaded_bytes": True,
            "nordic_candidate_id": candidate["id"],
            "commons_canonical_title": candidate["canonical_title"],
            "duration_authority": candidate["duration_authority"],
        },
    }


def write_evaluation_registry(
    candidate: Mapping[str, Any],
    *,
    sha256: str,
    downloaded_size: int,
    destination: str | Path,
    canonical_registry_path: str | Path = DEFAULT_CANONICAL_REGISTRY,
) -> Path:
    canonical = load_video_registry(canonical_registry_path)
    registry = {
        "schema_version": 2,
        "default": candidate["id"],
        "evaluation_policy": canonical["evaluation_policy"],
        "videos": [_evaluation_source(candidate, sha256, downloaded_size)],
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_video_registry(path)
    return path


def preflight_candidate(
    candidate_id: str,
    *,
    queue_path: str | Path = DEFAULT_QUEUE,
    pool_path: str | Path = DEFAULT_POOL,
    canonical_registry_path: str | Path = DEFAULT_CANONICAL_REGISTRY,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    queue = load_queue(queue_path, pool_path=pool_path)
    candidate = find_candidate(queue, candidate_id)
    if candidate["stage_a"].get("eligible_for_preflight") is not True:
        raise ValueError(f"{candidate_id}: candidate no longer passes Stage A")

    suffix = Path(str(candidate["canonical_title"])).suffix or ".webm"
    source_path = Path(input_root) / candidate_id / f"source{suffix}"
    source_path = ensure_source(
        source_path,
        url=str(candidate["media_url"]),
        expected_sha1=str(candidate["source_sha1"]),
        expected_size=int(candidate["source_size_bytes"]),
    )
    source_sha256 = sha256_file(source_path)
    output_dir = Path(output_root) / candidate_id
    registry_path = write_evaluation_registry(
        candidate,
        sha256=source_sha256,
        downloaded_size=source_path.stat().st_size,
        destination=output_dir / "evaluation-registry.json",
        canonical_registry_path=canonical_registry_path,
    )
    preflight_path = output_dir / "preflight.json"
    result = run_video_preflight(source_path, preflight_path)
    apply_preflight_to_registry(registry_path, candidate_id, result)
    return {
        "candidate_id": candidate_id,
        "source_sha256": source_sha256,
        "downloaded_size_bytes": source_path.stat().st_size,
        "preflight_manifest": str(preflight_path),
        "evaluation_registry": str(registry_path),
        "measurements": result["metrics"],
        "automatic_colmap_promotion": False,
    }


def preflight_batch(
    candidate_ids: Sequence[str],
    *,
    queue_path: str | Path = DEFAULT_QUEUE,
    pool_path: str | Path = DEFAULT_POOL,
    canonical_registry_path: str | Path = DEFAULT_CANONICAL_REGISTRY,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for candidate_id in candidate_ids:
        try:
            measured.append(
                preflight_candidate(
                    candidate_id,
                    queue_path=queue_path,
                    pool_path=pool_path,
                    canonical_registry_path=canonical_registry_path,
                    input_root=input_root,
                    output_root=output_root,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "candidate_id": candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    summary = {
        "schema_version": 1,
        "measured": measured,
        "failed": failed,
        "measured_count": len(measured),
        "failed_count": len(failed),
        "automatic_colmap_promotion": False,
    }
    summary_path = Path(output_root) / "batch.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing Stage-B preflight for explicitly queued Nordic candidates."
    )
    parser.add_argument("candidate_ids", nargs="+")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = preflight_batch(
        args.candidate_ids,
        queue_path=args.queue,
        pool_path=args.pool,
        input_root=args.input_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
