from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from processing.wikimedia_discovery import normalize_discovery, traverse_category

ROOT_CATEGORY = "Category:360-degree videos"
DEFAULT_POOL = Path("sources/discovery/wikimedia-360.json")
FORBIDDEN_QUALITY_FIELDS = {"rank", "score", "expected_success", "quality_score"}


def projection_state(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return only projection facts supported by Commons category evidence.

    A 2:1 raster or a '360' title is not enough to assert equirectangular projection.
    EAC has an explicit Commons category, so it can be identified fail-closed. All
    other 360 videos remain unknown until the materialized source is reviewed.
    """
    categories = {
        str(value).casefold()
        for value in [
            *candidate.get("commons_categories", []),
            *candidate.get("discovered_categories", []),
        ]
    }
    if "category:eac video" in categories:
        return {
            "projection_type": "eac",
            "projection_authority": "Wikimedia Commons Category:EAC Video",
            "projection_review_required": False,
            "equirectangular_processing_ready": False,
        }
    return {
        "projection_type": "unknown",
        "projection_authority": None,
        "projection_review_required": True,
        "equirectangular_processing_ready": False,
    }


def stage_a_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    license_record = candidate.get("license")
    resolution = candidate.get("resolution")
    checks = {
        "source_page_present": bool(candidate.get("source_page")),
        "media_url_present": bool(candidate.get("media_url")),
        "video_media_confirmed": candidate.get("confirmed_video") is True,
        "source_sha1_present": isinstance(candidate.get("source_sha1"), str)
        and len(str(candidate.get("source_sha1"))) == 40,
        "source_size_present": isinstance(candidate.get("source_size_bytes"), int)
        and int(candidate["source_size_bytes"]) > 0,
        "author_present": bool(candidate.get("author")),
        "license_verified": isinstance(license_record, Mapping)
        and license_record.get("status") == "verified"
        and bool(license_record.get("name"))
        and bool(license_record.get("url")),
        "duration_present": isinstance(candidate.get("duration_seconds"), (int, float)),
        "resolution_present": isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        **checks,
        "eligible_for_projection_review": not failed,
        "failed_requirements": failed,
    }


def candidate_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    width = raw.get("width")
    height = raw.get("height")
    resolution = (
        [width, height]
        if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0
        else None
    )
    license_record = raw.get("license")
    candidate: dict[str, Any] = {
        "canonical_title": raw.get("canonical_title"),
        "provider": "Wikimedia Commons",
        "source_page": raw.get("source_page"),
        "media_url": raw.get("media_url"),
        "author": raw.get("author"),
        "license": dict(license_record) if isinstance(license_record, Mapping) else {},
        "duration_seconds": raw.get("duration_seconds"),
        "duration_authority": raw.get("duration_authority"),
        "resolution": resolution,
        "source_sha1": raw.get("source_sha1"),
        "source_size_bytes": raw.get("source_size_bytes"),
        "mime": raw.get("mime"),
        "media_type": raw.get("media_type"),
        "confirmed_video": raw.get("confirmed_video") is True,
        "commons_categories": sorted(str(value) for value in raw.get("commons_categories", [])),
        "discovered_categories": sorted(
            str(value) for value in raw.get("discovered_categories", [])
        ),
        "discovery_paths": sorted([list(path) for path in raw.get("discovery_paths", [])]),
        "metadata_authority": raw.get("metadata_authority"),
        "evaluation_stage": "metadata",
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "camera_motion": "unknown",
        "static_scene": "unknown",
    }
    candidate.update(projection_state(candidate))
    candidate["stage_a"] = stage_a_gate(candidate)
    return candidate


def build_pool(
    *,
    request_json: Any = None,
    request_file_json: Any = None,
) -> dict[str, Any]:
    traverse_kwargs: dict[str, Any] = {}
    if request_json is not None:
        traverse_kwargs["request_json"] = request_json
    discovered = traverse_category(ROOT_CATEGORY, **traverse_kwargs)
    discovery = {
        "schema_version": 1,
        "authority": "Wikimedia Commons categorymembers",
        "candidates": [
            {
                **candidate,
                "regions": [],
            }
            for candidate in discovered
        ],
        "failures": [],
    }

    normalize_kwargs: dict[str, Any] = {}
    if request_json is not None:
        normalize_kwargs["request_json"] = request_json
    if request_file_json is not None:
        normalize_kwargs["request_file_json"] = request_file_json
    normalized = normalize_discovery(discovery, **normalize_kwargs)

    pool = {
        "schema_version": 1,
        "snapshot_state": "api",
        "authority": {
            "category_discovery": "Wikimedia Commons categorymembers",
            "video_metadata": "Wikimedia Commons videoinfo/categories",
            "duration_fallback": "MediaWiki REST API file information",
        },
        "root_category": ROOT_CATEGORY,
        "discovered_file_count": len(discovered),
        "candidate_count": len(normalized["candidates"]),
        "candidates": [candidate_record(candidate) for candidate in normalized["candidates"]],
        "discovery_failures": list(normalized.get("discovery_failures", [])),
        "metadata_failures": list(normalized.get("metadata_failures", [])),
        "policy": {
            "quality_is_not_inferred_from_metadata": True,
            "aspect_ratio_does_not_prove_equirectangular": True,
            "title_does_not_prove_equirectangular": True,
            "production_promotion_requires_stage_c_and_d_evidence": True,
        },
    }
    validate_pool(pool)
    return pool


def validate_pool(pool: Mapping[str, Any]) -> None:
    if pool.get("schema_version") != 1:
        raise ValueError("Wikimedia 360 pool schema_version must be 1")
    if pool.get("root_category") != ROOT_CATEGORY:
        raise ValueError(f"Wikimedia 360 pool root_category must be {ROOT_CATEGORY}")
    candidates = pool.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Wikimedia 360 pool requires candidates")

    titles: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Wikimedia 360 candidate must be an object")
        forbidden = FORBIDDEN_QUALITY_FIELDS.intersection(candidate)
        if forbidden:
            raise ValueError(f"Metadata quality ranking fields are forbidden: {sorted(forbidden)}")
        title = candidate.get("canonical_title")
        if not isinstance(title, str) or not title:
            raise ValueError("Wikimedia 360 candidate canonical_title is required")
        if title in titles:
            raise ValueError(f"Duplicate Wikimedia 360 canonical title: {title}")
        titles.add(title)
        if candidate.get("evaluation_stage") != "metadata":
            raise ValueError(f"{title}: discovery must remain at metadata stage")
        if not isinstance(candidate.get("stage_a"), Mapping):
            raise ValueError(f"{title}: stage_a is required")
        projection_type = candidate.get("projection_type")
        if projection_type not in {"unknown", "eac"}:
            raise ValueError(f"{title}: unsupported discovery projection_type={projection_type}")
        if candidate.get("equirectangular_processing_ready") is not False:
            raise ValueError(
                f"{title}: discovery metadata alone must not mark equirectangular processing ready"
            )


def refresh(output_path: str | Path = DEFAULT_POOL) -> dict[str, Any]:
    pool = build_pool()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output": path.as_posix(),
        "discovered_file_count": pool["discovered_file_count"],
        "candidate_count": pool["candidate_count"],
        "stage_a_projection_review_count": sum(
            1
            for candidate in pool["candidates"]
            if candidate["stage_a"].get("eligible_for_projection_review") is True
        ),
        "eac_count": sum(
            1 for candidate in pool["candidates"] if candidate["projection_type"] == "eac"
        ),
        "metadata_failure_count": len(pool["metadata_failures"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Wikimedia Commons 360-degree videos without turning metadata into quality claims."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--output", default=str(DEFAULT_POOL))
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", default=str(DEFAULT_POOL))
    args = parser.parse_args()

    if args.command == "refresh":
        print(json.dumps(refresh(args.output), ensure_ascii=False, indent=2))
        return

    pool = json.loads(Path(args.input).read_text(encoding="utf-8"))
    validate_pool(pool)
    print(json.dumps({"input": args.input, "candidate_count": len(pool["candidates"])}))


if __name__ == "__main__":
    main()
