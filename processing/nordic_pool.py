from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from processing.nordic_seeds import EXPECTED_REGION_IDS, load_nordic_seeds
from processing.wikimedia_discovery import discover_from_seed_config, normalize_discovery

DEFAULT_SEEDS = Path("sources/discovery/nordic-seeds.json")
DEFAULT_POOL = Path("sources/discovery/nordic-wikimedia.json")
DEFAULT_COVERAGE = Path("sources/discovery/nordic-coverage.json")
DEFAULT_QUEUE = Path("sources/discovery/nordic-preflight-queue.json")
FORBIDDEN_HEURISTIC_FIELDS = {"rank", "score", "expected_success"}
MINIMUM_DURATION_SECONDS = 120.0
_VOLATILE_KEYS = {"generated_at", "last_successful_discovery_at"}
DownloadChecker = Callable[[str], bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_candidate_id(canonical_title: str) -> str:
    stem = canonical_title.removeprefix("File:").rsplit(".", 1)[0]
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_stem.casefold()).strip("-")[:64] or "video"
    digest = hashlib.sha256(canonical_title.encode("utf-8")).hexdigest()[:10]
    return f"wikimedia-{slug}-{digest}"


def _seed_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _downloadable(url: str) -> bool:
    headers = {"User-Agent": "KAFKA2306-AutoPhotogrammetry/1.0"}
    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            if getattr(response, "status", 200) < 400:
                return True
    except Exception:
        pass
    try:
        request = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return getattr(response, "status", 200) in {200, 206}
    except Exception:
        return False


def stage_a_gate(
    candidate: Mapping[str, Any],
    *,
    minimum_duration_seconds: float = MINIMUM_DURATION_SECONDS,
) -> dict[str, Any]:
    license_record = candidate.get("license")
    resolution = candidate.get("resolution")
    checks = {
        "source_page_present": bool(candidate.get("source_page")),
        "media_url_present": bool(candidate.get("media_url")),
        "downloadability_checked": candidate.get("downloadable") is True,
        "video_media_confirmed": candidate.get("confirmed_video") is True,
        "source_sha1_present": isinstance(candidate.get("source_sha1"), str)
        and len(str(candidate.get("source_sha1"))) == 40,
        "source_size_present": isinstance(candidate.get("source_size_bytes"), int)
        and int(candidate["source_size_bytes"]) > 0,
        "author_present": bool(candidate.get("author")),
        "license_present": isinstance(license_record, Mapping)
        and bool(license_record.get("name"))
        and bool(license_record.get("url")),
        "license_verified": isinstance(license_record, Mapping)
        and license_record.get("status") == "verified",
        "duration_present": isinstance(candidate.get("duration_seconds"), (int, float)),
        "resolution_present": isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution),
        "duration_threshold_pass": isinstance(candidate.get("duration_seconds"), (int, float))
        and float(candidate["duration_seconds"]) >= minimum_duration_seconds,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        **checks,
        "minimum_duration_seconds": minimum_duration_seconds,
        "eligible_for_preflight": not failed,
        "failed_requirements": failed,
    }


def _candidate_record(
    raw: Mapping[str, Any],
    *,
    check_download: DownloadChecker,
    minimum_duration_seconds: float,
) -> dict[str, Any]:
    canonical_title = str(raw["canonical_title"])
    media_url = raw.get("media_url")
    width = raw.get("width")
    height = raw.get("height")
    resolution = (
        [width, height]
        if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0
        else None
    )
    duration = raw.get("duration_seconds")
    license_record = raw.get("license")
    should_check_download = bool(
        media_url
        and raw.get("author")
        and isinstance(license_record, Mapping)
        and license_record.get("name")
        and license_record.get("url")
        and isinstance(duration, (int, float))
        and float(duration) >= minimum_duration_seconds
        and resolution
    )
    candidate: dict[str, Any] = {
        "id": stable_candidate_id(canonical_title),
        "canonical_title": canonical_title,
        "title": canonical_title.removeprefix("File:").rsplit(".", 1)[0],
        "provider": "Wikimedia Commons",
        "regions": sorted(str(value) for value in raw.get("regions", [])),
        "source_page": raw.get("source_page"),
        "media_url": media_url,
        "author": raw.get("author"),
        "license": dict(license_record) if isinstance(license_record, Mapping) else {},
        "duration_seconds": duration,
        "duration_authority": raw.get("duration_authority"),
        "resolution": resolution,
        "source_sha1": raw.get("source_sha1"),
        "source_size_bytes": raw.get("source_size_bytes"),
        "mime": raw.get("mime"),
        "media_type": raw.get("media_type"),
        "confirmed_video": raw.get("confirmed_video") is True,
        "downloadable": bool(should_check_download and check_download(str(media_url))),
        "commons_categories": sorted(str(value) for value in raw.get("commons_categories", [])),
        "discovered_categories": sorted(
            str(value) for value in raw.get("discovered_categories", [])
        ),
        "discovery_paths": sorted([list(path) for path in raw.get("discovery_paths", [])]),
        "evaluation_stage": "metadata",
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "metadata_authority": raw.get("metadata_authority"),
    }
    candidate["stage_a"] = stage_a_gate(
        candidate,
        minimum_duration_seconds=minimum_duration_seconds,
    )
    return candidate


def build_pool(
    seed_path: str | Path = DEFAULT_SEEDS,
    *,
    request_json: Any = None,
    request_file_json: Any = None,
    check_download: DownloadChecker = _downloadable,
    minimum_duration_seconds: float = MINIMUM_DURATION_SECONDS,
) -> dict[str, Any]:
    seed_config = load_nordic_seeds(seed_path)
    discovery_kwargs: dict[str, Any] = {}
    if request_json is not None:
        discovery_kwargs["request_json"] = request_json
    discovery = discover_from_seed_config(seed_config, **discovery_kwargs)

    normalize_kwargs: dict[str, Any] = {}
    if request_json is not None:
        normalize_kwargs["request_json"] = request_json
    if request_file_json is not None:
        normalize_kwargs["request_file_json"] = request_file_json
    normalized = normalize_discovery(discovery, **normalize_kwargs)

    candidates = [
        _candidate_record(
            candidate,
            check_download=check_download,
            minimum_duration_seconds=minimum_duration_seconds,
        )
        for candidate in normalized["candidates"]
    ]
    raw_counts = {region_id: 0 for region_id in EXPECTED_REGION_IDS}
    title_regions: dict[str, list[str]] = {}
    for candidate in discovery["candidates"]:
        title = str(candidate.get("canonical_title") or "")
        regions = [str(region) for region in candidate.get("regions", [])]
        title_regions[title] = regions
        for region_id in regions:
            if region_id in raw_counts:
                raw_counts[region_id] += 1

    metadata_failures = []
    for failure in normalized.get("metadata_failures", []):
        row = dict(failure)
        row["regions"] = title_regions.get(str(row.get("canonical_title") or ""), [])
        metadata_failures.append(row)

    pool = {
        "schema_version": 1,
        "snapshot_state": "api",
        "authority": {
            "category_discovery": "Wikimedia Commons categorymembers",
            "video_metadata": "Wikimedia Commons videoinfo/categories",
            "duration_fallback": "MediaWiki REST API file information",
        },
        "seed_config_sha256": _seed_hash(seed_path),
        "regions": list(EXPECTED_REGION_IDS),
        "minimum_duration_seconds": minimum_duration_seconds,
        "raw_discovered_file_count_by_region": raw_counts,
        "candidates": sorted(
            candidates,
            key=lambda item: (item["regions"], item["canonical_title"]),
        ),
        "discovery_failures": list(discovery.get("failures", [])),
        "metadata_failures": metadata_failures,
    }
    validate_pool(pool)
    return pool


def validate_pool(pool: Mapping[str, Any]) -> None:
    if pool.get("schema_version") != 1:
        raise ValueError("Nordic pool schema_version must be 1")
    if tuple(pool.get("regions", [])) != EXPECTED_REGION_IDS:
        raise ValueError("Nordic pool must preserve exactly the eight configured regions")
    candidates = pool.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Nordic pool requires candidates")
    ids: set[str] = set()
    titles: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Nordic candidate must be an object")
        forbidden = FORBIDDEN_HEURISTIC_FIELDS.intersection(candidate)
        if forbidden:
            raise ValueError(f"Heuristic ranking fields are forbidden: {sorted(forbidden)}")
        candidate_id = candidate.get("id")
        canonical_title = candidate.get("canonical_title")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Nordic candidate id is required")
        if candidate_id in ids:
            raise ValueError(f"Duplicate Nordic candidate id: {candidate_id}")
        ids.add(candidate_id)
        if not isinstance(canonical_title, str) or not canonical_title:
            raise ValueError(f"{candidate_id}: canonical_title is required")
        if canonical_title in titles:
            raise ValueError(f"Duplicate Nordic canonical title: {canonical_title}")
        titles.add(canonical_title)
        if candidate.get("evaluation_stage") != "metadata":
            raise ValueError(f"{candidate_id}: discovery candidates must remain at metadata stage")
        if not isinstance(candidate.get("stage_a"), Mapping):
            raise ValueError(f"{candidate_id}: stage_a is required")


def _region_failure(pool: Mapping[str, Any], region_id: str) -> list[str]:
    errors = [
        str(item.get("error"))
        for item in pool.get("discovery_failures", [])
        if isinstance(item, Mapping) and item.get("region_id") == region_id
    ]
    errors.extend(
        str(item.get("error"))
        for item in pool.get("metadata_failures", [])
        if isinstance(item, Mapping) and region_id in item.get("regions", [])
    )
    return errors


def build_coverage(
    pool: Mapping[str, Any],
    seed_config: Mapping[str, Any],
    *,
    successful_at: str | None = None,
) -> dict[str, Any]:
    validate_pool(pool)
    candidates = pool["candidates"]
    raw_counts = pool.get("raw_discovered_file_count_by_region", {})
    rows: list[dict[str, Any]] = []
    for region in seed_config["regions"]:
        region_id = str(region["id"])
        region_candidates = [
            candidate for candidate in candidates if region_id in candidate.get("regions", [])
        ]
        failures = _region_failure(pool, region_id)
        configured_seed_count = len(region["seeds"])
        metadata_complete_count = sum(
            1
            for candidate in region_candidates
            if candidate.get("source_page")
            and candidate.get("media_url")
            and candidate.get("author")
            and candidate.get("duration_seconds") is not None
            and candidate.get("resolution")
            and isinstance(candidate.get("license"), Mapping)
            and candidate["license"].get("name")
            and candidate["license"].get("url")
        )
        verified_license_count = sum(
            1
            for candidate in region_candidates
            if isinstance(candidate.get("license"), Mapping)
            and candidate["license"].get("status") == "verified"
        )
        pass_count = sum(
            1
            for candidate in region_candidates
            if candidate["stage_a"].get("eligible_for_preflight") is True
        )
        if failures:
            state = "failed"
            reason = "; ".join(failures)
        elif region["status"] == "missing":
            state = "missing_seed"
            reason = region.get("missing_reason")
        elif int(raw_counts.get(region_id, 0)) == 0:
            state = "zero_candidates"
            reason = "Configured seed traversal completed with zero discovered files."
        else:
            state = "ok"
            reason = None
        rows.append(
            {
                "region_id": region_id,
                "display_name": region["display_name"],
                "seed_status": region["status"],
                "configured_seed_count": configured_seed_count,
                "reachable_seed_count": 0 if failures else configured_seed_count,
                "discovered_file_count": int(raw_counts.get(region_id, 0)),
                "confirmed_video_count": len(region_candidates),
                "metadata_complete_count": metadata_complete_count,
                "verified_license_count": verified_license_count,
                "stage_a_pass_count": pass_count,
                "stage_a_fail_count": len(region_candidates) - pass_count,
                "discovery_state": state,
                "reason": reason,
                "last_successful_discovery_at": (
                    successful_at if state in {"ok", "zero_candidates"} else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "regions": rows,
        "totals": {
            "region_count": len(rows),
            "candidate_count": len(candidates),
            "stage_a_pass_count": sum(row["stage_a_pass_count"] for row in rows),
        },
    }


def _without_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_volatile(item) for key, item in value.items() if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def persist_snapshot(path: str | Path, payload: Mapping[str, Any]) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = dict(payload)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _without_volatile(existing) == _without_volatile(incoming):
            return False
    incoming["generated_at"] = incoming.get("generated_at") or utc_now()
    destination.write_text(
        json.dumps(incoming, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def refresh_snapshots(
    *,
    seed_path: str | Path = DEFAULT_SEEDS,
    pool_path: str | Path = DEFAULT_POOL,
    coverage_path: str | Path = DEFAULT_COVERAGE,
    allow_partial: bool = False,
) -> dict[str, Any]:
    pool = build_pool(seed_path)
    if not allow_partial and (pool["discovery_failures"] or pool["metadata_failures"]):
        raise RuntimeError("Nordic discovery was partial; previous good snapshots were preserved")
    successful_at = utc_now()
    pool["generated_at"] = successful_at
    coverage = build_coverage(
        pool,
        load_nordic_seeds(seed_path),
        successful_at=successful_at,
    )
    coverage["generated_at"] = successful_at
    return {
        "candidate_count": len(pool["candidates"]),
        "stage_a_pass_count": coverage["totals"]["stage_a_pass_count"],
        "pool_changed": persist_snapshot(pool_path, pool),
        "coverage_changed": persist_snapshot(coverage_path, coverage),
    }


def load_pool(path: str | Path = DEFAULT_POOL) -> dict[str, Any]:
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pool(pool)
    return pool


def find_candidate(pool: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in pool["candidates"]:
        if candidate.get("id") == candidate_id:
            return dict(candidate)
    raise KeyError(f"Unknown Nordic candidate: {candidate_id}")


def validate_queue(queue: Mapping[str, Any], pool: Mapping[str, Any] | None = None) -> None:
    if queue.get("schema_version") != 1:
        raise ValueError("Nordic preflight queue schema_version must be 1")
    candidates = queue.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Nordic preflight queue requires candidates")
    ids: set[str] = set()
    pool_by_id = (
        {candidate["id"]: candidate for candidate in pool.get("candidates", [])}
        if pool is not None
        else {}
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Nordic preflight queue candidate must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Nordic preflight queue candidate id is required")
        if candidate_id in ids:
            raise ValueError(f"Duplicate Nordic preflight candidate id: {candidate_id}")
        ids.add(candidate_id)
        stage_a = candidate.get("stage_a")
        if not isinstance(stage_a, Mapping) or stage_a.get("eligible_for_preflight") is not True:
            raise ValueError(f"{candidate_id}: queue contains a Stage A failure")
        if FORBIDDEN_HEURISTIC_FIELDS.intersection(candidate):
            raise ValueError(f"{candidate_id}: heuristic ranking fields are forbidden")
        if pool is not None:
            current = pool_by_id.get(candidate_id)
            if current is None:
                raise ValueError(f"{candidate_id}: queued candidate is absent from current pool")
            if current.get("source_sha1") != candidate.get("source_sha1"):
                raise ValueError(f"{candidate_id}: queued Commons SHA-1 is stale")
            if current.get("media_url") != candidate.get("media_url"):
                raise ValueError(f"{candidate_id}: queued media URL is stale")


def queue_candidate(
    candidate_id: str,
    *,
    pool_path: str | Path = DEFAULT_POOL,
    queue_path: str | Path = DEFAULT_QUEUE,
) -> dict[str, Any]:
    pool = load_pool(pool_path)
    candidate = find_candidate(pool, candidate_id)
    if candidate["stage_a"].get("eligible_for_preflight") is not True:
        raise ValueError(
            f"{candidate_id}: Stage A incomplete: {candidate['stage_a']['failed_requirements']}"
        )
    destination = Path(queue_path)
    if destination.is_file():
        queue = json.loads(destination.read_text(encoding="utf-8"))
    else:
        queue = {
            "schema_version": 1,
            "source_pool": str(pool_path),
            "candidates": [],
        }
    validate_queue(queue)
    raw_candidates = queue.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Nordic preflight queue requires candidates")
    queue_candidates: list[dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise ValueError("Nordic preflight queue candidate must be an object")
        queue_candidates.append(item)
    existing = next(
        (item for item in queue_candidates if item.get("id") == candidate_id),
        None,
    )
    if existing is not None:
        if existing.get("source_sha1") != candidate.get("source_sha1"):
            raise ValueError(f"{candidate_id}: queued Commons SHA-1 changed; reselect explicitly")
        if existing.get("media_url") != candidate.get("media_url"):
            raise ValueError(f"{candidate_id}: queued media URL changed; reselect explicitly")
        return dict(existing)
    queue_candidates.append(candidate)
    queue_candidates.sort(key=lambda item: item["id"])
    queue["candidates"] = queue_candidates
    validate_queue(queue, pool)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maintain the Nordic Commons pool, coverage, and explicit Stage A queue."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--allow-partial", action="store_true")
    subparsers.add_parser("validate")
    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("candidate_id")
    args = parser.parse_args()

    if args.command == "refresh":
        result = refresh_snapshots(allow_partial=args.allow_partial)
    elif args.command == "validate":
        pool = load_pool(DEFAULT_POOL)
        coverage = json.loads(DEFAULT_COVERAGE.read_text(encoding="utf-8"))
        queue = json.loads(DEFAULT_QUEUE.read_text(encoding="utf-8"))
        if tuple(item["region_id"] for item in coverage["regions"]) != EXPECTED_REGION_IDS:
            raise ValueError("Nordic coverage is missing one or more configured regions")
        validate_queue(queue, pool)
        result = {
            "candidate_count": len(pool["candidates"]),
            "region_count": len(coverage["regions"]),
            "queued_count": len(queue["candidates"]),
        }
    else:
        result = queue_candidate(args.candidate_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
