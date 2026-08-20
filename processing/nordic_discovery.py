
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from processing.batch import ensure_source
from processing.provenance import sha256_file
from processing.video_preflight import apply_preflight_to_registry, run_video_preflight

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_REST = "https://commons.wikimedia.org/w/rest.php/v1/file/"
DEFAULT_SEEDS = Path("sources/discovery/nordic-seeds.json")
DEFAULT_POOL = Path("sources/discovery/nordic-wikimedia.json")
DEFAULT_COVERAGE = Path("sources/discovery/nordic-coverage.json")
DEFAULT_REGISTRY = Path("sources/videos.json")
REQUIRED_REGIONS = (
    "norway",
    "sweden",
    "finland",
    "denmark",
    "iceland",
    "greenland",
    "faroe-islands",
    "aland",
)
FORBIDDEN_HEURISTIC_FIELDS = {"rank", "score", "expected_success"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain_text(value: object) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return text or None


def _stable_id(title: str) -> str:
    stem = title.removeprefix("File:").rsplit(".", 1)[0]
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_stem.lower()).strip("-")[:64] or "video"
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:10]
    return f"wikimedia-{slug}-{digest}"


def _canonical_category_url(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":()")


def _canonical_file_url(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":(),")


def _duration_from_commonmetadata(value: object) -> float | None:
    if isinstance(value, Mapping):
        for key in ("length", "duration", "Duration"):
            raw = value.get(key)
            if isinstance(raw, Mapping):
                raw = raw.get("value")
            try:
                if raw is not None:
                    result = float(raw)
                    if result >= 0:
                        return result
            except (TypeError, ValueError):
                continue
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("key") or "").lower()
            if name not in {"length", "duration"}:
                continue
            raw = item.get("value")
            try:
                result = float(raw)
                if result >= 0:
                    return result
            except (TypeError, ValueError):
                pass
    return None


def _license_review_state(categories: Sequence[str], name: str | None, url: str | None) -> str:
    lowered = [category.lower() for category in categories]
    if any("license review" in category and ("needed" in category or "pending" in category) for category in lowered):
        return "needs_review"
    if name and url:
        return "verified"
    return "unknown"


def _content_without_generated_at(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("generated_at", None)
    return normalized


def persist_snapshot(path: str | Path, payload: Mapping[str, Any]) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = dict(payload)
    previous: dict[str, Any] | None = None
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if _content_without_generated_at(previous) == _content_without_generated_at(incoming):
            return False
    incoming["generated_at"] = utc_now()
    destination.write_text(json.dumps(incoming, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def load_seed_config(path: str | Path = DEFAULT_SEEDS) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Nordic seed config schema_version must be 1")
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Nordic seed config must contain regions")
    region_ids = [str(region.get("id")) for region in regions if isinstance(region, Mapping)]
    if tuple(region_ids) != REQUIRED_REGIONS:
        raise ValueError(f"Nordic seed config must contain exactly {REQUIRED_REGIONS}")
    seen: set[str] = set()
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("region must be an object")
        seeds = region.get("seeds")
        if not isinstance(seeds, list):
            raise ValueError(f"{region['id']}: seeds must be a list")
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise ValueError(f"{region['id']}: seed must be an object")
            state = seed.get("state")
            if state not in {"active", "missing"}:
                raise ValueError(f"{region['id']}: seed state must be active or missing")
            category = seed.get("category")
            if state == "active":
                if not isinstance(category, str) or not category.startswith("Category:"):
                    raise ValueError(f"{region['id']}: active seed requires exact Category: title")
                if category in seen:
                    raise ValueError(f"duplicate active seed category: {category}")
                seen.add(category)
                if seed.get("url") != _canonical_category_url(category):
                    raise ValueError(f"{region['id']}: seed URL does not match category title")
            elif category is not None:
                raise ValueError(f"{region['id']}: missing seed category must be null")
    return payload


class MediaWikiClient:
    def __init__(self, *, timeout: float = 45.0, user_agent: str = "AutoPhotogrammetry/0.7") -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get_json(self, params: Mapping[str, object]) -> dict[str, Any]:
        response = self.session.get(WIKIMEDIA_API, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("error"):
            raise RuntimeError(f"MediaWiki API error: {payload.get('error') if isinstance(payload, dict) else payload}")
        return payload

    def category_members(self, category: str) -> list[dict[str, Any]]:
        members: list[dict[str, Any]] = []
        continuation: str | None = None
        while True:
            params: dict[str, object] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "file|subcat",
                "cmprop": "ids|title|type",
                "cmlimit": "max",
            }
            if continuation is not None:
                params["cmcontinue"] = continuation
            payload = self.get_json(params)
            page = payload.get("query", {}).get("categorymembers", [])
            if not isinstance(page, list):
                raise RuntimeError(f"invalid categorymembers response for {category}")
            members.extend(item for item in page if isinstance(item, dict))
            continuation = (payload.get("continue") or {}).get("cmcontinue")
            if not continuation:
                break
        return members

    def video_info(self, title: str) -> dict[str, Any]:
        payload = self.get_json(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "videoinfo|categories",
                "titles": title,
                "viprop": "url|size|sha1|mime|mediatype|dimensions|commonmetadata|extmetadata",
                "viextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|License",
                "cllimit": "max",
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages or not isinstance(pages[0], Mapping) or pages[0].get("missing"):
            raise RuntimeError(f"Wikimedia file not found: {title}")
        page = pages[0]
        records = page.get("videoinfo")
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"videoinfo unavailable: {title}")
        info = records[0]
        if not isinstance(info, Mapping):
            raise RuntimeError(f"invalid videoinfo response: {title}")
        categories = [
            str(item.get("title"))
            for item in (page.get("categories") or [])
            if isinstance(item, Mapping) and item.get("title")
        ]
        metadata = info.get("extmetadata") or {}
        author = _plain_text((metadata.get("Artist") or {}).get("value")) if isinstance(metadata, Mapping) else None
        license_name = _plain_text((metadata.get("LicenseShortName") or {}).get("value")) if isinstance(metadata, Mapping) else None
        license_url = _plain_text((metadata.get("LicenseUrl") or {}).get("value")) if isinstance(metadata, Mapping) else None
        duration = _duration_from_commonmetadata(info.get("commonmetadata"))
        rest = {}
        if duration is None or not info.get("width") or not info.get("height"):
            try:
                rest = self.rest_file_info(str(page.get("title") or title))
            except requests.RequestException:
                rest = {}
        return {
            "canonical_title": str(page.get("title") or title),
            "media_url": info.get("url") or rest.get("url"),
            "source_sha1": info.get("sha1"),
            "source_size_bytes": info.get("size") or rest.get("size"),
            "mime": info.get("mime") or rest.get("mimetype"),
            "mediatype": info.get("mediatype"),
            "width": info.get("width") or rest.get("width"),
            "height": info.get("height") or rest.get("height"),
            "duration_seconds": duration if duration is not None else rest.get("duration"),
            "author": author,
            "license_name": license_name,
            "license_url": license_url,
            "license_state": _license_review_state(categories, license_name, license_url),
            "file_categories": sorted(categories),
        }

    def rest_file_info(self, title: str) -> dict[str, Any]:
        url = WIKIMEDIA_REST + quote(title, safe=":")
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        original = payload.get("original") if isinstance(payload, Mapping) else None
        return dict(original) if isinstance(original, Mapping) else {}

    def is_downloadable(self, url: str) -> bool:
        try:
            response = self.session.head(url, allow_redirects=True, timeout=self.timeout)
            if response.status_code < 400:
                return True
        except requests.RequestException:
            pass
        try:
            with self.session.get(
                url,
                headers={"Range": "bytes=0-0"},
                allow_redirects=True,
                stream=True,
                timeout=self.timeout,
            ) as response:
                return response.status_code in {200, 206}
        except requests.RequestException:
            return False


def enumerate_region_files(
    client: MediaWikiClient,
    region: Mapping[str, Any],
) -> tuple[dict[str, set[str]], list[dict[str, str]], int]:
    files: dict[str, set[str]] = {}
    failures: list[dict[str, str]] = []
    reachable_seed_count = 0

    for seed in region["seeds"]:
        if seed["state"] != "active":
            continue
        visited: set[str] = set()
        root = str(seed["category"])
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        root_ok = False
        max_depth = int(seed.get("max_depth", 4 if seed.get("recursive", True) else 0))
        while queue:
            category, depth = queue.popleft()
            if category in visited:
                continue
            visited.add(category)
            try:
                members = client.category_members(category)
                if category == root:
                    root_ok = True
            except Exception as exc:
                failures.append(
                    {
                        "region": str(region["id"]),
                        "seed": root,
                        "category": category,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for item in members:
                item_type = item.get("type")
                title = item.get("title")
                if not isinstance(title, str):
                    continue
                if item_type == "file":
                    files.setdefault(title, set()).add(category)
                elif item_type == "subcat" and seed.get("recursive", True) and depth < max_depth:
                    queue.append((title, depth + 1))
        reachable_seed_count += int(root_ok)

    return files, failures, reachable_seed_count


def stage_a_gate(candidate: Mapping[str, Any], *, minimum_duration_seconds: float) -> dict[str, Any]:
    license_record = candidate.get("license")
    resolution = candidate.get("resolution")
    checks = {
        "source_page_present": bool(candidate.get("source_page")),
        "media_url_present": bool(candidate.get("media_url")),
        "downloadability_checked": candidate.get("downloadable") is True,
        "video_media_confirmed": str(candidate.get("mime") or "").startswith("video/")
        or str(candidate.get("mediatype") or "").upper() == "VIDEO",
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
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "eligible_for_preflight": not failed, "failed_requirements": failed}


def discover(
    seed_path: str | Path = DEFAULT_SEEDS,
    *,
    check_downloadability: bool = True,
    client: MediaWikiClient | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_seed_config(seed_path)
    client = client or MediaWikiClient()
    candidates_by_title: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    region_stats: dict[str, dict[str, Any]] = {}
    minimum_duration = float(config.get("minimum_duration_seconds", 120))

    for region in config["regions"]:
        region_id = str(region["id"])
        files, region_failures, reachable_seed_count = enumerate_region_files(client, region)
        failures.extend(region_failures)
        region_stats[region_id] = {
            "configured_seed_count": sum(1 for seed in region["seeds"] if seed["state"] == "active"),
            "reachable_seed_count": reachable_seed_count,
            "discovered_file_count": len(files),
        }
        for title, categories in files.items():
            current = candidates_by_title.setdefault(
                title,
                {
                    "regions": set(),
                    "discovered_categories": set(),
                },
            )
            current["regions"].add(region_id)
            current["discovered_categories"].update(categories)

    normalized: list[dict[str, Any]] = []
    for title in sorted(candidates_by_title):
        discovery = candidates_by_title[title]
        try:
            info = client.video_info(title)
        except Exception as exc:
            failures.append(
                {
                    "region": ",".join(sorted(discovery["regions"])),
                    "seed": "",
                    "category": ",".join(sorted(discovery["discovered_categories"])),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        is_video = str(info.get("mime") or "").startswith("video/") or str(info.get("mediatype") or "").upper() == "VIDEO"
        if not is_video:
            continue
        media_url = str(info["media_url"]) if info.get("media_url") else None
        downloadable = client.is_downloadable(media_url) if check_downloadability and media_url else None
        candidate = {
            "id": _stable_id(str(info["canonical_title"])),
            "status": "discovered",
            "evaluation_stage": "metadata",
            "regions": sorted(discovery["regions"]),
            "country": None,
            "locality": None,
            "title": str(info["canonical_title"]).removeprefix("File:"),
            "target": None,
            "provider": "Wikimedia Commons",
            "canonical_file_title": str(info["canonical_title"]),
            "source_page": _canonical_file_url(str(info["canonical_title"])),
            "media_url": media_url,
            "author": info.get("author"),
            "license": {
                "name": info.get("license_name"),
                "status": info.get("license_state"),
                "url": info.get("license_url"),
            },
            "duration_seconds": info.get("duration_seconds"),
            "resolution": [info["width"], info["height"]]
            if isinstance(info.get("width"), int) and isinstance(info.get("height"), int)
            else None,
            "source_sha1": info.get("source_sha1"),
            "source_size_bytes": info.get("source_size_bytes"),
            "mime": info.get("mime"),
            "mediatype": info.get("mediatype"),
            "downloadable": downloadable,
            "discovered_categories": sorted(discovery["discovered_categories"]),
            "file_categories": info.get("file_categories") or [],
            "measurements": {"preflight": None, "colmap": None, "splat": None},
        }
        candidate["stage_a"] = stage_a_gate(candidate, minimum_duration_seconds=minimum_duration)
        if FORBIDDEN_HEURISTIC_FIELDS.intersection(candidate):
            raise AssertionError("heuristic ranking fields must never be emitted")
        normalized.append(candidate)

    pool = {
        "schema_version": 1,
        "source_authority": "Wikimedia Commons MediaWiki Action API categorymembers + videoinfo",
        "seed_config": str(Path(seed_path).as_posix()),
        "minimum_duration_seconds": minimum_duration,
        "regions": [str(region["id"]) for region in config["regions"]],
        "candidates": sorted(normalized, key=lambda item: (item["regions"], item["canonical_file_title"])),
        "discovery_failures": sorted(failures, key=lambda item: (item["region"], item["category"], item["error"])),
    }
    coverage = build_coverage(pool, config, region_stats)
    return pool, coverage


def build_coverage(
    pool: Mapping[str, Any],
    config: Mapping[str, Any],
    region_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = pool.get("candidates") or []
    failures = pool.get("discovery_failures") or []
    region_stats = region_stats or {}
    rows: list[dict[str, Any]] = []
    for region in config["regions"]:
        region_id = str(region["id"])
        region_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping) and region_id in (candidate.get("regions") or [])
        ]
        region_failures = [
            failure
            for failure in failures
            if isinstance(failure, Mapping) and region_id in str(failure.get("region") or "").split(",")
        ]
        stats = region_stats.get(region_id, {})
        active_seeds = [seed for seed in region["seeds"] if seed["state"] == "active"]
        row = {
            "region": region_id,
            "configured_seed_count": int(stats.get("configured_seed_count", len(active_seeds))),
            "reachable_seed_count": int(stats.get("reachable_seed_count", 0)),
            "discovered_file_count": int(stats.get("discovered_file_count", 0)),
            "confirmed_video_count": len(region_candidates),
            "metadata_complete_count": sum(
                1
                for candidate in region_candidates
                if candidate.get("source_page")
                and candidate.get("media_url")
                and candidate.get("author")
                and candidate.get("duration_seconds") is not None
                and candidate.get("resolution")
            ),
            "verified_license_count": sum(
                1
                for candidate in region_candidates
                if isinstance(candidate.get("license"), Mapping)
                and candidate["license"].get("status") == "verified"
            ),
            "stage_a_pass_count": sum(
                1 for candidate in region_candidates if (candidate.get("stage_a") or {}).get("eligible_for_preflight")
            ),
            "stage_a_fail_count": sum(
                1 for candidate in region_candidates if not (candidate.get("stage_a") or {}).get("eligible_for_preflight")
            ),
            "discovery_failure_count": len(region_failures),
            "missing_seed_count": sum(1 for seed in region["seeds"] if seed["state"] == "missing"),
            "failure_reasons": sorted({str(failure.get("error")) for failure in region_failures}),
        }
        rows.append(row)
    return {
        "schema_version": 1,
        "regions": rows,
        "totals": {
            "candidate_count": len(candidates),
            "stage_a_pass_count": sum(
                1 for candidate in candidates if (candidate.get("stage_a") or {}).get("eligible_for_preflight")
            ),
            "discovery_failure_count": len(failures),
        },
    }


def load_pool(path: str | Path = DEFAULT_POOL) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("candidates"), list):
        raise ValueError("invalid Nordic discovery pool")
    ids = [candidate.get("id") for candidate in payload["candidates"]]
    if any(not source_id for source_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Nordic discovery candidate ids must be non-empty and unique")
    for candidate in payload["candidates"]:
        if FORBIDDEN_HEURISTIC_FIELDS.intersection(candidate):
            raise ValueError("heuristic ranking fields are forbidden in Nordic discovery pool")
    return payload


def candidate_by_id(pool: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    for candidate in pool["candidates"]:
        if candidate.get("id") == source_id:
            return dict(candidate)
    raise KeyError(f"unknown Nordic candidate: {source_id}")


def _registry_entry(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not (candidate.get("stage_a") or {}).get("eligible_for_preflight"):
        raise ValueError(f"{candidate.get('id')}: Stage A did not pass")
    license_record = candidate["license"]
    return {
        "id": candidate["id"],
        "status": "candidate",
        "evaluation_stage": "metadata",
        "title": candidate["title"],
        "provider": "Wikimedia Commons",
        "source_page": candidate["source_page"],
        "media_url": candidate["media_url"],
        "author": candidate["author"],
        "license": {
            "name": license_record["name"],
            "status": "verified",
            "url": license_record["url"],
        },
        "duration_seconds": candidate["duration_seconds"],
        "resolution": candidate["resolution"],
        "target": candidate.get("target") or candidate["title"],
        "metadata_notes": [
            "Promoted from Nordic Wikimedia discovery pool after Stage A metadata gate."
        ],
        "measurements": {"preflight": None, "colmap": None, "splat": None},
        "metadata_evidence": {
            "authority": "Wikimedia Commons MediaWiki Action API videoinfo/extmetadata",
            "resolved_via": "nordic-discovery",
            "source_sha1": candidate.get("source_sha1"),
            "source_size_bytes": candidate.get("source_size_bytes"),
            "mime": candidate.get("mime"),
            "download_url_available": candidate.get("downloadable") is True,
            "license_verified": True,
        },
    }


def promote_candidate(
    source_id: str,
    *,
    pool_path: str | Path = DEFAULT_POOL,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    pool = load_pool(pool_path)
    candidate = candidate_by_id(pool, source_id)
    entry = _registry_entry(candidate)
    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    videos = registry.get("videos")
    if not isinstance(videos, list):
        raise ValueError("invalid video registry")
    existing = next((item for item in videos if item.get("id") == source_id), None)
    if existing is None:
        videos.append(entry)
    else:
        if existing.get("source_page") != entry["source_page"]:
            raise ValueError(f"{source_id}: registry id collision with different source page")
        for key in (
            "title",
            "provider",
            "media_url",
            "author",
            "license",
            "duration_seconds",
            "resolution",
            "target",
            "metadata_evidence",
        ):
            existing[key] = entry[key]
    registry_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def materialize_and_preflight(
    source_id: str,
    *,
    pool_path: str | Path = DEFAULT_POOL,
    registry_path: str | Path = DEFAULT_REGISTRY,
    input_root: str | Path = "input",
    output_root: str | Path = "output",
) -> dict[str, Any]:
    pool = load_pool(pool_path)
    candidate = candidate_by_id(pool, source_id)
    promote_candidate(source_id, pool_path=pool_path, registry_path=registry_path)
    suffix = Path(urlparse(str(candidate["media_url"])).path).suffix.lower()
    if suffix not in {".webm", ".ogv", ".ogg", ".mp4"}:
        suffix = ".video"
    source_path = Path(input_root) / source_id / f"source{suffix}"
    ensure_source(
        source_path,
        url=str(candidate["media_url"]),
        expected_sha1=str(candidate["source_sha1"]) if candidate.get("source_sha1") else None,
        expected_size=int(candidate["source_size_bytes"]) if candidate.get("source_size_bytes") else None,
    )
    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    source = next(item for item in registry["videos"] if item["id"] == source_id)
    source["sha256"] = sha256_file(source_path)
    evidence = dict(source.get("metadata_evidence") or {})
    evidence["downloaded_size_bytes"] = source_path.stat().st_size
    evidence["sha256_verified_from_downloaded_bytes"] = True
    source["metadata_evidence"] = evidence
    registry_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_path = Path(output_root) / source_id / "preflight.json"
    result = run_video_preflight(source_path, result_path)
    apply_preflight_to_registry(registry_file, source_id, result)
    return {
        "source_id": source_id,
        "source_path": str(source_path),
        "source_sha256": source["sha256"],
        "preflight_manifest": str(result_path),
        "metrics": result["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and gate Nordic Wikimedia video candidates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    discover_parser.add_argument("--pool", default=str(DEFAULT_POOL))
    discover_parser.add_argument("--coverage", default=str(DEFAULT_COVERAGE))
    discover_parser.add_argument("--skip-download-check", action="store_true")
    discover_parser.add_argument("--allow-partial", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    validate_parser.add_argument("--pool", default=str(DEFAULT_POOL))

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("source_id")
    promote_parser.add_argument("--pool", default=str(DEFAULT_POOL))
    promote_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("source_id")
    preflight_parser.add_argument("--pool", default=str(DEFAULT_POOL))
    preflight_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    preflight_parser.add_argument("--input-root", default="input")
    preflight_parser.add_argument("--output-root", default="output")

    args = parser.parse_args()
    if args.command == "discover":
        pool, coverage = discover(args.seeds, check_downloadability=not args.skip_download_check)
        if pool["discovery_failures"] and not args.allow_partial:
            print(json.dumps({"status": "failed", "failure_count": len(pool["discovery_failures"]), "failures": pool["discovery_failures"]}, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        pool_changed = persist_snapshot(args.pool, pool)
        coverage_changed = persist_snapshot(args.coverage, coverage)
        print(json.dumps({"pool_changed": pool_changed, "coverage_changed": coverage_changed, "candidate_count": len(pool["candidates"]), "stage_a_pass_count": coverage["totals"]["stage_a_pass_count"]}, indent=2))
    elif args.command == "validate":
        seeds = load_seed_config(args.seeds)
        pool = load_pool(args.pool)
        coverage = build_coverage(pool, seeds)
        if [row["region"] for row in coverage["regions"]] != list(REQUIRED_REGIONS):
            raise SystemExit("Nordic coverage is incomplete")
        print(json.dumps({"regions": len(coverage["regions"]), "candidates": len(pool["candidates"])}, indent=2))
    elif args.command == "promote":
        print(json.dumps(promote_candidate(args.source_id, pool_path=args.pool, registry_path=args.registry), ensure_ascii=False, indent=2))
    elif args.command == "preflight":
        print(json.dumps(materialize_and_preflight(args.source_id, pool_path=args.pool, registry_path=args.registry, input_root=args.input_root, output_root=args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
