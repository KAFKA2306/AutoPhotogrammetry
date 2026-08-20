from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from processing.nordic_seeds import enabled_seed_categories, load_nordic_seeds

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AutoPhotogrammetry/0.7 (+Nordic Commons discovery)"
JsonRequester = Callable[[Mapping[str, str]], dict[str, Any]]


def _request_json(params: Mapping[str, str]) -> dict[str, Any]:
    query = urlencode({"action": "query", "format": "json", "formatversion": "2", **params})
    request = Request(
        f"{WIKIMEDIA_API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Wikimedia API returned a non-object payload")
    if payload.get("error"):
        raise RuntimeError(f"Wikimedia API error: {payload['error']}")
    return payload


def _category_pages(
    category_title: str,
    *,
    request_json: JsonRequester,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    continuation: dict[str, str] = {}
    seen_continuations: set[tuple[tuple[str, str], ...]] = set()

    while True:
        params = {
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "file|subcat",
            "cmprop": "ids|title|type",
            "cmlimit": "max",
            **continuation,
        }
        payload = request_json(params)
        query = payload.get("query")
        if not isinstance(query, Mapping):
            raise RuntimeError(f"categorymembers response has no query object: {category_title}")
        page_members = query.get("categorymembers")
        if not isinstance(page_members, list):
            raise RuntimeError(
                f"categorymembers response has no categorymembers list: {category_title}"
            )
        for member in page_members:
            if isinstance(member, Mapping):
                members.append(dict(member))

        next_value = payload.get("continue")
        if not isinstance(next_value, Mapping):
            break
        continuation = {
            str(key): str(value)
            for key, value in next_value.items()
            if isinstance(key, str) and value is not None
        }
        if not continuation:
            break
        identity = tuple(sorted(continuation.items()))
        if identity in seen_continuations:
            raise RuntimeError(f"repeated categorymembers continuation: {category_title}")
        seen_continuations.add(identity)

    return members


def _member_type(member: Mapping[str, Any]) -> str | None:
    member_type = member.get("type")
    if member_type in {"file", "subcat"}:
        return str(member_type)
    namespace = member.get("ns")
    if namespace == 6:
        return "file"
    if namespace == 14:
        return "subcat"
    return None


def traverse_category(
    root_category: str,
    *,
    request_json: JsonRequester = _request_json,
) -> list[dict[str, Any]]:
    """Recursively enumerate Commons files with deterministic category-path provenance."""
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(root_category, (root_category,))])
    visited_categories: set[str] = set()
    files: dict[str, dict[str, Any]] = {}

    while queue:
        category, path = queue.popleft()
        if category in visited_categories:
            continue
        visited_categories.add(category)
        members = _category_pages(category, request_json=request_json)
        for member in sorted(members, key=lambda item: str(item.get("title", ""))):
            title = member.get("title")
            if not isinstance(title, str) or not title:
                continue
            kind = _member_type(member)
            if kind == "subcat":
                if title not in visited_categories:
                    queue.append((title, (*path, title)))
                continue
            if kind != "file":
                continue

            record = files.setdefault(
                title,
                {
                    "canonical_title": title,
                    "discovered_categories": set(),
                    "discovery_paths": set(),
                },
            )
            record["discovered_categories"].add(category)
            record["discovery_paths"].add(path)

    result = []
    for title in sorted(files):
        record = files[title]
        result.append(
            {
                "canonical_title": title,
                "discovered_categories": sorted(record["discovered_categories"]),
                "discovery_paths": [
                    list(path) for path in sorted(record["discovery_paths"])
                ],
            }
        )
    return result


def discover_from_seed_config(
    seed_config: Mapping[str, Any] | None = None,
    *,
    request_json: JsonRequester = _request_json,
) -> dict[str, Any]:
    """Discover every configured region independently; one failure never erases another."""
    config = dict(seed_config) if seed_config is not None else load_nordic_seeds()
    seeds = enabled_seed_categories(config)
    files: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for seed in seeds:
        try:
            discovered = traverse_category(
                seed["category_title"],
                request_json=request_json,
            )
        except Exception as exc:
            failures.append(
                {
                    "region_id": seed["region_id"],
                    "category_title": seed["category_title"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for candidate in discovered:
            title = candidate["canonical_title"]
            record = files.setdefault(
                title,
                {
                    "canonical_title": title,
                    "regions": set(),
                    "discovered_categories": set(),
                    "discovery_paths": set(),
                },
            )
            record["regions"].add(seed["region_id"])
            record["discovered_categories"].update(candidate["discovered_categories"])
            record["discovery_paths"].update(
                tuple(path) for path in candidate["discovery_paths"]
            )

    candidates = []
    for title in sorted(files):
        record = files[title]
        candidates.append(
            {
                "canonical_title": title,
                "regions": sorted(record["regions"]),
                "discovered_categories": sorted(record["discovered_categories"]),
                "discovery_paths": [
                    list(path) for path in sorted(record["discovery_paths"])
                ],
            }
        )
    return {
        "schema_version": 1,
        "authority": "Wikimedia Commons categorymembers",
        "candidates": candidates,
        "failures": sorted(
            failures,
            key=lambda item: (item["region_id"], item["category_title"]),
        ),
    }


def _plain_text(value: object) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return text or None


def _metadata_map(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, list):
        return {}
    result: dict[str, object] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        key = item.get("name") or item.get("key")
        if isinstance(key, str):
            result[key] = item.get("value")
    return result


def _metadata_value(metadata: Mapping[str, object], key: str) -> object:
    value = metadata.get(key)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _source_page(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/" + quote(
        title.replace(" ", "_"),
        safe=":(),",
    )


def fetch_videoinfo(
    canonical_title: str,
    *,
    request_json: JsonRequester = _request_json,
) -> dict[str, Any]:
    payload = request_json(
        {
            "prop": "videoinfo|categories",
            "titles": canonical_title,
            "viprop": "url|size|dimensions|sha1|mime|mediatype|commonmetadata|extmetadata",
            "viextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
            "cllimit": "max",
        }
    )
    query = payload.get("query")
    if not isinstance(query, Mapping):
        raise RuntimeError(f"videoinfo response has no query object: {canonical_title}")
    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        raise RuntimeError(f"videoinfo response has no page: {canonical_title}")
    page = pages[0]
    if not isinstance(page, Mapping) or page.get("missing"):
        raise RuntimeError(f"Commons file is missing: {canonical_title}")
    infos = page.get("videoinfo")
    if not isinstance(infos, list) or not infos or not isinstance(infos[0], Mapping):
        raise RuntimeError(f"Commons file has no videoinfo: {canonical_title}")
    return {"page": dict(page), "videoinfo": dict(infos[0])}


def normalize_videoinfo(
    discovery_record: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_title = discovery_record.get("canonical_title")
    if not isinstance(canonical_title, str) or not canonical_title:
        raise ValueError("discovery record requires canonical_title")

    page = response.get("page")
    info = response.get("videoinfo")
    if not isinstance(page, Mapping) or not isinstance(info, Mapping):
        raise ValueError("normalized videoinfo requires page and videoinfo objects")

    extmetadata = _metadata_map(info.get("extmetadata"))
    commonmetadata = _metadata_map(info.get("commonmetadata"))
    categories = sorted(
        str(category["title"])
        for category in page.get("categories", [])
        if isinstance(category, Mapping) and isinstance(category.get("title"), str)
    )
    review_needed = any("license review needed" in title.casefold() for title in categories)

    license_name = _plain_text(_metadata_value(extmetadata, "LicenseShortName"))
    license_url = _plain_text(_metadata_value(extmetadata, "LicenseUrl"))
    author = _plain_text(_metadata_value(extmetadata, "Artist"))
    if review_needed:
        license_state = "needs_review"
    elif license_name and license_url:
        license_state = "verified"
    else:
        license_state = "unknown"

    mime = _plain_text(info.get("mime"))
    media_type = _plain_text(info.get("mediatype"))
    confirmed_video = bool(
        (mime and mime.casefold().startswith("video/"))
        or (media_type and media_type.casefold() == "video")
    )

    dimensions = info.get("dimensions")
    width = info.get("width")
    height = info.get("height")
    if isinstance(dimensions, Mapping):
        width = width if width is not None else dimensions.get("width")
        height = height if height is not None else dimensions.get("height")

    duration = info.get("duration")
    if duration is None:
        duration = _metadata_value(commonmetadata, "duration")

    return {
        "canonical_title": canonical_title,
        "source_page": _source_page(canonical_title),
        "media_url": _plain_text(info.get("url")),
        "source_sha1": _plain_text(info.get("sha1")),
        "source_size_bytes": info.get("size") if isinstance(info.get("size"), int) else None,
        "mime": mime,
        "media_type": media_type,
        "confirmed_video": confirmed_video,
        "width": width if isinstance(width, int) else None,
        "height": height if isinstance(height, int) else None,
        "duration_seconds": float(duration) if isinstance(duration, (int, float)) else None,
        "author": author,
        "license": {
            "name": license_name,
            "url": license_url,
            "status": license_state,
        },
        "commons_categories": categories,
        "regions": sorted(str(value) for value in discovery_record.get("regions", [])),
        "discovered_categories": sorted(
            str(value) for value in discovery_record.get("discovered_categories", [])
        ),
        "discovery_paths": sorted(
            [list(path) for path in discovery_record.get("discovery_paths", [])],
        ),
        "metadata_authority": "Wikimedia Commons videoinfo/categories",
    }


def normalize_discovery(
    discovery: Mapping[str, Any],
    *,
    request_json: JsonRequester = _request_json,
) -> dict[str, Any]:
    candidates = discovery.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("discovery payload requires candidates list")

    normalized_by_identity: dict[str, dict[str, Any]] = {}
    metadata_failures: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        title = candidate.get("canonical_title")
        if not isinstance(title, str):
            continue
        try:
            response = fetch_videoinfo(title, request_json=request_json)
            normalized = normalize_videoinfo(candidate, response)
        except Exception as exc:
            metadata_failures.append(
                {
                    "canonical_title": title,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not normalized["confirmed_video"]:
            continue

        identity = normalized.get("source_sha1") or normalized["canonical_title"]
        existing = normalized_by_identity.get(str(identity))
        if existing is None:
            normalized_by_identity[str(identity)] = normalized
            continue
        existing["regions"] = sorted(set(existing["regions"]) | set(normalized["regions"]))
        existing["discovered_categories"] = sorted(
            set(existing["discovered_categories"]) | set(normalized["discovered_categories"])
        )
        existing["discovery_paths"] = sorted(
            {
                tuple(path)
                for path in [*existing["discovery_paths"], *normalized["discovery_paths"]]
            }
        )
        existing["discovery_paths"] = [list(path) for path in existing["discovery_paths"]]

    return {
        "schema_version": 1,
        "authority": "Wikimedia Commons videoinfo/categories",
        "candidates": sorted(
            normalized_by_identity.values(),
            key=lambda item: item["canonical_title"],
        ),
        "discovery_failures": list(discovery.get("failures", [])),
        "metadata_failures": sorted(
            metadata_failures,
            key=lambda item: item["canonical_title"],
        ),
    }
