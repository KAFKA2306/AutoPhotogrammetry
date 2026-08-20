from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from processing.batch import resolve_media_url
from processing.video_sources import load_video_registry


def _plain_text(value: object) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return text or None


def _verified_license(source: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict[str, str]:
    current = source.get("license") or {}
    current_name = current.get("name") if isinstance(current, Mapping) else None
    current_url = current.get("url") if isinstance(current, Mapping) else None
    current_status = current.get("status") if isinstance(current, Mapping) else None

    if current_status == "verified" and current_name and current_url:
        return {"name": str(current_name), "status": "verified", "url": str(current_url)}

    name = _plain_text(resolved.get("license"))
    url = _plain_text(resolved.get("license_url"))
    if not name or not url:
        raise ValueError("Commons API did not return a verifiable license name and URL")
    return {"name": name, "status": "verified", "url": url}


def refresh_source_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh one Wikimedia source without downloading its video bytes.

    Existing pinned SHA-256/transcode URLs are preserved. Missing direct media and
    rights metadata are filled only from the Wikimedia Commons imageinfo API.
    """
    updated = dict(source)
    current_license = source.get("license") or {}
    needs_api = (
        not source.get("media_url")
        or not source.get("author")
        or not isinstance(current_license, Mapping)
        or current_license.get("status") != "verified"
        or not current_license.get("name")
        or not current_license.get("url")
    )

    if needs_api:
        # Force API lookup even when a registry media URL exists, because
        # resolve_media_url short-circuits registry URLs and would omit rights data.
        query_source = dict(source)
        query_source.pop("media_url", None)
        resolved = resolve_media_url(query_source)
        if resolved.get("resolved_via") != "wikimedia-api":
            raise ValueError(
                f"{source.get('id')}: exact Commons metadata unavailable via API; "
                f"resolved_via={resolved.get('resolved_via')}"
            )
    else:
        resolved = {
            "media_url": source["media_url"],
            "source_sha1": (source.get("metadata_evidence") or {}).get("source_sha1"),
            "source_size_bytes": (source.get("metadata_evidence") or {}).get("source_size_bytes"),
            "mime": (source.get("metadata_evidence") or {}).get("mime"),
            "author": source.get("author"),
            "license": current_license.get("name"),
            "license_url": current_license.get("url"),
            "resolved_via": "registry-verified",
        }

    if not source.get("media_url"):
        updated["media_url"] = str(resolved["media_url"])
    author = _plain_text(source.get("author")) or _plain_text(resolved.get("author"))
    if not author:
        raise ValueError(f"{source.get('id')}: Commons API did not return an author")
    updated["author"] = author
    updated["license"] = _verified_license(source, resolved)

    evidence = dict(source.get("metadata_evidence") or {})
    evidence.update(
        {
            "authority": "Wikimedia Commons imageinfo/extmetadata",
            "resolved_via": resolved.get("resolved_via"),
            "source_sha1": resolved.get("source_sha1") or evidence.get("source_sha1"),
            "source_size_bytes": resolved.get("source_size_bytes")
            or evidence.get("source_size_bytes"),
            "mime": resolved.get("mime") or evidence.get("mime"),
            "download_url_available": bool(updated.get("media_url")),
            "license_verified": updated["license"]["status"] == "verified",
        }
    )
    updated["metadata_evidence"] = evidence
    return updated


def refresh_registry(
    registry_path: str | Path = "sources/videos.json",
    *,
    request_delay_seconds: float = 0.5,
) -> dict[str, Any]:
    registry_ref = Path(registry_path)
    path = registry_ref.expanduser().resolve()
    registry = load_video_registry(path)
    refreshed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source in registry["videos"]:
        try:
            refreshed.append(refresh_source_metadata(source))
        except Exception as exc:
            refreshed.append(dict(source))
            failures.append(
                {
                    "id": str(source.get("id")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    registry["videos"] = refreshed
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "registry": registry_ref.as_posix(),
        "source_count": len(refreshed),
        "verified_license_count": sum(
            1 for source in refreshed if (source.get("license") or {}).get("status") == "verified"
        ),
        "direct_media_url_count": sum(1 for source in refreshed if source.get("media_url")),
        "author_count": sum(1 for source in refreshed if source.get("author")),
        "failure_count": len(failures),
        "failures": failures,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Wikimedia Commons Stage-A author/license/media metadata without downloading videos."
    )
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    args = parser.parse_args()
    result = refresh_registry(
        args.registry,
        request_delay_seconds=args.request_delay_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
