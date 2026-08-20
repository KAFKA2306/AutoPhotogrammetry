from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError

from processing.video_sources import load_video_registry


def sha256_stream(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _open_with_wikimedia_backoff(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    max_attempts: int = 4,
    minimum_retry_seconds: float = 60.0,
):
    for attempt in range(max_attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout_seconds)
        except HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                server_delay = float(retry_after) if retry_after else 0.0
            except (TypeError, ValueError):
                server_delay = 0.0
            exponential_delay = minimum_retry_seconds * (2**attempt)
            time.sleep(min(600.0, max(server_delay, exponential_delay)))
    raise RuntimeError("source-media retry loop ended unexpectedly")


def hash_source_media(source: dict, *, timeout_seconds: float = 120.0) -> tuple[str, int]:
    media_url = source.get("media_url")
    if not media_url:
        raise ValueError(f"{source.get('id')}: media_url is required")

    request = urllib.request.Request(
        str(media_url),
        headers={"User-Agent": "KAFKA2306-AutoPhotogrammetry/1.0"},
    )
    with _open_with_wikimedia_backoff(
        request,
        timeout_seconds=timeout_seconds,
    ) as response:
        sha256, size = sha256_stream(response)

    expected_size = (source.get("metadata_evidence") or {}).get("source_size_bytes")
    if expected_size is not None and int(expected_size) != size:
        raise ValueError(
            f"{source.get('id')}: downloaded byte size {size} does not match Commons metadata {expected_size}"
        )
    return sha256, size


def _apply_hash_result(source: dict, sha256: str, size: int) -> dict:
    source_id = str(source["id"])
    existing = source.get("sha256")
    if existing and existing != sha256:
        raise ValueError(
            f"{source_id}: existing sha256 {existing} does not match downloaded bytes {sha256}"
        )

    source["sha256"] = sha256
    evidence = dict(source.get("metadata_evidence") or {})
    evidence["downloaded_size_bytes"] = size
    evidence["sha256_verified_from_downloaded_bytes"] = True
    source["metadata_evidence"] = evidence
    return {"id": source_id, "sha256": sha256, "size_bytes": size}


def update_registry_source_hash(
    source_id: str,
    registry_path: str | Path = "sources/videos.json",
    *,
    timeout_seconds: float = 120.0,
) -> dict:
    path = Path(registry_path)
    registry = load_video_registry(path)
    source = next((item for item in registry["videos"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"Unknown video source: {source_id}")

    sha256, size = hash_source_media(source, timeout_seconds=timeout_seconds)
    result = _apply_hash_result(source, sha256, size)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def update_unhashed_registry_sources(
    registry_path: str | Path = "sources/videos.json",
    *,
    timeout_seconds: float = 120.0,
    request_delay_seconds: float = 0.0,
) -> dict:
    """Hash every registry source that lacks byte-verified SHA-256 evidence.

    Each source is isolated: one download failure does not hide successful hashes for
    other sources. The registry is persisted after every successful source so a long
    batch can resume without repeating completed downloads. An optional inter-source
    delay allows hosted evidence runs to respect Wikimedia rate limits.
    """
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")

    path = Path(registry_path)
    registry = load_video_registry(path)
    results: list[dict] = []
    failures: list[dict] = []
    skipped: list[str] = []

    unresolved = [
        source
        for source in registry["videos"]
        if not (
            source.get("sha256")
            and (source.get("metadata_evidence") or {}).get("sha256_verified_from_downloaded_bytes")
            is True
        )
    ]
    skipped.extend(str(source["id"]) for source in registry["videos"] if source not in unresolved)

    for index, source in enumerate(unresolved):
        try:
            sha256, size = hash_source_media(source, timeout_seconds=timeout_seconds)
            results.append(_apply_hash_result(source, sha256, size))
            path.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # preserve independent candidate evidence
            failures.append({"id": str(source.get("id")), "error": str(exc)})
        if request_delay_seconds > 0 and index < len(unresolved) - 1:
            time.sleep(request_delay_seconds)

    return {
        "hashed": results,
        "failed": failures,
        "skipped_verified": skipped,
        "hashed_count": len(results),
        "failed_count": len(failures),
        "skipped_verified_count": len(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download exact registry media URLs as streams and freeze SHA-256 identities."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-id")
    mode.add_argument("--all-unhashed", action="store_true")
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.all_unhashed:
        result = update_unhashed_registry_sources(
            args.registry,
            timeout_seconds=args.timeout_seconds,
            request_delay_seconds=args.request_delay_seconds,
        )
    else:
        result = update_registry_source_hash(
            args.source_id,
            args.registry,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.all_unhashed and result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
