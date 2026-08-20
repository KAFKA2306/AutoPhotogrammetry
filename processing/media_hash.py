from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import BinaryIO

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


def hash_source_media(source: dict, *, timeout_seconds: float = 120.0) -> tuple[str, int]:
    media_url = source.get("media_url")
    if not media_url:
        raise ValueError(f"{source.get('id')}: media_url is required")

    request = urllib.request.Request(
        str(media_url),
        headers={"User-Agent": "KAFKA2306-AutoPhotogrammetry/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        sha256, size = sha256_stream(response)

    expected_size = (source.get("metadata_evidence") or {}).get("source_size_bytes")
    if expected_size is not None and int(expected_size) != size:
        raise ValueError(
            f"{source.get('id')}: downloaded byte size {size} does not match Commons metadata {expected_size}"
        )
    return sha256, size


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

    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"id": source_id, "sha256": sha256, "size_bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one exact registry media URL as a stream and freeze its SHA-256."
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = update_registry_source_hash(
        args.source_id,
        args.registry,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
