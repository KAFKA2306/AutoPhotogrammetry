"""Official Hugging Face Storage Bucket upload and read-back verification."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from processing.provenance import sha256_file


def publish_and_verify(
    bucket_id: str,
    local_path: str | Path,
    remote_path: str,
    *,
    uploader: Callable[..., object] | None = None,
    downloader: Callable[..., object] | None = None,
    token: str | bool | None = None,
) -> dict[str, object]:
    """Upload one artifact with the official API, then verify exact read-back."""
    from huggingface_hub import batch_bucket_files, download_bucket_files

    path = Path(local_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"artifact is missing or empty: {path}")
    expected_sha = sha256_file(path)
    expected_size = path.stat().st_size
    upload = uploader or batch_bucket_files
    download = downloader or download_bucket_files
    upload(bucket_id, add=[(str(path), remote_path)], token=token)
    with tempfile.TemporaryDirectory(prefix="hf-artifact-readback-") as temp_dir:
        readback = Path(temp_dir) / path.name
        download(
            bucket_id,
            files=[(remote_path, readback)],
            raise_on_missing_files=True,
            token=token,
        )
        actual_sha = sha256_file(readback)
        actual_size = readback.stat().st_size
    if actual_sha != expected_sha or actual_size != expected_size:
        raise ValueError(
            "Hugging Face read-back mismatch: "
            f"expected {expected_sha}/{expected_size}, got {actual_sha}/{actual_size}"
        )
    return {
        "status": "PUBLISHED",
        "remote_verified": True,
        "remote_uri": f"hf://buckets/{bucket_id}/{remote_path}",
        "sha256": expected_sha,
        "size_bytes": expected_size,
    }
