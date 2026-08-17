from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

from processing.provenance import write_json

USER_AGENT = "AutoPhotogrammetry/0.3 (+explicit-source)"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ImageRecord:
    source_page: str
    image_url: str
    local_path: str
    sha256: str
    width: int
    height: int
    content_type: str


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http/https URLs are accepted: {url!r}")
    if not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"Invalid or credential-bearing URL: {url!r}")
    return url


def _response_bytes(response: requests.Response, max_bytes: int) -> bytes:
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            size = int(declared_length)
        except ValueError:
            size = None
        if size is not None and size > max_bytes:
            raise ValueError(
                f"Image is larger than the configured limit: {declared_length} bytes"
            )

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"Downloaded image exceeded the configured limit: {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(data)) as candidate:
            return candidate.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Downloaded content is not a decodable image") from exc


def _download_image(
    session: requests.Session,
    source_page: str,
    image_url: str,
    output_dir: Path,
    timeout: int,
    max_bytes: int,
) -> ImageRecord:
    response = session.get(image_url, timeout=timeout, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"Response is not an image: {content_type or 'unknown'}")

    data = _response_bytes(response, max_bytes)
    digest = hashlib.sha256(data).hexdigest()
    image = _decode_image(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{digest}.jpg"
    if not destination.exists():
        image.save(destination, format="JPEG", quality=95, optimize=True)

    return ImageRecord(
        source_page=source_page,
        image_url=image_url,
        local_path=str(destination),
        sha256=digest,
        width=image.width,
        height=image.height,
        content_type=content_type,
    )


def collect_images(
    keywords: Sequence[str],
    page_urls: Sequence[str],
    output_dir: str | Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    session: requests.Session | None = None,
) -> list[ImageRecord]:
    """Collect images only from explicitly supplied HTML pages."""
    if timeout <= 0 or max_bytes <= 0:
        raise ValueError("timeout and max_bytes must be positive")
    if not page_urls:
        raise ValueError("At least one explicit page URL is required")

    normalized_keywords = [word.casefold() for word in keywords if word.strip()]
    output_path = Path(output_dir)
    client = session or requests.Session()
    client.headers.setdefault("User-Agent", USER_AGENT)

    records_by_hash: dict[str, ImageRecord] = {}
    for raw_page_url in page_urls:
        page_url = _validate_http_url(raw_page_url)
        page_response = client.get(page_url, timeout=timeout)
        page_response.raise_for_status()
        page_type = page_response.headers.get("Content-Type", "").lower()
        if "html" not in page_type:
            raise ValueError(f"Page URL did not return HTML: {page_url}")

        soup = BeautifulSoup(page_response.text, "html.parser")
        for tag in soup.find_all("img"):
            source = tag.get("src") or tag.get("data-src")
            if not source:
                continue
            image_url = urljoin(page_url, source)
            try:
                _validate_http_url(image_url)
            except ValueError:
                continue

            searchable = f"{image_url} {tag.get('alt', '')}".casefold()
            if normalized_keywords and not any(
                keyword in searchable for keyword in normalized_keywords
            ):
                continue

            try:
                record = _download_image(
                    client,
                    page_url,
                    image_url,
                    output_path,
                    timeout,
                    max_bytes,
                )
            except (requests.RequestException, ValueError) as exc:
                print(f"skip {image_url}: {exc}")
                continue
            records_by_hash.setdefault(record.sha256, record)

    records = sorted(records_by_hash.values(), key=lambda item: item.sha256)
    write_json(output_path / "manifest.json", [asdict(record) for record in records])
    return records
