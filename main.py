from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.filters import laplace
from skimage.metrics import structural_similarity
from skimage.transform import resize

USER_AGENT = "AutoPhotogrammetry/0.2 (+explicit-source prototype)"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
FEATURE_IMAGE_SIZE = (256, 256)


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
            if int(declared_length) > max_bytes:
                raise ValueError(
                    f"Image is larger than the configured limit: {declared_length} bytes"
                )
        except ValueError as exc:
            if "larger" in str(exc):
                raise

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
    """明示されたHTMLページから画像を収集し、来歴付きレコードを返す。

    検索エンジンの無断スクレイピングは行わない。利用者は各ページの規約、
    robots.txt、画像ライセンスを確認する必要がある。
    """

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
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def _rgb_array(image_path: str | Path, size: tuple[int, int] = FEATURE_IMAGE_SIZE) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return resize(rgb, size, anti_aliasing=True, preserve_range=True)


def extract_features(image_path: str | Path) -> np.ndarray:
    """画像寸法に依存しない固定長特徴量を返す。"""

    rgb = _rgb_array(image_path)
    gray = rgb2gray(rgb)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    lbp_histogram, _ = np.histogram(
        lbp,
        bins=np.arange(0, 11),
        range=(0, 10),
        density=True,
    )

    color_histograms = [
        np.histogram(rgb[..., channel], bins=16, range=(0, 1), density=True)[0]
        for channel in range(3)
    ]
    feature_vector = np.concatenate(
        [hog_features, lbp_histogram, *color_histograms]
    ).astype(np.float64)
    if not np.isfinite(feature_vector).all():
        raise ValueError(f"Non-finite feature detected: {image_path}")
    return feature_vector


def cluster_images(
    image_paths: Sequence[str | Path],
    *,
    eps: float = 1.5,
    min_samples: int = 3,
) -> np.ndarray:
    if eps <= 0 or min_samples < 1:
        raise ValueError("eps must be positive and min_samples must be at least 1")
    if not image_paths:
        return np.array([], dtype=int)

    matrix = np.vstack([extract_features(path) for path in image_paths])
    scaled = StandardScaler().fit_transform(matrix)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)


def calculate_sharpness(image_path: str | Path) -> float:
    gray = rgb2gray(_rgb_array(image_path))
    return float(np.var(laplace(gray)))


def calculate_similarity(image_path1: str | Path, image_path2: str | Path) -> float:
    image1 = np.clip(rgb2gray(_rgb_array(image_path1)) * 255, 0, 255).astype(np.uint8)
    image2 = np.clip(rgb2gray(_rgb_array(image_path2)) * 255, 0, 255).astype(np.uint8)
    return float(structural_similarity(image1, image2, data_range=255))


def select_images(
    image_paths: Iterable[str | Path],
    *,
    sharpness_threshold: float,
    similarity_threshold: float,
    output_dir: str | Path,
) -> list[Path]:
    """鮮明度順に重複を除外し、元画像を移動せずコピーする。"""

    if sharpness_threshold < 0:
        raise ValueError("sharpness_threshold must be non-negative")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    ranked = sorted(
        ((Path(path), calculate_sharpness(path)) for path in image_paths),
        key=lambda item: (-item[1], str(item[0])),
    )
    selected_sources: list[Path] = []
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for image_path, sharpness in ranked:
        if sharpness < sharpness_threshold:
            continue
        if any(
            calculate_similarity(image_path, selected) >= similarity_threshold
            for selected in selected_sources
        ):
            continue
        selected_sources.append(image_path)
        shutil.copy2(image_path, destination_dir / image_path.name)

    return [destination_dir / source.name for source in selected_sources]


def _write_cluster_manifest(
    records: Sequence[ImageRecord], labels: Sequence[int], output_path: Path
) -> None:
    rows = [
        {**asdict(record), "cluster": int(label)}
        for record, label in zip(records, labels, strict=True)
    ]
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect explicitly sourced real photographs, create fixed-length image "
            "features, cluster them, and copy non-duplicate sharp images."
        )
    )
    parser.add_argument(
        "--page-url",
        action="append",
        required=True,
        help="Explicit HTML page URL. Repeat for multiple pages.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Optional URL/alt-text filter. Repeat for multiple keywords.",
    )
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sharpness-threshold", type=float, default=0.0001)
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    arguments = parser.parse_args()

    work_dir = Path(arguments.work_dir)
    collected_dir = work_dir / "collected"
    selected_dir = work_dir / "selected"

    records = collect_images(
        arguments.keyword,
        arguments.page_url,
        collected_dir,
    )
    if not records:
        raise SystemExit("No valid images were collected")

    image_paths = [record.local_path for record in records]
    labels = cluster_images(
        image_paths,
        eps=arguments.eps,
        min_samples=arguments.min_samples,
    )
    _write_cluster_manifest(records, labels, work_dir / "clusters.json")

    selected = select_images(
        image_paths,
        sharpness_threshold=arguments.sharpness_threshold,
        similarity_threshold=arguments.similarity_threshold,
        output_dir=selected_dir,
    )
    print(
        json.dumps(
            {
                "collected": len(records),
                "selected": len(selected),
                "clusters": sorted({int(value) for value in labels}),
                "generated_views_used": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
