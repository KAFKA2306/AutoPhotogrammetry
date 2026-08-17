from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.metrics import structural_similarity
from skimage.transform import resize

IMAGE_SIZE = (256, 256)


def _rgb_array(
    image_path: str | Path,
    size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return resize(rgb, size, anti_aliasing=True, preserve_range=True)


def calculate_sharpness(image_path: str | Path) -> float:
    gray = rgb2gray(_rgb_array(image_path))
    return float(np.var(laplace(gray)))


def calculate_similarity(
    image_path1: str | Path,
    image_path2: str | Path,
) -> float:
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


def select_video_frames(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    sharpness_threshold: float = 0.0001,
    similarity_threshold: float = 0.92,
    sharpness_fn: Callable[[str | Path], float] | None = None,
    similarity_fn: Callable[[str | Path, str | Path], float] | None = None,
) -> dict:
    """Keep sharp frames and remove near-duplicates of the last accepted frame."""
    if sharpness_threshold < 0:
        raise ValueError("sharpness_threshold must be non-negative")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    sharpness_fn = sharpness_fn or calculate_sharpness
    similarity_fn = similarity_fn or calculate_similarity
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    rejected_blur = 0
    rejected_duplicate = 0

    for raw_path in sorted(map(Path, frame_paths)):
        if sharpness_fn(raw_path) < sharpness_threshold:
            rejected_blur += 1
            continue
        if selected and similarity_fn(raw_path, selected[-1]) >= similarity_threshold:
            rejected_duplicate += 1
            continue
        target = destination / raw_path.name
        shutil.copy2(raw_path, target)
        selected.append(target)

    return {
        "input": len(frame_paths),
        "selected": len(selected),
        "rejected_blur": rejected_blur,
        "rejected_duplicate": rejected_duplicate,
        "selected_paths": [str(path) for path in selected],
    }
