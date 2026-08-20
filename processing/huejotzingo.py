from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from urllib.request import Request, urlopen

from processing.image_selection import select_video_frames
from processing.nerfstudio import nerfstudio_process_images_command, run_splatfacto_export
from processing.provenance import (
    VideoSource,
    sha256_file,
    source_revision,
    utc_now,
    write_json,
    write_source_manifest,
)
from processing.video import probe_video
from processing.video_sources import get_video_source

SOURCE_CONFIG = get_video_source("huejotzingo")
DATASET = SOURCE_CONFIG["id"]
SOURCE_PAGE = SOURCE_CONFIG["source_page"]
MEDIA_URL = SOURCE_CONFIG["media_url"]
SOURCE_SHA256 = SOURCE_CONFIG["sha256"]
EXPECTED_FRAME_COUNT = SOURCE_CONFIG["expected_frame_count"]
SOURCE = VideoSource(
    title=SOURCE_CONFIG["title"],
    source_page=SOURCE_PAGE,
    media_url=MEDIA_URL,
    author=SOURCE_CONFIG["author"],
    license=SOURCE_CONFIG["license"]["name"],
    license_url=SOURCE_CONFIG["license"]["url"],
    target=SOURCE_CONFIG["target"],
)


def ensure_source(
    destination: str | Path,
    *,
    url: str = MEDIA_URL,
    expected_sha256: str = SOURCE_SHA256,
) -> Path:
    """Reuse the exact source or download it atomically and verify its SHA-256."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Existing source hash mismatch: expected {expected_sha256}, got {actual}: {path}"
            )
        return path

    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = Request(
        url,
        headers={
            "User-Agent": "AutoPhotogrammetry/0.4 (https://github.com/KAFKA2306/AutoPhotogrammetry)"
        },
    )
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Downloaded source hash mismatch: expected {expected_sha256}, got {actual}"
            )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _run_recorded(
    command: Sequence[str],
    *,
    name: str,
    log_dir: Path,
    records: list[dict],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    started_at = utc_now()
    completed = subprocess.run(
        list(map(str, command)),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    finished_at = utc_now()
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    records.append(
        {
            "name": name,
            "command": list(map(str, command)),
            "started_at": started_at,
            "finished_at": finished_at,
            "return_code": completed.returncode,
            "stdout_log": stdout_path.relative_to(log_dir.parent).as_posix(),
            "stderr_log": stderr_path.relative_to(log_dir.parent).as_posix(),
        }
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def colmap_commands(
    image_dir: str | Path,
    colmap_dir: str | Path,
    *,
    executable: str = "colmap",
) -> list[tuple[str, list[str]]]:
    image_dir = str(Path(image_dir))
    colmap_dir = Path(colmap_dir)
    database = str(colmap_dir / "database.db")
    sparse = str(colmap_dir / "sparse")
    return [
        (
            "colmap-feature-extractor",
            [
                executable,
                "feature_extractor",
                "--database_path",
                database,
                "--image_path",
                image_dir,
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.use_gpu",
                "0",
                "--SiftExtraction.max_image_size",
                "1024",
                "--SiftExtraction.max_num_features",
                "4096",
            ],
        ),
        (
            "colmap-sequential-matcher",
            [
                executable,
                "sequential_matcher",
                "--database_path",
                database,
                "--SiftMatching.use_gpu",
                "0",
            ],
        ),
        (
            "colmap-mapper",
            [
                executable,
                "mapper",
                "--database_path",
                database,
                "--image_path",
                image_dir,
                "--output_path",
                sparse,
            ],
        ),
    ]


def _colmap_metrics(text: str) -> dict:
    patterns = {
        "registered_images": (r"Registered images:\s*(\d+)", int),
        "points": (r"Points:\s*(\d+)", int),
        "mean_reprojection_error_px": (
            r"Mean reprojection error:\s*([0-9.]+)\s*px",
            float,
        ),
    }
    metrics = {}
    for key, (pattern, caster) in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[key] = caster(match.group(1))
    return metrics


def run_huejotzingo(
    *,
    input_root: str | Path = "input",
    output_root: str | Path = "output",
) -> dict:
    """Run the verified Huejotzingo source through COLMAP and Nerfstudio Splatfacto."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    source_path = input_root / DATASET / "source.webm"
    dataset_output = output_root / DATASET
    if dataset_output.exists():
        shutil.rmtree(dataset_output)
    log_dir = dataset_output / "logs"
    log_dir.mkdir(parents=True)
    manifest_path = dataset_output / "manifest.json"
    records: list[dict] = []
    manifest: dict = {
        "schema_version": 1,
        "dataset": DATASET,
        "status": "running",
        "started_at": utc_now(),
        "source_revision": source_revision(),
        "source": {
            "source_registry_id": SOURCE_CONFIG["id"],
            "source_page": SOURCE_PAGE,
            "media_url": MEDIA_URL,
            "expected_sha256": SOURCE_SHA256,
        },
        "commands": records,
    }
    phase = "source"

    try:
        source_path = ensure_source(source_path)
        manifest["source"]["path"] = str(source_path)
        manifest["source"]["sha256"] = sha256_file(source_path)

        phase = "probe"
        probe = probe_video(source_path)
        write_source_manifest(
            source_path,
            SOURCE,
            probe,
            dataset_output / "source-manifest.json",
        )

        phase = "frames"
        frames_dir = dataset_output / "frames"
        frames_dir.mkdir()
        ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vf",
            "fps=1/3,scale=1024:-2",
            "-q:v",
            "2",
            str(frames_dir / "frame-%06d.jpg"),
        ]
        _run_recorded(
            ffmpeg_command,
            name="ffmpeg-frames",
            log_dir=log_dir,
            records=records,
        )
        frames = sorted(frames_dir.glob("frame-*.jpg"))
        if len(frames) != EXPECTED_FRAME_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_FRAME_COUNT} frames from the verified source, got {len(frames)}"
            )
        manifest["frames"] = {"count": len(frames), "directory": str(frames_dir)}

        phase = "selection"
        selected_dir = dataset_output / "selected"
        selection = select_video_frames(frames, selected_dir)
        if not selection["selected"]:
            raise RuntimeError("Frame selection produced no images")
        manifest["selection"] = selection

        phase = "colmap"
        colmap_dir = dataset_output / "colmap"
        (colmap_dir / "sparse").mkdir(parents=True)
        for name, command in colmap_commands(selected_dir, colmap_dir):
            _run_recorded(command, name=name, log_dir=log_dir, records=records)
        sparse_model = colmap_dir / "sparse" / "0"
        if not sparse_model.is_dir():
            raise RuntimeError(f"COLMAP did not produce the expected sparse model: {sparse_model}")
        analyzer = _run_recorded(
            ["colmap", "model_analyzer", "--path", str(sparse_model)],
            name="colmap-model-analyzer",
            log_dir=log_dir,
            records=records,
        )
        analyzer_text = (analyzer.stdout or "") + "\n" + (analyzer.stderr or "")
        manifest["colmap"] = {
            "model_path": str(sparse_model),
            "metrics": _colmap_metrics(analyzer_text),
        }

        phase = "nerfstudio-process-data"
        nerfstudio_data = dataset_output / "nerfstudio-data"
        process_command = nerfstudio_process_images_command(
            selected_dir,
            nerfstudio_data,
            extra_args=(
                "--skip-colmap",
                "--colmap-model-path",
                "../colmap/sparse/0",
            ),
        )
        _run_recorded(
            process_command,
            name="nerfstudio-process-data",
            log_dir=log_dir,
            records=records,
        )
        if not (nerfstudio_data / "transforms.json").is_file():
            raise RuntimeError("Nerfstudio did not generate transforms.json")

        phase = "splatfacto"
        splat = run_splatfacto_export(
            nerfstudio_data,
            dataset_output / "runs",
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        splat_manifest = Path(splat["manifest_path"])
        ply_path = splat_manifest.parent / splat["output"]["ply_path"]
        manifest["splatfacto"] = {
            "manifest_path": str(splat_manifest),
            "ply_path": str(ply_path),
            "ply_sha256": splat["output"]["sha256"],
            "ply_size_bytes": splat["output"]["size_bytes"],
        }
        manifest["status"] = "success"
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_phase"] = phase
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise
