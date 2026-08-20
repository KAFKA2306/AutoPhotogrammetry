from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from processing.huejotzingo import _colmap_metrics, _run_recorded, colmap_commands
from processing.image_selection import select_video_frames
from processing.nerfstudio import nerfstudio_process_images_command, run_splatfacto_export
from processing.provenance import (
    VideoSource,
    sha256_file,
    utc_now,
    write_json,
    write_source_manifest,
)
from processing.video import extract_frames_command, probe_video
from processing.video_sources import load_video_registry

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


def _file_title(source_page: str) -> str:
    path = urlparse(source_page).path
    if "/wiki/" not in path:
        raise ValueError(f"Not a Wikimedia Commons file page: {source_page}")
    title = unquote(path.split("/wiki/", 1)[1]).replace("_", " ")
    if title.startswith("File:"):
        title = title[5:]
    return title


def resolve_media_url(source: Mapping) -> dict:
    """Resolve a registry source page to the original downloadable video."""
    if source.get("media_url"):
        return {
            "media_url": source["media_url"],
            "source_sha1": None,
            "source_size_bytes": None,
            "resolved_via": "registry",
        }

    title = _file_title(str(source["source_page"]))
    query = (
        f"{WIKIMEDIA_API}?action=query&format=json&formatversion=2"
        f"&prop=imageinfo&iiprop=url%7Csize%7Cmime%7Csha1%7Cextmetadata"
        f"&titles={quote('File:' + title, safe='')}"
    )
    request = Request(query, headers={"User-Agent": "AutoPhotogrammetry/0.6"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code == 429 and attempt == 3:
                # Commons can temporarily rate-limit the metadata API while its
                # public Special:Redirect/file endpoint remains available.
                return {
                    "media_url": (
                        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
                        + quote(title, safe="")
                    ),
                    "source_sha1": None,
                    "source_size_bytes": None,
                    "mime": "video/webm",
                    "author": None,
                    "license": None,
                    "license_url": None,
                    "resolved_via": "wikimedia-direct-redirect-after-429",
                }
            if exc.code != 429 or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = (
                    max(2.0, min(30.0, float(retry_after))) if retry_after else 2.0 ** (attempt + 1)
                )
            except (TypeError, ValueError):
                delay = 2.0 ** (attempt + 1)
            time.sleep(delay)
    else:
        raise RuntimeError(f"Wikimedia API did not resolve source: {title}")

    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or not pages[0].get("imageinfo"):
        raise RuntimeError(f"Wikimedia file metadata was not found: {title}")
    info = pages[0]["imageinfo"][0]
    media_url = info.get("url")
    if not media_url:
        raise RuntimeError(f"Wikimedia file has no downloadable URL: {title}")
    metadata = info.get("extmetadata", {})
    return {
        "media_url": media_url,
        "source_sha1": info.get("sha1"),
        "source_size_bytes": info.get("size"),
        "mime": info.get("mime"),
        "author": (metadata.get("Artist") or {}).get("value"),
        "license": (metadata.get("LicenseShortName") or {}).get("value"),
        "license_url": (metadata.get("LicenseUrl") or {}).get("value"),
        "resolved_via": "wikimedia-api",
    }


def _sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_download(request: Request, *, timeout: float = 120):
    for attempt in range(3):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 2:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(5.0, min(600.0, float(retry_after))) if retry_after else 60.0
            except (TypeError, ValueError):
                delay = 60.0
            time.sleep(delay)
    raise RuntimeError("Download retry loop unexpectedly ended")


def ensure_source(
    destination: str | Path,
    *,
    url: str,
    expected_sha256: str | None = None,
    expected_sha1: str | None = None,
    expected_size: int | None = None,
) -> Path:
    """Download atomically and verify the hashes supplied by the source API."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    def verify(candidate: Path) -> None:
        if expected_sha256:
            actual = sha256_file(candidate)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"Existing source hash mismatch: expected {expected_sha256}, got {actual}: {path}"
                )
        if expected_sha1:
            actual_sha1 = _sha1_file(candidate)
            if actual_sha1 != expected_sha1:
                raise RuntimeError(f"Source SHA-1 mismatch: expected {expected_sha1}: {path}")

    if path.exists():
        verify(path)
        return path

    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        download_size = expected_size
        range_supported = False
        # Commons throttles repeated Range requests more aggressively than a
        # single streamed response. Known API sizes therefore use the stable
        # sequential stream; Range is reserved for direct-redirect fallback.
        if download_size is None:
            probe = Request(
                url,
                headers={
                    "Range": "bytes=0-0",
                    "User-Agent": "AutoPhotogrammetry/0.6",
                },
            )
            with _open_download(probe) as response:
                range_supported = getattr(response, "status", None) == 206
                if download_size is None and range_supported:
                    content_range = response.headers.get("Content-Range", "")
                    total = content_range.rsplit("/", 1)[-1]
                    if total.isdigit():
                        download_size = int(total)
                response.read(1)
            if range_supported and download_size:
                partial.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("wb") as handle:
                    handle.truncate(download_size)
                chunk_size = 64 * 1024 * 1024
                ranges = [
                    (start, min(download_size - 1, start + chunk_size - 1))
                    for start in range(0, download_size, chunk_size)
                ]

                def download_range(byte_range: tuple[int, int]) -> None:
                    start, end = byte_range
                    cursor = start
                    with partial.open("r+b") as handle:
                        handle.seek(start)
                        while cursor <= end:
                            request = Request(
                                url,
                                headers={
                                    "Range": f"bytes={cursor}-{end}",
                                    "User-Agent": "AutoPhotogrammetry/0.6",
                                },
                            )
                            with _open_download(request) as response:
                                if getattr(response, "status", None) != 206:
                                    raise RuntimeError(
                                        f"Range download was not honored for bytes {cursor}-{end}"
                                    )
                                received = 0
                                while cursor + received <= end:
                                    chunk = response.read(
                                        min(1024 * 1024, end - cursor - received + 1)
                                    )
                                    if not chunk:
                                        break
                                    handle.write(chunk)
                                    received += len(chunk)
                            if received == 0:
                                raise RuntimeError(f"Empty range download for bytes {cursor}-{end}")
                            cursor += received

                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(download_range, ranges))
            else:
                request = Request(url, headers={"User-Agent": "AutoPhotogrammetry/0.6"})
                with _open_download(request) as response, partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
        else:
            request = Request(url, headers={"User-Agent": "AutoPhotogrammetry/0.6"})
            with _open_download(request) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        verify(partial)
        if download_size and partial.stat().st_size != download_size:
            raise RuntimeError(
                f"Source size mismatch: expected {download_size}, got {partial.stat().st_size}: {path}"
            )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _video_source(source: Mapping, resolved: Mapping) -> VideoSource:
    license_info = source.get("license") or {}
    return VideoSource(
        title=str(source.get("title", source["id"])),
        source_page=str(source.get("source_page", "")),
        media_url=str(resolved["media_url"]),
        author=str(source.get("author") or resolved.get("author") or "Unknown"),
        license=str(license_info.get("name") or resolved.get("license") or "Unverified"),
        license_url=str(license_info.get("url") or resolved.get("license_url") or ""),
        target=str(source.get("target") or source.get("title") or source["id"]),
    )


def _successful_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        output = manifest["splatfacto"]
        ply_path = Path(output["ply_path"])
        if not ply_path.is_file() or ply_path.stat().st_size == 0:
            return None
        if output.get("ply_sha256") and sha256_file(ply_path) != output["ply_sha256"]:
            return None
        if manifest.get("status") != "success":
            return None
        manifest["skipped"] = True
        return manifest
    except (KeyError, OSError, json.JSONDecodeError):
        return None


def run_video(
    source: Mapping,
    *,
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    frame_interval_seconds: float = 3.0,
    frame_width: int = 1024,
    train_iterations: int = 2000,
    timeout: float | None = None,
    fresh: bool = False,
) -> dict:
    """Run one registry video through COLMAP and Gaussian Splatting."""
    dataset = str(source["id"])
    input_root = Path(input_root)
    dataset_output = Path(output_root) / dataset
    manifest_path = dataset_output / "manifest.json"
    if not fresh:
        cached = _successful_manifest(manifest_path)
        if cached:
            return cached
    if dataset_output.exists():
        shutil.rmtree(dataset_output)
    log_dir = dataset_output / "logs"
    log_dir.mkdir(parents=True)
    records: list[dict] = []
    manifest: dict = {
        "schema_version": 2,
        "dataset": dataset,
        "status": "running",
        "started_at": utc_now(),
        "registry": dict(source),
        "commands": records,
    }
    phase = "resolve-source"
    try:
        if train_iterations <= 0:
            raise ValueError("train_iterations must be positive")
        resolved = resolve_media_url(source)
        manifest["source_resolution"] = resolved
        source_path = input_root / dataset / "source.webm"
        try:
            source_path = ensure_source(
                source_path,
                url=resolved["media_url"],
                expected_sha256=source.get("sha256"),
                expected_sha1=resolved.get("source_sha1"),
                expected_size=resolved.get("source_size_bytes"),
            )
        except RuntimeError as exc:
            if "Source SHA-1 mismatch" not in str(exc):
                raise
            manifest["source_resolution"]["sha1_retry"] = True
            source_path = ensure_source(
                source_path,
                url=resolved["media_url"],
                expected_sha256=source.get("sha256"),
                expected_sha1=resolved.get("source_sha1"),
                expected_size=resolved.get("source_size_bytes"),
            )
        actual_source_sha1 = _sha1_file(source_path)
        expected_source_sha1 = resolved.get("source_sha1")
        manifest["source"] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "sha1": actual_source_sha1,
            "expected_sha1": expected_source_sha1,
            "sha1_match": not expected_source_sha1 or actual_source_sha1 == expected_source_sha1,
        }

        phase = "probe"
        probe = probe_video(source_path)
        manifest["probe"] = probe
        write_source_manifest(
            source_path,
            _video_source(source, resolved),
            probe,
            dataset_output / "source-manifest.json",
        )

        phase = "frames"
        frames_dir = dataset_output / "frames"
        frames_dir.mkdir()
        command = extract_frames_command(
            source_path,
            frames_dir,
            fps=1.0 / frame_interval_seconds,
            width=frame_width,
        )
        _run_recorded(command, name="ffmpeg-frames", log_dir=log_dir, records=records)
        frames = sorted(frames_dir.glob("frame-*.jpg"))
        if not frames:
            raise RuntimeError("FFmpeg produced no frames")
        manifest["frames"] = {"count": len(frames), "directory": str(frames_dir)}

        phase = "selection"
        selection = select_video_frames(frames, dataset_output / "selected")
        if not selection["selected"]:
            raise RuntimeError("Frame selection produced no images")
        manifest["selection"] = selection

        phase = "colmap"
        colmap_dir = dataset_output / "colmap"
        (colmap_dir / "sparse").mkdir(parents=True)
        for name, colmap_command in colmap_commands(dataset_output / "selected", colmap_dir):
            _run_recorded(colmap_command, name=name, log_dir=log_dir, records=records)
        sparse_model = colmap_dir / "sparse" / "0"
        if not sparse_model.is_dir():
            raise RuntimeError(f"COLMAP did not produce sparse model: {sparse_model}")
        analyzer = _run_recorded(
            ["colmap", "model_analyzer", "--path", str(sparse_model)],
            name="colmap-model-analyzer",
            log_dir=log_dir,
            records=records,
        )
        metrics = _colmap_metrics((analyzer.stdout or "") + "\n" + (analyzer.stderr or ""))
        if metrics.get("registered_images", 0) < 1 or metrics.get("points", 0) < 1:
            raise RuntimeError(f"COLMAP produced no usable reconstruction: {metrics}")
        manifest["colmap"] = {"model_path": str(sparse_model), "metrics": metrics}

        phase = "nerfstudio-process-data"
        nerfstudio_data = dataset_output / "nerfstudio-data"
        process_command = nerfstudio_process_images_command(
            dataset_output / "selected",
            nerfstudio_data,
            extra_args=(
                "--skip-colmap",
                "--colmap-model-path",
                str(sparse_model.resolve()),
            ),
        )
        _run_recorded(
            process_command, name="nerfstudio-process-data", log_dir=log_dir, records=records
        )
        if not (nerfstudio_data / "transforms.json").is_file():
            raise RuntimeError("Nerfstudio did not generate transforms.json")

        phase = "splatfacto"
        splat = run_splatfacto_export(
            nerfstudio_data,
            dataset_output / "runs",
            train_extra_args=(
                "--max-num-iterations",
                str(train_iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ),
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        splat_manifest = Path(splat["manifest_path"])
        ply_path = splat_manifest.parent / splat["output"]["ply_path"]
        if not ply_path.is_file() or ply_path.stat().st_size == 0:
            raise RuntimeError(f"Gaussian Splat export did not produce a non-empty PLY: {ply_path}")
        manifest["splatfacto"] = {
            "manifest_path": str(splat_manifest),
            "ply_path": str(ply_path),
            "ply_sha256": sha256_file(ply_path),
            "ply_size_bytes": ply_path.stat().st_size,
        }
        manifest["status"] = "success"
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
    except Exception as exc:
        manifest.update(
            status="failed",
            failed_phase=phase,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=utc_now(),
        )
        write_json(manifest_path, manifest)
        raise


def run_all_videos(
    *,
    registry_path: str | Path = "sources/videos.json",
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    ids: Sequence[str] | None = None,
    train_iterations: int = 2000,
    timeout: float | None = None,
    fresh: bool = False,
) -> dict:
    registry = load_video_registry(registry_path)
    wanted = set(ids or ())
    unknown = wanted - {video["id"] for video in registry["videos"]}
    if unknown:
        raise KeyError(f"Unknown video ids: {sorted(unknown)}")
    sources = [video for video in registry["videos"] if not wanted or video["id"] in wanted]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_path = output_root / "batch-manifest.json"
    batch = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "registry": str(Path(registry_path)),
        "requested": len(sources),
        "results": [],
    }
    write_json(batch_path, batch)
    for source in sources:
        try:
            result = run_video(
                source,
                input_root=input_root,
                output_root=output_root,
                train_iterations=train_iterations,
                timeout=timeout,
                fresh=fresh,
            )
            batch["results"].append(
                {
                    "id": source["id"],
                    "status": result["status"],
                    "manifest_path": result.get(
                        "manifest_path", str(output_root / source["id"] / "manifest.json")
                    ),
                    "ply_path": result.get("splatfacto", {}).get("ply_path"),
                    "ply_sha256": result.get("splatfacto", {}).get("ply_sha256"),
                }
            )
        except Exception as exc:
            manifest_path = output_root / source["id"] / "manifest.json"
            batch["results"].append(
                {
                    "id": source["id"],
                    "status": "failed",
                    "manifest_path": str(manifest_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        batch["succeeded"] = sum(item["status"] == "success" for item in batch["results"])
        batch["failed"] = sum(item["status"] == "failed" for item in batch["results"])
        write_json(batch_path, batch)
    batch["status"] = "success" if batch["succeeded"] == batch["requested"] else "failed"
    batch["finished_at"] = utc_now()
    batch["manifest_path"] = str(batch_path)
    write_json(batch_path, batch)
    return batch
