from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from processing.batch import (
    _sha1_file,
    _successful_manifest,
    _video_source,
    ensure_source,
    resolve_media_url,
)
from processing.colmap_evaluation import aggregate_models, model_directories, parse_model_analyzer
from processing.huejotzingo import _run_recorded, colmap_commands
from processing.image_selection import select_video_frames
from processing.nerfstudio import nerfstudio_process_images_command, run_splatfacto_export
from processing.provenance import sha256_file, utc_now, write_json, write_source_manifest
from processing.video import extract_frames_command, probe_video
from processing.video_preflight import apply_preflight_to_registry, run_video_preflight
from processing.video_sources import EVALUATION_STAGES, load_video_registry


def _expected_source_identity(
    source: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = source.get("metadata_evidence")
    metadata = evidence if isinstance(evidence, Mapping) else {}
    return {
        "sha1": resolved.get("source_sha1") or metadata.get("source_sha1"),
        "size_bytes": resolved.get("source_size_bytes") or metadata.get("source_size_bytes"),
        "sha256": source.get("sha256"),
    }


def _apply_source_identity(
    registry_path: str | Path,
    source_id: str,
    *,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = load_video_registry(registry_file)
    source = next((item for item in registry["videos"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"unknown video source: {source_id}")
    existing = source.get("sha256")
    if existing and existing != sha256:
        raise ValueError(f"{source_id}: downloaded SHA-256 drift: {existing} != {sha256}")
    source["sha256"] = sha256
    evidence = source.setdefault("metadata_evidence", {})
    evidence["downloaded_size_bytes"] = size_bytes
    evidence["sha256_verified_from_downloaded_bytes"] = True
    registry_file.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def _canonical_colmap_metrics(colmap: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registered_images": colmap["registered_images"],
        "input_images": colmap["input_images"],
        "registration_ratio": colmap["registration_ratio"],
        "largest_model_ratio": colmap["largest_model_ratio"],
        "submodel_count": colmap["submodel_count"],
        "sparse_points": colmap["sparse_points"],
        "mean_reprojection_error_px": colmap["mean_reprojection_error_px"],
        "mean_track_length": colmap["mean_track_length"],
    }


def apply_colmap_to_registry(
    registry_path: str | Path,
    source_id: str,
    colmap_result: Mapping[str, Any],
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = load_video_registry(registry_file)
    source = next((item for item in registry["videos"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"unknown video source: {source_id}")
    if source["measurements"].get("preflight") is None:
        raise ValueError(f"{source_id}: COLMAP evidence requires Stage-B evidence first")
    canonical = _canonical_colmap_metrics(colmap_result)
    if int(canonical["registered_images"]) < 1 or int(canonical["sparse_points"]) < 1:
        raise ValueError(f"{source_id}: unusable COLMAP evidence")
    source["measurements"]["colmap"] = canonical
    source["colmap_evidence"] = {
        "largest_model_path": colmap_result.get("largest_model_path"),
        "observations": colmap_result.get("observations"),
        "mean_observations_per_image": colmap_result.get("mean_observations_per_image"),
        "models": colmap_result.get("models"),
        "aggregation": colmap_result.get("aggregation"),
    }
    current_stage = source["evaluation_stage"]
    if EVALUATION_STAGES.index(current_stage) < EVALUATION_STAGES.index("colmap"):
        source["evaluation_stage"] = "colmap"
    registry_file.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def _analyze_colmap_models(
    sparse_root: Path,
    *,
    input_images: int,
    log_dir: Path,
    records: list[dict],
) -> dict[str, Any]:
    models = []
    for index, model_path in enumerate(model_directories(sparse_root)):
        completed = _run_recorded(
            ["colmap", "model_analyzer", "--path", str(model_path)],
            name=f"colmap-model-analyzer-{index:03d}",
            log_dir=log_dir,
            records=records,
        )
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        models.append(
            {
                "model_path": str(model_path),
                "metrics": parse_model_analyzer(text),
            }
        )
    return aggregate_models(models, input_images=input_images)


def run_shot_video(
    source: Mapping[str, Any],
    *,
    registry_path: str | Path = "sources/videos.json",
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    preflight_sample_fps: float = 2.0,
    preflight_width: int = 640,
    frame_interval_seconds: float = 3.0,
    frame_width: int = 1024,
    train_iterations: int = 2000,
    timeout: float | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Run one source from its measured best continuous shot through Stage C and PLY export."""
    dataset = str(source["id"])
    input_root = Path(input_root)
    dataset_output = Path(output_root) / dataset
    manifest_path = dataset_output / "manifest.json"
    if not fresh:
        cached = _successful_manifest(manifest_path)
        cached_preflight = cached.get("preflight") if isinstance(cached, Mapping) else None
        if (
            cached
            and cached.get("schema_version") == 3
            and isinstance(cached_preflight, Mapping)
            and cached_preflight.get("selected_shot_id")
        ):
            return cached
    if dataset_output.exists():
        shutil.rmtree(dataset_output)
    log_dir = dataset_output / "logs"
    log_dir.mkdir(parents=True)
    records: list[dict] = []
    manifest: dict[str, Any] = {
        "schema_version": 3,
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
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive")
        resolved = resolve_media_url(source)
        manifest["source_resolution"] = resolved
        identity = _expected_source_identity(source, resolved)
        source_path = input_root / dataset / "source.webm"
        source_path = ensure_source(
            source_path,
            url=str(resolved["media_url"]),
            expected_sha256=identity["sha256"],
            expected_sha1=identity["sha1"],
            expected_size=identity["size_bytes"],
        )
        actual_sha256 = sha256_file(source_path)
        actual_sha1 = _sha1_file(source_path)
        expected_sha1 = identity["sha1"]
        manifest["source"] = {
            "path": str(source_path),
            "sha256": actual_sha256,
            "sha1": actual_sha1,
            "expected_sha1": expected_sha1,
            "sha1_match": not expected_sha1 or actual_sha1 == expected_sha1,
            "size_bytes": source_path.stat().st_size,
        }
        _apply_source_identity(
            registry_path,
            dataset,
            sha256=actual_sha256,
            size_bytes=source_path.stat().st_size,
        )

        phase = "probe"
        probe = probe_video(source_path)
        manifest["probe"] = probe
        write_source_manifest(
            source_path,
            _video_source(source, resolved),
            probe,
            dataset_output / "source-manifest.json",
        )

        phase = "shot-preflight"
        preflight_path = dataset_output / "preflight.json"
        preflight = run_video_preflight(
            source_path,
            preflight_path,
            sample_fps=preflight_sample_fps,
            width=preflight_width,
        )
        shot_evidence = preflight["shot_evidence"]
        start_seconds = float(shot_evidence["selected_start_seconds"])
        duration_seconds = float(shot_evidence["selected_duration_seconds"])
        manifest["preflight"] = {
            "manifest_path": str(preflight_path),
            "metrics": preflight["metrics"],
            "selected_shot_id": shot_evidence["selected_shot_id"],
            "selected_start_seconds": start_seconds,
            "selected_duration_seconds": duration_seconds,
            "selected_geometry": shot_evidence["selected_geometry"],
        }
        apply_preflight_to_registry(registry_path, dataset, preflight)

        phase = "frames"
        frames_dir = dataset_output / "frames"
        frames_dir.mkdir()
        frame_command = extract_frames_command(
            source_path,
            frames_dir,
            fps=1.0 / frame_interval_seconds,
            width=frame_width,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        _run_recorded(frame_command, name="ffmpeg-selected-shot", log_dir=log_dir, records=records)
        frames = sorted(frames_dir.glob("frame-*.jpg"))
        if not frames:
            raise RuntimeError("FFmpeg produced no frames from selected shot")
        manifest["frames"] = {
            "count": len(frames),
            "directory": str(frames_dir),
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
        }

        phase = "selection"
        selection = select_video_frames(frames, dataset_output / "selected")
        if not selection["selected"]:
            raise RuntimeError("Frame selection produced no images")
        manifest["selection"] = selection

        phase = "colmap"
        colmap_dir = dataset_output / "colmap"
        sparse_root = colmap_dir / "sparse"
        sparse_root.mkdir(parents=True)
        for name, command in colmap_commands(dataset_output / "selected", colmap_dir):
            _run_recorded(command, name=name, log_dir=log_dir, records=records)
        colmap = _analyze_colmap_models(
            sparse_root,
            input_images=int(selection["selected"]),
            log_dir=log_dir,
            records=records,
        )
        if int(colmap["registered_images"]) < 1 or int(colmap["sparse_points"]) < 1:
            raise RuntimeError(f"COLMAP produced no usable reconstruction: {colmap}")
        sparse_model = Path(str(colmap["largest_model_path"]))
        canonical_model = sparse_root / "canonical"
        if canonical_model.exists():
            shutil.rmtree(canonical_model)
        shutil.copytree(sparse_model, canonical_model)
        colmap["canonical_model_path"] = str(canonical_model)
        manifest["colmap"] = colmap
        apply_colmap_to_registry(registry_path, dataset, colmap)

        phase = "nerfstudio-process-data"
        nerfstudio_data = dataset_output / "nerfstudio-data"
        process_command = nerfstudio_process_images_command(
            dataset_output / "selected",
            nerfstudio_data,
            extra_args=(
                "--skip-colmap",
                "--colmap-model-path",
                str(canonical_model.resolve()),
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

        phase = "splatfacto-export"
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
            "status": "artifact-only",
            "stage_d_complete": False,
            "note": "Stage D requires deterministic hold-out metrics from processing.quality_sweep.",
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


def run_all_shot_videos(
    *,
    registry_path: str | Path = "sources/videos.json",
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    ids: Sequence[str] | None = None,
    train_iterations: int = 2000,
    timeout: float | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    registry = load_video_registry(registry_path)
    wanted = set(ids or ())
    unknown = wanted - {video["id"] for video in registry["videos"]}
    if unknown:
        raise KeyError(f"Unknown video ids: {sorted(unknown)}")
    sources = [video for video in registry["videos"] if not wanted or video["id"] in wanted]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_path = output_root / "batch-manifest.json"
    batch: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": utc_now(),
        "registry": str(Path(registry_path)),
        "requested": len(sources),
        "results": [],
    }
    write_json(batch_path, batch)
    for source in sources:
        try:
            result = run_shot_video(
                source,
                registry_path=registry_path,
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
                    "selected_shot_id": result.get("preflight", {}).get("selected_shot_id"),
                    "registration_ratio": result.get("colmap", {}).get("registration_ratio"),
                    "ply_path": result.get("splatfacto", {}).get("ply_path"),
                    "ply_sha256": result.get("splatfacto", {}).get("ply_sha256"),
                    "stage_d_complete": False,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more registry videos from measured continuous shots."
    )
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--train-iterations", type=int, default=2000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    registry = load_video_registry(args.registry)
    ids = args.ids or [str(registry["default"])]
    result = run_all_shot_videos(
        registry_path=args.registry,
        input_root=args.input_root,
        output_root=args.output_root,
        ids=ids,
        train_iterations=args.train_iterations,
        timeout=args.timeout,
        fresh=args.fresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
