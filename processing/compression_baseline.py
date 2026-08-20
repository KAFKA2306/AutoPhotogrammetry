from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from processing.backend_evaluation import gaussian_artifact_metrics
from processing.huejotzingo import _colmap_metrics, colmap_commands
from processing.nerfstudio import (
    nerfstudio_process_images_command,
    run_nerfstudio_eval,
    run_splatfacto_export,
)
from processing.provenance import sha256_file, write_json
from processing.quality_sweep import quality_sweep_train_args, verify_gpu_runtime
from processing.video import extract_frames_command, frame_timestamp_records, probe_video


def compression_command(
    source: str | Path,
    output: str | Path,
    *,
    crf: int,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    if not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be between 0 and 51")
    return [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(Path(source)),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(Path(output)),
    ]


def build_timestamp_split(
    source_video_sha256: str,
    frame_names: Sequence[str],
    *,
    fps: float,
    holdout_count: int | None = None,
) -> dict:
    """Freeze train/holdout by source time/name, independent of compression bytes."""
    names = list(frame_names)
    if len(names) < 2:
        raise ValueError("at least two frames are required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    count = holdout_count if holdout_count is not None else max(1, round(len(names) * 0.1))
    if count < 1 or count >= len(names):
        raise ValueError("holdout_count must be between 1 and frame_count - 1")

    records = frame_timestamp_records(names, fps=fps)
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{source_video_sha256}:{record['frame']}:{record['source_time_seconds']:.9f}".encode()
        ).hexdigest(),
    )
    holdout = {record["frame"] for record in ranked[:count]}
    annotated = [
        {**record, "split": "holdout" if record["frame"] in holdout else "train"}
        for record in records
    ]
    return {
        "schema_version": 1,
        "source_video_sha256": source_video_sha256,
        "fps": fps,
        "frames": annotated,
        "train_frame_names": [
            record["frame"] for record in annotated if record["split"] == "train"
        ],
        "holdout_frame_names": [
            record["frame"] for record in annotated if record["split"] == "holdout"
        ],
    }


def write_named_split_transforms(
    transforms_json: str | Path,
    split: Mapping,
    output_path: str | Path,
) -> dict:
    source = Path(transforms_json).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Nerfstudio transforms requires frames")
    by_name = {}
    for frame in frames:
        name = Path(str(frame.get("file_path", ""))).name
        if not name:
            raise ValueError("Nerfstudio frame is missing file_path")
        if name in by_name:
            raise ValueError(f"duplicate frame basename in transforms: {name}")
        by_name[name] = str(frame["file_path"])

    expected = {record["frame"] for record in split["frames"]}
    if set(by_name) != expected:
        raise ValueError(
            f"transforms/split frame mismatch; missing={sorted(expected - set(by_name))}, extra={sorted(set(by_name) - expected)}"
        )
    train = [by_name[name] for name in split["train_frame_names"]]
    holdout = [by_name[name] for name in split["holdout_frame_names"]]
    payload["train_filenames"] = train
    payload["val_filenames"] = holdout
    payload["test_filenames"] = holdout
    destination = Path(output_path).expanduser().resolve()
    write_json(destination, payload)
    return {
        "transforms_path": str(destination),
        "train_filenames": train,
        "holdout_filenames": holdout,
    }


def _run(
    command: Sequence[str], *, cwd: Path, stdout: Path, stderr: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(map(str, command)),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(completed.stdout or "", encoding="utf-8")
    stderr.write_text(completed.stderr or "", encoding="utf-8")
    return completed


def _require_success(completed: subprocess.CompletedProcess[str], command: Sequence[str]) -> None:
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            list(command),
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _extract_fixed_frames(
    video: Path,
    destination: Path,
    *,
    fps: float,
    width: int,
    log_root: Path,
) -> list[Path]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    command = extract_frames_command(video, destination, fps=fps, width=width)
    completed = _run(
        command,
        cwd=destination.parent,
        stdout=log_root / "ffmpeg-frames.stdout.log",
        stderr=log_root / "ffmpeg-frames.stderr.log",
    )
    _require_success(completed, command)
    frames = sorted(destination.glob("frame-*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg produced no frames")
    return frames


def _run_condition(
    name: str,
    video: Path,
    root: Path,
    split: Mapping,
    *,
    fps: float,
    width: int,
    iterations: int,
    timeout: float | None,
) -> dict:
    condition = root / name
    condition.mkdir(parents=True, exist_ok=True)
    logs = condition / "logs"
    frames = _extract_fixed_frames(
        video,
        condition / "frames",
        fps=fps,
        width=width,
        log_root=logs,
    )
    frame_names = [path.name for path in frames]
    if frame_names != [record["frame"] for record in split["frames"]]:
        raise RuntimeError(f"compression condition changed timestamp/frame cardinality: {name}")

    colmap = condition / "colmap"
    (colmap / "sparse").mkdir(parents=True)
    commands = []
    for step_name, command in colmap_commands(condition / "frames", colmap):
        completed = _run(
            command,
            cwd=condition,
            stdout=logs / f"{step_name}.stdout.log",
            stderr=logs / f"{step_name}.stderr.log",
        )
        commands.append(
            {"name": step_name, "command": command, "return_code": completed.returncode}
        )
        _require_success(completed, command)
    sparse = colmap / "sparse" / "0"
    if not sparse.is_dir():
        raise RuntimeError(f"COLMAP did not produce sparse/0 for {name}")
    analyzer_command = ["colmap", "model_analyzer", "--path", str(sparse)]
    analyzer = _run(
        analyzer_command,
        cwd=condition,
        stdout=logs / "colmap-model-analyzer.stdout.log",
        stderr=logs / "colmap-model-analyzer.stderr.log",
    )
    _require_success(analyzer, analyzer_command)
    colmap_metrics = _colmap_metrics((analyzer.stdout or "") + "\n" + (analyzer.stderr or ""))

    ns_data = condition / "nerfstudio-data"
    process_command = nerfstudio_process_images_command(
        condition / "frames",
        ns_data,
        extra_args=(
            "--skip-colmap",
            "--skip-image-processing",
            "--colmap-model-path",
            str(sparse),
        ),
    )
    process = _run(
        process_command,
        cwd=condition,
        stdout=logs / "ns-process-data.stdout.log",
        stderr=logs / "ns-process-data.stderr.log",
    )
    _require_success(process, process_command)
    transforms = ns_data / "transforms.json"
    if not transforms.is_file():
        raise RuntimeError("ns-process-data did not produce transforms.json")
    split_info = write_named_split_transforms(
        transforms,
        split,
        condition / "evaluation-transforms.json",
    )

    started = time.perf_counter()
    training = run_splatfacto_export(
        split_info["transforms_path"],
        condition / "run",
        train_extra_args=quality_sweep_train_args(iterations=iterations, variant="default"),
        timeout=timeout,
        env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )
    train_export_seconds = time.perf_counter() - started
    manifest = Path(training["manifest_path"])
    run_dir = manifest.parent
    ply = run_dir / training["output"]["ply_path"]
    config = run_dir / training["training"]["config_path"]
    evaluation = run_nerfstudio_eval(
        config,
        run_dir / "evaluation",
        timeout=timeout,
        env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )

    return {
        "name": name,
        "status": "success",
        "video": {
            "path": str(video),
            "size_bytes": video.stat().st_size,
            "sha256": sha256_file(video),
            "probe": probe_video(video),
        },
        "frame_count": len(frames),
        "frame_sha256": [sha256_file(path) for path in frames],
        "colmap": {"metrics": colmap_metrics, "model_path": str(sparse)},
        "split": split_info,
        "training_manifest_path": str(manifest),
        "evaluation_manifest_path": evaluation["manifest_path"],
        "metrics": {
            "psnr_against_condition_holdout": evaluation["metrics"].get("psnr"),
            "ssim_against_condition_holdout": evaluation["metrics"].get("ssim"),
            "lpips_against_condition_holdout": evaluation["metrics"].get("lpips"),
            "train_export_wall_clock_seconds": train_export_seconds,
            "peak_gpu_memory_bytes": (training.get("training") or {}).get("peak_gpu_memory_bytes"),
            **gaussian_artifact_metrics(ply),
        },
        "ply_path": str(ply),
        "ply_sha256": training["output"]["sha256"],
        "ply_size_bytes": training["output"]["size_bytes"],
        "commands": commands,
        "metric_semantics": (
            "PSNR/SSIM/LPIPS use the same source timestamps but each condition's own compressed "
            "holdout images as ground truth. They measure reconstruction fidelity to that condition, "
            "not absolute fidelity to the original uncompressed source."
        ),
    }


def run_compression_baseline(
    source_video: str | Path,
    output_root: str | Path,
    *,
    crfs: Sequence[int] = (18, 35),
    fps: float = 1.0 / 3.0,
    width: int = 1024,
    holdout_count: int | None = None,
    iterations: int = 30000,
    timeout: float | None = None,
) -> dict:
    """Run original + H.264 CRF conditions at identical source timestamps."""
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source video does not exist: {source}")
    if width <= 0 or fps <= 0 or iterations <= 0:
        raise ValueError("width, fps and iterations must be positive")
    unique_crfs = tuple(dict.fromkeys(int(value) for value in crfs))
    if not unique_crfs:
        raise ValueError("at least one CRF condition is required")
    for value in unique_crfs:
        if not 0 <= value <= 51:
            raise ValueError("H.264 CRF must be between 0 and 51")

    runtime = verify_gpu_runtime()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg_version = subprocess.run(
        ["ffmpeg", "-version"],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[:1]

    reference_frames = _extract_fixed_frames(
        source,
        root / "reference-frame-contract",
        fps=fps,
        width=width,
        log_root=root / "reference-frame-contract-logs",
    )
    split = build_timestamp_split(
        sha256_file(source),
        [path.name for path in reference_frames],
        fps=fps,
        holdout_count=holdout_count,
    )
    write_json(root / "timestamp-split.json", split)
    shutil.rmtree(root / "reference-frame-contract")
    shutil.rmtree(root / "reference-frame-contract-logs")

    videos: list[tuple[str, Path, dict]] = [
        ("original", source, {"kind": "original", "transcode_command": None})
    ]
    for crf in unique_crfs:
        destination = root / "encoded" / f"h264-crf-{crf}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = compression_command(source, destination, crf=crf)
        completed = _run(
            command,
            cwd=root,
            stdout=root / "encoded" / f"h264-crf-{crf}.stdout.log",
            stderr=root / "encoded" / f"h264-crf-{crf}.stderr.log",
        )
        _require_success(completed, command)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError(f"transcode produced no output: {destination}")
        videos.append(
            (
                f"h264-crf-{crf}",
                destination,
                {"kind": "h264-crf", "crf": crf, "preset": "medium", "transcode_command": command},
            )
        )

    summary = {
        "schema_version": 1,
        "status": "running",
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "probe": probe_video(source),
        },
        "ffmpeg_version": ffmpeg_version,
        "frame_contract": {
            "fps": fps,
            "width": width,
            "split_manifest": str(root / "timestamp-split.json"),
            "train_frame_names": split["train_frame_names"],
            "holdout_frame_names": split["holdout_frame_names"],
        },
        "runtime": runtime,
        "iterations": iterations,
        "conditions": [],
    }
    summary_path = root / "compression-baseline.json"
    write_json(summary_path, summary)

    for name, video, condition_config in videos:
        try:
            result = _run_condition(
                name,
                video,
                root / "conditions",
                split,
                fps=fps,
                width=width,
                iterations=iterations,
                timeout=timeout,
            )
            result["condition"] = condition_config
        except Exception as exc:
            result = {
                "name": name,
                "status": "failed",
                "condition": condition_config,
                "error": f"{type(exc).__name__}: {exc}",
            }
        summary["conditions"].append(result)
        write_json(summary_path, summary)

    summary["status"] = (
        "success"
        if all(condition["status"] == "success" for condition in summary["conditions"])
        else "failed"
    )
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure current COLMAP+Splatfacto robustness at identical timestamps under controlled H.264 compression."
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--crf", action="append", type=int, dest="crfs")
    parser.add_argument("--fps", type=float, default=1.0 / 3.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_compression_baseline(
        args.source_video,
        args.output_root,
        crfs=tuple(args.crfs) if args.crfs else (18, 35),
        fps=args.fps,
        width=args.width,
        holdout_count=args.holdout_count,
        iterations=args.iterations,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
