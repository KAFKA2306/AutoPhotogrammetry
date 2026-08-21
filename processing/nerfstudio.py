from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from processing.provenance import image_records, sha256_file, write_json


class NerfstudioConfigurationError(RuntimeError):
    """Raised when the external Nerfstudio CLI is not available."""


def nerfstudio_process_images_command(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-process-data",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "images",
        "--data",
        str(Path(image_dir)),
        "--output-dir",
        str(Path(output_dir)),
        *map(str, extra_args),
    ]


def splatfacto_train_command(
    data_dir: str | Path,
    *,
    executable: str = "ns-train",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "splatfacto",
        "--data",
        str(Path(data_dir)),
        *map(str, extra_args),
    ]


def gaussian_splat_export_command(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "ns-export",
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        executable,
        "gaussian-splat",
        "--load-config",
        str(Path(config_path)),
        "--output-dir",
        str(Path(output_dir)),
        *map(str, extra_args),
    ]


def nerfstudio_eval_command(
    config_path: str | Path,
    output_path: str | Path,
    *,
    render_output_path: str | Path | None = None,
    executable: str = "ns-eval",
) -> list[str]:
    command = [
        executable,
        "--load-config",
        str(Path(config_path)),
        "--output-path",
        str(Path(output_path)),
    ]
    if render_output_path is not None:
        command.extend(("--render-output-path", str(Path(render_output_path))))
    return command


def _resolve_cli(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    else:
        found = shutil.which(str(candidate))
        if found:
            return Path(found).resolve()
    raise NerfstudioConfigurationError(
        f"{executable} was not found. Install Nerfstudio in the execution environment and ensure "
        "its CLI is on PATH, or pass an explicit executable path."
    )


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_recorded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None,
    env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    run_env = None if env is None else {**os.environ, **env}
    try:
        return subprocess.run(
            list(map(str, command)),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout} seconds."
        return subprocess.CompletedProcess(list(command), 124, stdout, stderr)


def _descendant_pids(root_pid: int) -> set[int]:
    """Return a best-effort Linux process-tree snapshot rooted at root_pid."""
    seen = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="utf-8").split()
        except OSError:
            continue
        for value in children:
            try:
                child = int(value)
            except ValueError:
                continue
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return seen


def _namespace_visible_pids(pids: set[int]) -> set[int]:
    """Include outer namespace PIDs so nvidia-smi output can match container processes."""
    visible = set(pids)
    for pid in list(pids):
        status_path = Path(f"/proc/{pid}/status")
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.startswith("NSpid:"):
                continue
            for value in line.split(":", 1)[1].split():
                try:
                    visible.add(int(value))
                except ValueError:
                    pass
            break
    return visible


def _query_process_gpu_memory_bytes(root_pid: int, nvidia_smi: str) -> int | None:
    """Measure framebuffer memory for the process tree when NVIDIA exposes it.

    NVIDIA documents --query-compute-apps as the selective query for active compute
    processes. Memory can be unavailable under some driver models, in which case this
    function returns None rather than inventing a value.
    """
    visible_pids = _namespace_visible_pids(_descendant_pids(root_pid))
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    total_mib = 0.0
    found = False
    for line in (result.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid not in visible_pids or fields[1].upper() in {"N/A", "[N/A]"}:
            continue
        try:
            total_mib += float(fields[1])
        except ValueError:
            continue
        found = True
    if not found:
        return None
    return int(total_mib * 1024 * 1024)


def _run_recorded_command_with_peak_gpu_memory(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None,
    env: Mapping[str, str] | None,
    poll_seconds: float = 0.2,
) -> tuple[subprocess.CompletedProcess[str], int | None]:
    """Run a command and best-effort sample peak process-tree GPU memory.

    If nvidia-smi is unavailable, preserve the normal subprocess.run path and return
    an unmeasured peak (None). This keeps CPU-only CI deterministic.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return (
            _run_recorded_command(command, cwd=cwd, timeout=timeout, env=env),
            None,
        )

    argv = list(map(str, command))
    run_env = None if env is None else {**os.environ, **env}
    process = subprocess.Popen(
        argv,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=run_env,
    )
    stop = threading.Event()
    peak_gpu_memory_bytes: int | None = None

    def monitor() -> None:
        nonlocal peak_gpu_memory_bytes
        while not stop.is_set():
            measured = _query_process_gpu_memory_bytes(process.pid, nvidia_smi)
            if measured is not None:
                peak_gpu_memory_bytes = max(peak_gpu_memory_bytes or 0, measured)
            if stop.wait(poll_seconds):
                break

    thread = threading.Thread(target=monitor, name="gpu-memory-monitor", daemon=True)
    thread.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        thread.join(timeout=max(1.0, poll_seconds * 4))

    return_code = 124 if timed_out else process.returncode
    if timed_out:
        stderr = (stderr or "") + f"\nTimed out after {timeout} seconds."
    return (
        subprocess.CompletedProcess(argv, return_code, stdout or "", stderr or ""),
        peak_gpu_memory_bytes,
    )


def _nerfstudio_input_images(data: Path) -> list[dict]:
    if data.is_dir():
        return image_records(data)
    if not data.is_file() or data.suffix.lower() != ".json":
        raise ValueError(f"Nerfstudio data must be a directory or JSON file: {data}")
    meta = json.loads(data.read_text(encoding="utf-8"))
    records = []
    for frame in meta.get("frames") or []:
        declared = frame.get("file_path")
        if not declared:
            raise ValueError("Nerfstudio frame is missing file_path")
        candidate = Path(str(declared))
        resolved = candidate if candidate.is_absolute() else data.parent / candidate
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise ValueError(f"Nerfstudio frame does not exist: {resolved}")
        records.append(
            {
                "path": str(declared),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if not records:
        raise ValueError(f"Nerfstudio JSON contains no image frames: {data}")
    return records


def run_nerfstudio_eval(
    config_path: str | Path,
    output_root: str | Path,
    *,
    executable: str = "ns-eval",
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run ns-eval, save per-holdout renders, and return auditable image metrics."""
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"Nerfstudio config does not exist: {config}")
    eval_cli = _resolve_cli(executable)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.json"
    renders_path = root / "renders"
    stdout_path = root / "eval.stdout.log"
    stderr_path = root / "eval.stderr.log"
    manifest_path = root / "eval-manifest.json"
    command = nerfstudio_eval_command(
        config,
        metrics_path,
        render_output_path=renders_path,
        executable=str(eval_cli),
    )
    # ns-train records ``trainer.load_dir`` relative to the training run root,
    # while ns-eval is otherwise launched from the evaluation output directory.
    # Reuse the run root when a sibling training manifest identifies it so the
    # recorded checkpoint path resolves identically for training and evaluation.
    evaluation_cwd = root
    for parent in (config.parent, *config.parents):
        if (parent / "manifest.json").is_file():
            evaluation_cwd = parent
            break
    started_at = _utc_now()
    completed = _run_recorded_command(
        command,
        cwd=evaluation_cwd,
        timeout=timeout,
        env=env,
    )
    finished_at = _utc_now()
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    record = {
        "schema_version": 2,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": completed.returncode,
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
        "metrics_path": metrics_path.name,
        "render_output_path": renders_path.name,
        "metrics": None,
        "renders": None,
    }
    if completed.returncode != 0:
        record["status"] = "failed"
        write_json(manifest_path, record)
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if not metrics_path.is_file():
        record["status"] = "failed"
        write_json(manifest_path, record)
        raise RuntimeError("ns-eval succeeded but did not write metrics.json")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        record["status"] = "failed"
        write_json(manifest_path, record)
        raise RuntimeError("ns-eval metrics.json does not contain a results object")
    measured = {}
    for key in ("psnr", "ssim", "lpips"):
        value = results.get(key)
        if value is not None:
            if not isinstance(value, (int, float)):
                raise RuntimeError(f"ns-eval result {key} is not numeric: {value!r}")
            measured[key] = float(value)

    renders = image_records(renders_path) if renders_path.is_dir() else []
    if not renders:
        record["status"] = "failed"
        record["metrics"] = measured
        record["raw_results"] = results
        write_json(manifest_path, record)
        raise RuntimeError("ns-eval succeeded but did not create any hold-out render images")

    record["status"] = "success"
    record["metrics"] = measured
    record["raw_results"] = results
    record["renders"] = renders
    record["render_count"] = len(renders)
    write_json(manifest_path, record)
    record["manifest_path"] = str(manifest_path)
    return record


def run_splatfacto_export(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    train_executable: str = "ns-train",
    export_executable: str = "ns-export",
    train_extra_args: Sequence[str] = (),
    export_extra_args: Sequence[str] = (),
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Run external Nerfstudio Splatfacto training and export one auditable PLY."""
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir() and not (data.is_file() and data.suffix.lower() == ".json"):
        raise ValueError(f"Nerfstudio data directory/JSON does not exist: {data}")

    train_cli = _resolve_cli(train_executable)
    export_cli = _resolve_cli(export_executable)
    nerfstudio_version = _package_version("nerfstudio")
    gsplat_version = _package_version("gsplat")
    if nerfstudio_version is None or gsplat_version is None:
        missing = [
            name
            for name, version in (
                ("nerfstudio", nerfstudio_version),
                ("gsplat", gsplat_version),
            )
            if version is None
        ]
        raise NerfstudioConfigurationError(
            "Installed package version could not be resolved for: " + ", ".join(missing)
        )

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).expanduser().resolve() / "splatfacto" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    train_stdout = run_dir / "train.stdout.log"
    train_stderr = run_dir / "train.stderr.log"
    export_stdout = run_dir / "export.stdout.log"
    export_stderr = run_dir / "export.stderr.log"
    input_images = _nerfstudio_input_images(data)

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "running",
        "input": {
            "data_dir": str(data),
            "image_count": len(input_images),
            "images": input_images,
        },
        "versions": {
            "nerfstudio": nerfstudio_version,
            "gsplat": gsplat_version,
        },
        "training": None,
        "export": None,
        "output": None,
    }

    train_command = splatfacto_train_command(
        data,
        executable=str(train_cli),
        extra_args=train_extra_args,
    )
    train_started = _utc_now()
    train_result, peak_gpu_memory_bytes = _run_recorded_command_with_peak_gpu_memory(
        train_command,
        cwd=run_dir,
        timeout=timeout,
        env=env,
    )
    train_finished = _utc_now()
    train_stdout.write_text(train_result.stdout or "", encoding="utf-8")
    train_stderr.write_text(train_result.stderr or "", encoding="utf-8")
    manifest["training"] = {
        "command": train_command,
        "started_at": train_started,
        "finished_at": train_finished,
        "return_code": train_result.returncode,
        "stdout_log": train_stdout.name,
        "stderr_log": train_stderr.name,
        "config_path": None,
        "checkpoint_path": None,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "gpu_memory_measurement": (
            "nvidia-smi --query-compute-apps process-tree sampling"
            if peak_gpu_memory_bytes is not None
            else None
        ),
    }
    if train_result.returncode != 0:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "training"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(
            train_result.returncode,
            train_command,
            output=train_result.stdout,
            stderr=train_result.stderr,
        )

    configs = sorted({*run_dir.rglob("config.yml"), *run_dir.rglob("config.yaml")})
    if len(configs) != 1:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "config_discovery"
        manifest["config_candidates"] = [str(path.relative_to(run_dir)) for path in configs]
        write_json(manifest_path, manifest)
        raise RuntimeError(f"Expected exactly one Nerfstudio config, found {len(configs)}")
    config_path = configs[0]

    checkpoints = sorted(
        run_dir.rglob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )
    if not checkpoints:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "checkpoint_discovery"
        write_json(manifest_path, manifest)
        raise RuntimeError("Nerfstudio training succeeded but no checkpoint was found")
    checkpoint_path = checkpoints[-1]
    manifest["training"]["config_path"] = str(config_path.relative_to(run_dir))
    manifest["training"]["checkpoint_path"] = str(checkpoint_path.relative_to(run_dir))

    export_dir = run_dir / "export"
    export_dir.mkdir()
    export_command = gaussian_splat_export_command(
        config_path,
        export_dir,
        executable=str(export_cli),
        extra_args=export_extra_args,
    )
    export_started = _utc_now()
    export_result = _run_recorded_command(
        export_command,
        cwd=run_dir,
        timeout=timeout,
        env=env,
    )
    export_finished = _utc_now()
    export_stdout.write_text(export_result.stdout or "", encoding="utf-8")
    export_stderr.write_text(export_result.stderr or "", encoding="utf-8")
    manifest["export"] = {
        "command": export_command,
        "started_at": export_started,
        "finished_at": export_finished,
        "return_code": export_result.returncode,
        "stdout_log": export_stdout.name,
        "stderr_log": export_stderr.name,
    }
    if export_result.returncode != 0:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "export"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(
            export_result.returncode,
            export_command,
            output=export_result.stdout,
            stderr=export_result.stderr,
        )

    ply_files = sorted(export_dir.rglob("*.ply"))
    if len(ply_files) != 1:
        manifest["status"] = "failed"
        manifest["failed_phase"] = "ply_discovery"
        manifest["ply_candidates"] = [str(path.relative_to(run_dir)) for path in ply_files]
        write_json(manifest_path, manifest)
        raise RuntimeError(f"Expected exactly one exported PLY, found {len(ply_files)}")

    ply_path = ply_files[0]
    manifest["status"] = "success"
    manifest["output"] = {
        "ply_path": str(ply_path.relative_to(run_dir)),
        "size_bytes": ply_path.stat().st_size,
        "sha256": sha256_file(ply_path),
    }
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
