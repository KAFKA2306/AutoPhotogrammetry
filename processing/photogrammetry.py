from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

SUPPORTED_BACKENDS = ("meshroom", "visualsfm", "colmap")
ENV_VARS = {
    "meshroom": "AUTOPHOTOGRAMMETRY_MESHROOM_EXECUTABLE",
    "visualsfm": "AUTOPHOTOGRAMMETRY_VISUALSFM_EXECUTABLE",
    "colmap": "AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE",
}
DEFAULT_EXECUTABLES = {
    "meshroom": "meshroom_photogrammetry.exe"
    if platform.system() == "Windows"
    else "meshroom_photogrammetry",
    "visualsfm": "VisualSFM.exe" if platform.system() == "Windows" else "visualsfm",
    "colmap": "colmap.exe" if platform.system() == "Windows" else "colmap",
}


class BackendConfigurationError(RuntimeError):
    """Raised when an external backend is not explicitly available."""


@dataclass(frozen=True)
class BackendConfig:
    executable: str | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    backend: str
    run_id: str
    run_dir: str
    command: list[str]
    executable: str
    version: str | None
    started_at: str
    finished_at: str
    return_code: int
    stdout_log: str
    stderr_log: str
    artifacts: list[str]
    manifest_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Expected one of {SUPPORTED_BACKENDS}."
        )
    return normalized


def resolve_executable(
    backend: str,
    config: BackendConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    backend = _validate_backend(backend)
    config = config or BackendConfig()
    env = env or os.environ
    candidate = (
        config.executable
        or env.get(ENV_VARS[backend])
        or DEFAULT_EXECUTABLES[backend]
    )
    candidate_path = Path(candidate).expanduser()

    if candidate_path.is_absolute() or candidate_path.parent != Path("."):
        resolved = candidate_path.resolve()
        if resolved.is_file():
            return resolved
    else:
        found = shutil.which(str(candidate_path))
        if found:
            return Path(found).resolve()

    env_name = ENV_VARS[backend]
    raise BackendConfigurationError(
        f"{backend} executable was not found. Install it outside this application, then set "
        f"{env_name} to the executable path or pass BackendConfig(executable=...)."
    )


def build_command(
    backend: str,
    executable: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    extra_args: Sequence[str] = (),
) -> list[str]:
    backend = _validate_backend(backend)
    executable = str(Path(executable))
    image_dir = str(Path(image_dir))
    output_dir = str(Path(output_dir))

    if backend == "meshroom":
        args = [executable, "--input", image_dir, "--output", output_dir]
    elif backend == "visualsfm":
        args = [executable, image_dir, output_dir]
    else:
        args = [
            executable,
            "automatic_reconstructor",
            "--image_path",
            image_dir,
            "--workspace_path",
            output_dir,
        ]
    return [*args, *map(str, extra_args)]


def _read_version(executable: Path) -> str | None:
    for flag in ("--version", "-version", "-h"):
        try:
            completed = subprocess.run(
                [str(executable), flag],
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = (completed.stdout or completed.stderr).strip()
        if text:
            return text.splitlines()[0][:500]
    return None


def _list_artifacts(run_dir: Path, excluded: set[Path]) -> list[str]:
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path not in excluded
    )


def run_backend(
    backend: str,
    image_dir: str | Path,
    output_root: str | Path,
    config: BackendConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    backend = _validate_backend(backend)
    config = config or BackendConfig()
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise ValueError(
            f"Image directory does not exist or is not a directory: {image_dir}"
        )

    executable = resolve_executable(backend, config, env)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root).expanduser().resolve() / backend / run_id
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=False)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    manifest_path = run_dir / "manifest.json"
    command = build_command(
        backend, executable, image_dir, model_dir, config.extra_args
    )
    started_at = _utc_now()

    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout} seconds."

    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")
    finished_at = _utc_now()
    artifacts = _list_artifacts(run_dir, {stdout_log, stderr_log, manifest_path})

    manifest = {
        "schema_version": 1,
        "backend": backend,
        "run_id": run_id,
        "executable": str(executable),
        "version": _read_version(executable),
        "command": command,
        "image_dir": str(image_dir),
        "model_dir": str(model_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": return_code,
        "stdout_log": stdout_log.name,
        "stderr_log": stderr_log.name,
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = RunResult(
        backend=backend,
        run_id=run_id,
        run_dir=str(run_dir),
        command=command,
        executable=str(executable),
        version=manifest["version"],
        started_at=started_at,
        finished_at=finished_at,
        return_code=return_code,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        artifacts=artifacts,
        manifest_path=str(manifest_path),
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, command, output=stdout, stderr=stderr
        )
    return result


def run_photogrammetry(
    image_dir: str | Path,
    output_root: str | Path,
    software_list: Sequence[str],
    configs: Mapping[str, BackendConfig] | None = None,
) -> list[RunResult]:
    configs = configs or {}
    return [
        run_backend(backend, image_dir, output_root, configs.get(backend))
        for backend in software_list
    ]


def load_backend_configs(path: str | Path) -> dict[str, BackendConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(data) - set(SUPPORTED_BACKENDS))
    if unknown:
        raise ValueError(f"Unsupported backend keys: {unknown}")
    configs: dict[str, BackendConfig] = {}
    for backend, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration for {backend} must be an object.")
        extra_args = raw.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(item, str) for item in extra_args
        ):
            raise ValueError(
                f"extra_args for {backend} must be an array of strings."
            )
        executable = raw.get("executable")
        if executable is not None and not isinstance(executable, str):
            raise ValueError(
                f"executable for {backend} must be a string or null."
            )
        configs[backend] = BackendConfig(
            executable=executable,
            extra_args=tuple(extra_args),
        )
    return configs
