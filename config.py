from __future__ import annotations

import json
from pathlib import Path

from photogrammetry import BackendConfig, SUPPORTED_BACKENDS


def load_backend_configs(path: str | Path) -> dict[str, BackendConfig]:
    """Load optional executable paths and argument arrays from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(data) - set(SUPPORTED_BACKENDS))
    if unknown:
        raise ValueError(f"Unsupported backend keys: {unknown}")
    configs: dict[str, BackendConfig] = {}
    for backend, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration for {backend} must be an object.")
        extra_args = raw.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError(f"extra_args for {backend} must be an array of strings.")
        executable = raw.get("executable")
        if executable is not None and not isinstance(executable, str):
            raise ValueError(f"executable for {backend} must be a string or null.")
        configs[backend] = BackendConfig(executable=executable, extra_args=tuple(extra_args))
    return configs
