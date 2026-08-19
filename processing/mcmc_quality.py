from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata
from pathlib import Path

from processing.nerfstudio import run_splatfacto_export
from processing.provenance import write_json

NERFSTUDIO_MCMC_REVISION = "c7bd9539728515eded9cc4ed137ca703f900e28a"
GSPLAT_MCMC_VERSION = "1.4.0"


def verify_research_environment(nerfstudio_source: str | Path) -> dict:
    """Verify the exact unreleased Nerfstudio commit that introduced the MCMC strategy."""
    source = Path(nerfstudio_source).expanduser().resolve()
    if not (source / ".git").exists():
        raise ValueError(f"nerfstudio_source must be a Git checkout: {source}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != NERFSTUDIO_MCMC_REVISION:
        raise ValueError(
            "Nerfstudio research checkout revision mismatch: "
            f"expected {NERFSTUDIO_MCMC_REVISION}, got {revision}"
        )
    try:
        gsplat_version = metadata.version("gsplat")
    except metadata.PackageNotFoundError as exc:
        raise ValueError("gsplat is not installed in the research environment") from exc
    if gsplat_version != GSPLAT_MCMC_VERSION:
        raise ValueError(
            f"gsplat version mismatch: expected {GSPLAT_MCMC_VERSION}, got {gsplat_version}"
        )
    return {
        "nerfstudio_repository": "https://github.com/nerfstudio-project/nerfstudio",
        "nerfstudio_revision": revision,
        "gsplat_repository": "https://github.com/nerfstudio-project/gsplat",
        "gsplat_version": gsplat_version,
    }


def mcmc_train_args(*, iterations: int, enabled: bool) -> tuple[str, ...]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    args = [
        "--max-num-iterations",
        str(iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    if enabled:
        args.extend(("--pipeline.model.strategy", "mcmc"))
    return tuple(args)


def run_mcmc_comparison(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    nerfstudio_source: str | Path,
    iterations: int = 30000,
    train_executable: str = "ns-train",
    export_executable: str = "ns-export",
    timeout: float | None = None,
) -> dict:
    """Compare DefaultStrategy vs MCMCStrategy without using the confounded splatfacto-mcmc preset."""
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"Nerfstudio data directory does not exist: {data}")
    environment = verify_research_environment(nerfstudio_source)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "mcmc-comparison.json"
    summary = {
        "schema_version": 1,
        "data_dir": str(data),
        "research_environment": environment,
        "iterations": iterations,
        "experiments": [],
    }

    for enabled in (False, True):
        variant = "mcmc" if enabled else "default"
        result = run_splatfacto_export(
            data,
            root / variant,
            train_executable=train_executable,
            export_executable=export_executable,
            train_extra_args=mcmc_train_args(iterations=iterations, enabled=enabled),
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        manifest_path = Path(result["manifest_path"])
        summary["experiments"].append(
            {
                "strategy": variant,
                "manifest_path": str(manifest_path),
                "ply_path": str(manifest_path.parent / result["output"]["ply_path"]),
                "ply_sha256": result["output"]["sha256"],
                "ply_size_bytes": result["output"]["size_bytes"],
            }
        )
        write_json(summary_path, summary)

    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Nerfstudio DefaultStrategy and MCMCStrategy at a pinned upstream commit."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nerfstudio-source", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--ns-train", default="ns-train")
    parser.add_argument("--ns-export", default="ns-export")
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_mcmc_comparison(
        args.data,
        args.output_root,
        nerfstudio_source=args.nerfstudio_source,
        iterations=args.iterations,
        train_executable=args.ns_train,
        export_executable=args.ns_export,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
