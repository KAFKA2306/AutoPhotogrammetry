from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.gaussian_ply import gaussian_ply_metrics
from processing.mcmc_quality import DEFAULT_NERFSTUDIO_SOURCE, verify_research_environment
from processing.nerfstudio import run_splatfacto_export
from processing.provenance import write_json


def quality_sweep_train_args(*, iterations: int, variant: str) -> tuple[str, ...]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    args = [
        "--max-num-iterations",
        str(iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    if variant == "default":
        pass
    elif variant == "scale-regularized":
        args.extend(("--pipeline.model.use-scale-regularization", "True"))
    elif variant == "mcmc":
        args.extend(("--pipeline.model.strategy", "mcmc"))
    else:
        raise ValueError(f"unknown quality sweep variant: {variant}")
    return tuple(args)


def run_quality_sweep(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    nerfstudio_source: str | Path = DEFAULT_NERFSTUDIO_SOURCE,
    iterations: int = 30000,
    timeout: float | None = None,
) -> dict:
    """Run the three first-line Splatfacto candidates and measure their exported PLYs.

    The sweep changes one refinement choice at a time on one exact Nerfstudio/gsplat
    environment: upstream default, PhysGaussian-derived scale regularization, and MCMC.
    Clean-GS is intentionally excluded because it requires semantic masks and is a
    post-processing experiment rather than a training-strategy variable.
    """
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"Nerfstudio data directory does not exist: {data}")
    environment = verify_research_environment(nerfstudio_source)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "quality-sweep.json"
    summary = {
        "schema_version": 1,
        "data_dir": str(data),
        "environment": environment,
        "iterations": iterations,
        "variants": [],
    }

    for variant in ("default", "scale-regularized", "mcmc"):
        result = run_splatfacto_export(
            data,
            root / variant,
            train_extra_args=quality_sweep_train_args(iterations=iterations, variant=variant),
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        manifest_path = Path(result["manifest_path"])
        ply_path = manifest_path.parent / result["output"]["ply_path"]
        ply_metrics = gaussian_ply_metrics(ply_path)
        summary["variants"].append(
            {
                "variant": variant,
                "manifest_path": str(manifest_path),
                "ply_path": str(ply_path),
                "ply_sha256": result["output"]["sha256"],
                "ply_size_bytes": result["output"]["size_bytes"],
                "ply_metrics": ply_metrics,
            }
        )
        write_json(summary_path, summary)

    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run default, scale-regularized and MCMC Splatfacto on one pinned GPU environment."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nerfstudio-source", default=str(DEFAULT_NERFSTUDIO_SOURCE))
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    result = run_quality_sweep(
        args.data,
        args.output_root,
        nerfstudio_source=args.nerfstudio_source,
        iterations=args.iterations,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
