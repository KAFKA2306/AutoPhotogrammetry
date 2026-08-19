from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from processing.nerfstudio import run_splatfacto_export
from processing.provenance import write_json


def quality_train_args(*, iterations: int, scale_regularization: bool) -> tuple[str, ...]:
    """Build the one-variable Splatfacto training arguments used for quality A/B runs."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    args = [
        "--max-num-iterations",
        str(iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    if scale_regularization:
        args.extend(("--pipeline.model.use-scale-regularization", "True"))
    return tuple(args)


def run_quality_comparison(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    iterations: Sequence[int] = (2000,),
    timeout: float | None = None,
) -> dict:
    """Run baseline and scale-regularized Splatfacto with the same processed dataset.

    Each child run keeps the exact Nerfstudio/gsplat versions, argv, checkpoint and PLY hash
    in the existing Splatfacto manifest. This function only coordinates the controlled A/B
    matrix and never changes culling thresholds or silently substitutes another backend.
    """
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"Nerfstudio data directory does not exist: {data}")
    budgets = tuple(dict.fromkeys(int(value) for value in iterations))
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("iterations must contain one or more positive values")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "quality-comparison.json"
    summary = {
        "schema_version": 1,
        "data_dir": str(data),
        "experiments": [],
    }

    for budget in budgets:
        for scale_regularization in (False, True):
            variant = "scale-regularized" if scale_regularization else "baseline"
            result = run_splatfacto_export(
                data,
                root / f"iterations-{budget}" / variant,
                train_extra_args=quality_train_args(
                    iterations=budget,
                    scale_regularization=scale_regularization,
                ),
                timeout=timeout,
                env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
            )
            manifest_path = Path(result["manifest_path"])
            summary["experiments"].append(
                {
                    "iterations": budget,
                    "scale_regularization": scale_regularization,
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
        description="Run controlled Splatfacto baseline/scale-regularization quality comparisons."
    )
    parser.add_argument("--data", required=True, help="Existing Nerfstudio processed dataset.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iterations", action="append", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    result = run_quality_comparison(
        args.data,
        args.output_root,
        iterations=args.iterations,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
