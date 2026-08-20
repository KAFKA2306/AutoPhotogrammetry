from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from processing.backend_evaluation import (
    build_nerfstudio_dataset_contract,
    dataset_identity,
    write_comparison,
    write_nerfstudio_split_transforms,
)
from processing.mcmc_quality import DEFAULT_NERFSTUDIO_SOURCE, verify_research_environment
from processing.provenance import write_json
from processing.quality_sweep import (
    run_splatfacto_experiment,
    verify_gpu_runtime,
    winner_train_args,
)

WINNERS = ("default", "scale-regularized", "mcmc")
CULLING_FLAGS = {
    "cull_alpha_thresh": "--pipeline.model.cull-alpha-thresh",
    "cull_scale_thresh": "--pipeline.model.cull-scale-thresh",
    "cull_screen_size": "--pipeline.model.cull-screen-size",
    "stop_screen_size_at": "--pipeline.model.stop-screen-size-at",
}
MCMC_EFFECTIVE_CULLING_PARAMETERS = {"cull_alpha_thresh"}


def _validate_culling_parameter_for_winner(winner: str, parameter: str) -> None:
    if parameter not in CULLING_FLAGS:
        raise ValueError(f"unknown culling parameter: {parameter}")
    if winner == "mcmc" and parameter not in MCMC_EFFECTIVE_CULLING_PARAMETERS:
        raise ValueError(
            f"{parameter} is not consumed by the pinned Splatfacto MCMCStrategy; "
            "do not run a no-op culling experiment"
        )


def _format_culling_value(parameter: str, value: float | int) -> str:
    if parameter == "stop_screen_size_at":
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError("stop_screen_size_at must be an integer")
        return str(int(numeric))
    return str(float(value))


def culling_train_args(
    *,
    winner: str,
    iterations: int,
    parameter: str | None = None,
    value: float | int | None = None,
) -> tuple[str, ...]:
    """Freeze the #43 winner and optionally change exactly one effective culling field."""
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    args = list(winner_train_args(iterations=iterations, winner=winner))
    if parameter is None:
        if value is not None:
            raise ValueError("culling value requires a parameter")
        return tuple(args)
    _validate_culling_parameter_for_winner(winner, parameter)
    if value is None:
        raise ValueError("culling parameter requires a value")
    args.extend((CULLING_FLAGS[parameter], _format_culling_value(parameter, value)))
    return tuple(args)


def _prepare_dataset(
    data_dir: str | Path,
    source_video: str | Path,
    output_root: str | Path,
    *,
    holdout_count: int | None,
) -> tuple[Path, Path, dict]:
    data = Path(data_dir).expanduser().resolve()
    transforms = data / "transforms.json"
    if not transforms.is_file():
        raise ValueError(f"Nerfstudio transforms.json does not exist: {transforms}")
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source video does not exist: {source}")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=holdout_count)
    write_json(root / "evaluation-dataset.json", dataset)
    split = write_nerfstudio_split_transforms(
        transforms,
        dataset,
        root / "evaluation-transforms.json",
    )
    return root, Path(split["transforms_path"]), dataset


def run_culling_sweep(
    data_dir: str | Path,
    source_video: str | Path,
    output_root: str | Path,
    *,
    winner: str,
    parameter: str,
    values: Sequence[float],
    iterations: int = 30000,
    holdout_count: int | None = None,
    nerfstudio_source: str | Path = DEFAULT_NERFSTUDIO_SOURCE,
    timeout: float | None = None,
) -> dict:
    """Compare one effective culling parameter family after freezing the #43 winner."""
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    _validate_culling_parameter_for_winner(winner, parameter)
    unique_values = tuple(dict.fromkeys(float(value) for value in values))
    if not unique_values:
        raise ValueError("at least one culling value is required")

    runtime = verify_gpu_runtime()
    environment = verify_research_environment(nerfstudio_source)
    root, split, dataset = _prepare_dataset(
        data_dir,
        source_video,
        output_root,
        holdout_count=holdout_count,
    )
    summary = {
        "schema_version": 1,
        "experiment_type": "culling-only",
        "dataset_id": dataset_identity(dataset),
        "winner": winner,
        "iterations": iterations,
        "culling_parameter": parameter,
        "runtime": runtime,
        "environment": environment,
        "experiments": [],
    }
    results = []

    baseline_args = culling_train_args(winner=winner, iterations=iterations)
    entry, result = run_splatfacto_experiment(
        name=f"splatfacto-{winner}-culling-baseline",
        dataset=dataset,
        split_transforms=split,
        output_root=root / "baseline",
        environment=environment,
        train_args=baseline_args,
        config={
            "experiment_type": "culling-only",
            "winner": winner,
            "iterations": iterations,
            "changed_parameter": None,
        },
        timeout=timeout,
    )
    summary["experiments"].append({"changed_parameter": None, "value": None, **entry})
    results.append(result)

    for value in unique_values:
        args = culling_train_args(
            winner=winner,
            iterations=iterations,
            parameter=parameter,
            value=value,
        )
        label = str(value).replace("-", "m").replace(".", "p")
        entry, result = run_splatfacto_experiment(
            name=f"splatfacto-{winner}-{parameter}-{label}",
            dataset=dataset,
            split_transforms=split,
            output_root=root / f"{parameter}-{label}",
            environment=environment,
            train_args=args,
            config={
                "experiment_type": "culling-only",
                "winner": winner,
                "iterations": iterations,
                "changed_parameter": parameter,
                "value": value,
            },
            timeout=timeout,
        )
        summary["experiments"].append(
            {"changed_parameter": parameter, "value": value, **entry}
        )
        results.append(result)

    comparison_path = root / "comparison.json"
    summary["comparison"] = write_comparison(comparison_path, results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["all_experiments_succeeded"] = all(
        experiment["status"] == "success" for experiment in summary["experiments"]
    )
    summary_path = root / "culling-sweep.json"
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def run_budget_sweep(
    data_dir: str | Path,
    source_video: str | Path,
    output_root: str | Path,
    *,
    winner: str,
    budgets: Sequence[int],
    holdout_count: int | None = None,
    nerfstudio_source: str | Path = DEFAULT_NERFSTUDIO_SOURCE,
    timeout: float | None = None,
) -> dict:
    """Compare iteration budgets while preserving the exact #43 winner configuration."""
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    unique_budgets = tuple(dict.fromkeys(int(value) for value in budgets))
    if len(unique_budgets) < 2 or any(value <= 0 for value in unique_budgets):
        raise ValueError("budget sweep requires at least two distinct positive iteration values")

    runtime = verify_gpu_runtime()
    environment = verify_research_environment(nerfstudio_source)
    root, split, dataset = _prepare_dataset(
        data_dir,
        source_video,
        output_root,
        holdout_count=holdout_count,
    )
    summary = {
        "schema_version": 1,
        "experiment_type": "budget-only",
        "dataset_id": dataset_identity(dataset),
        "winner": winner,
        "budgets": list(unique_budgets),
        "runtime": runtime,
        "environment": environment,
        "experiments": [],
    }
    results = []

    for budget in unique_budgets:
        args = winner_train_args(iterations=budget, winner=winner)
        entry, result = run_splatfacto_experiment(
            name=f"splatfacto-{winner}-iterations-{budget}",
            dataset=dataset,
            split_transforms=split,
            output_root=root / f"iterations-{budget}",
            environment=environment,
            train_args=args,
            config={
                "experiment_type": "budget-only",
                "winner": winner,
                "iterations": budget,
            },
            timeout=timeout,
        )
        summary["experiments"].append({"iterations": budget, **entry})
        results.append(result)

    comparison_path = root / "comparison.json"
    summary["comparison"] = write_comparison(comparison_path, results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["all_experiments_succeeded"] = all(
        experiment["status"] == "success" for experiment in summary["experiments"]
    )
    summary_path = root / "budget-sweep.json"
    summary["manifest_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run #43-winner-fixed culling-only or budget-only Splatfacto follow-ups."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    culling = subparsers.add_parser("culling")
    culling.add_argument("--data", required=True)
    culling.add_argument("--source-video", required=True)
    culling.add_argument("--output-root", required=True)
    culling.add_argument("--winner", choices=WINNERS, required=True)
    culling.add_argument("--parameter", choices=tuple(CULLING_FLAGS), required=True)
    culling.add_argument("--value", action="append", type=float, required=True)
    culling.add_argument("--iterations", type=int, default=30000)
    culling.add_argument("--holdout-count", type=int)
    culling.add_argument("--nerfstudio-source", default=str(DEFAULT_NERFSTUDIO_SOURCE))
    culling.add_argument("--timeout", type=float)

    budget = subparsers.add_parser("budget")
    budget.add_argument("--data", required=True)
    budget.add_argument("--source-video", required=True)
    budget.add_argument("--output-root", required=True)
    budget.add_argument("--winner", choices=WINNERS, required=True)
    budget.add_argument("--iterations", action="append", type=int, required=True)
    budget.add_argument("--holdout-count", type=int)
    budget.add_argument("--nerfstudio-source", default=str(DEFAULT_NERFSTUDIO_SOURCE))
    budget.add_argument("--timeout", type=float)

    args = parser.parse_args()
    if args.command == "culling":
        result = run_culling_sweep(
            args.data,
            args.source_video,
            args.output_root,
            winner=args.winner,
            parameter=args.parameter,
            values=args.value,
            iterations=args.iterations,
            holdout_count=args.holdout_count,
            nerfstudio_source=args.nerfstudio_source,
            timeout=args.timeout,
        )
    else:
        result = run_budget_sweep(
            args.data,
            args.source_video,
            args.output_root,
            winner=args.winner,
            budgets=args.iterations,
            holdout_count=args.holdout_count,
            nerfstudio_source=args.nerfstudio_source,
            timeout=args.timeout,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_experiments_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
