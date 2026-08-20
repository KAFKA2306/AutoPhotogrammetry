from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from processing.backend_evaluation import (
    artifact_record,
    build_nerfstudio_dataset_contract,
    dataset_identity,
    empty_metrics,
    gaussian_artifact_metrics,
    validate_backend_result,
    write_comparison,
    write_nerfstudio_split_transforms,
)
from processing.mcmc_quality import DEFAULT_NERFSTUDIO_SOURCE, verify_research_environment
from processing.nerfstudio import run_nerfstudio_eval, run_splatfacto_export
from processing.provenance import write_json
from processing.quality_sweep import quality_sweep_train_args, verify_gpu_runtime

WINNERS = ("default", "scale-regularized", "mcmc")
CULLING_FLAGS = {
    "cull_alpha_thresh": "--pipeline.model.cull-alpha-thresh",
    "cull_scale_thresh": "--pipeline.model.cull-scale-thresh",
    "cull_screen_size": "--pipeline.model.cull-screen-size",
    "stop_screen_size_at": "--pipeline.model.stop-screen-size-at",
}
MCMC_EFFECTIVE_CULLING_PARAMETERS = {"cull_alpha_thresh"}


def winner_train_args(*, winner: str, iterations: int) -> tuple[str, ...]:
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    return quality_sweep_train_args(iterations=iterations, variant=winner)


def _validate_culling_parameter(winner: str, parameter: str) -> None:
    if parameter not in CULLING_FLAGS:
        raise ValueError(f"unknown culling parameter: {parameter}")
    if winner == "mcmc" and parameter not in MCMC_EFFECTIVE_CULLING_PARAMETERS:
        raise ValueError(
            f"{parameter} is not consumed by the pinned Splatfacto MCMCStrategy; "
            "refusing a no-op culling experiment"
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
    args = list(winner_train_args(winner=winner, iterations=iterations))
    if parameter is None:
        if value is not None:
            raise ValueError("culling value requires a parameter")
        return tuple(args)
    _validate_culling_parameter(winner, parameter)
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
    dataset = build_nerfstudio_dataset_contract(
        source,
        transforms,
        holdout_count=holdout_count,
    )
    dataset_path = root / "evaluation-dataset.json"
    write_json(dataset_path, dataset)
    split = write_nerfstudio_split_transforms(
        transforms,
        dataset,
        root / "evaluation-transforms.json",
    )
    return root, Path(split["transforms_path"]), dataset


def _base_metrics(dataset: Mapping) -> dict:
    metrics = empty_metrics()
    metrics.update(
        input_frame_count=len(dataset["frames"]),
        train_frame_count=len(dataset["train_frame_sha256"]),
        holdout_frame_count=len(dataset["holdout_frame_sha256"]),
        camera_pose_available=True,
    )
    return metrics


def _run_experiment(
    *,
    name: str,
    dataset: Mapping,
    split_transforms: Path,
    output_root: Path,
    environment: Mapping,
    train_args: Sequence[str],
    config: Mapping,
    timeout: float | None,
) -> tuple[dict, dict]:
    """Run one follow-up and preserve an explicit success/failure backend result."""
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict | None = None
    evaluation: dict | None = None
    manifest_path: Path | None = None
    ply_path: Path | None = None
    phase = "training-export"
    fallback_command = [
        "ns-train",
        "splatfacto",
        "--data",
        str(split_transforms),
        *map(str, train_args),
    ]

    try:
        result = run_splatfacto_export(
            split_transforms,
            output_root,
            train_extra_args=train_args,
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )
        train_export_seconds = time.perf_counter() - started
        manifest_path = Path(result["manifest_path"])
        run_dir = manifest_path.parent
        ply_path = run_dir / result["output"]["ply_path"]
        config_path = run_dir / result["training"]["config_path"]

        phase = "evaluation"
        evaluation = run_nerfstudio_eval(
            config_path,
            run_dir / "evaluation",
            timeout=timeout,
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        )

        phase = "artifact-metrics"
        metrics = _base_metrics(dataset)
        metrics.update(
            reconstruction_success=True,
            psnr=evaluation["metrics"].get("psnr"),
            ssim=evaluation["metrics"].get("ssim"),
            lpips=evaluation["metrics"].get("lpips"),
            wall_clock_seconds=train_export_seconds,
            peak_gpu_memory_bytes=(result.get("training") or {}).get("peak_gpu_memory_bytes"),
            **gaussian_artifact_metrics(ply_path),
        )
        backend_result = {
            "schema_version": 2,
            "dataset_id": dataset_identity(dataset),
            "backend": {
                "name": name,
                "upstream_revision": environment["nerfstudio_revision"],
            },
            "command": list(result["training"]["command"]),
            "config": dict(config),
            "return_code": 0,
            "status": "success",
            "failure_phase": None,
            "artifact": artifact_record(ply_path, format="ply"),
            "metrics": metrics,
            "training_manifest_path": str(manifest_path),
            "evaluation_manifest_path": evaluation["manifest_path"],
        }
        validate_backend_result(backend_result, dataset)
        result_path = output_root / "backend-result.json"
        write_json(result_path, backend_result)
        return (
            {
                "name": name,
                "status": "success",
                "backend_result_path": str(result_path),
                "training_manifest_path": str(manifest_path),
                "evaluation_manifest_path": evaluation["manifest_path"],
                "render_count": evaluation.get("render_count"),
                "renders": evaluation.get("renders"),
                "ply_path": str(ply_path),
                "metrics": metrics,
            },
            backend_result,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        metrics = _base_metrics(dataset)
        training = (result or {}).get("training") or {}
        metrics.update(
            reconstruction_success=result is not None,
            wall_clock_seconds=elapsed,
            peak_gpu_memory_bytes=(
                training.get("peak_gpu_memory_bytes") if isinstance(training, Mapping) else None
            ),
        )

        artifact = None
        if result is not None and manifest_path is not None:
            try:
                ply_path = manifest_path.parent / result["output"]["ply_path"]
                if ply_path.is_file():
                    artifact = artifact_record(ply_path, format="ply")
                    metrics.update(gaussian_artifact_metrics(ply_path))
            except Exception:
                artifact = None

        if isinstance(exc, subprocess.CalledProcessError):
            command = (
                list(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
            )
            return_code = int(exc.returncode)
        else:
            command = list(map(str, training.get("command") or fallback_command))
            return_code = 1

        backend_result = {
            "schema_version": 2,
            "dataset_id": dataset_identity(dataset),
            "backend": {
                "name": name,
                "upstream_revision": environment["nerfstudio_revision"],
            },
            "command": command,
            "config": dict(config),
            "return_code": return_code,
            "status": "failed",
            "failure_phase": phase,
            "artifact": artifact,
            "metrics": metrics,
            "training_manifest_path": str(manifest_path) if manifest_path else None,
            "evaluation_manifest_path": (
                evaluation.get("manifest_path") if evaluation is not None else None
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        validate_backend_result(backend_result, dataset)
        result_path = output_root / "backend-result.json"
        write_json(result_path, backend_result)
        return (
            {
                "name": name,
                "status": "failed",
                "failure_phase": phase,
                "backend_result_path": str(result_path),
                "error": backend_result["error"],
                "metrics": metrics,
            },
            backend_result,
        )


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
    """Run baseline + one culling parameter family after #43 fixes the winner."""
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    _validate_culling_parameter(winner, parameter)
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

    experiments: list[tuple[str, float | None, tuple[str, ...]]] = [
        ("baseline", None, culling_train_args(winner=winner, iterations=iterations))
    ]
    for value in unique_values:
        experiments.append(
            (
                f"{parameter}-{str(value).replace('-', 'm').replace('.', 'p')}",
                value,
                culling_train_args(
                    winner=winner,
                    iterations=iterations,
                    parameter=parameter,
                    value=value,
                ),
            )
        )

    for label, value, args in experiments:
        entry, backend_result = _run_experiment(
            name=f"splatfacto-{winner}-culling-{label}",
            dataset=dataset,
            split_transforms=split,
            output_root=root / label,
            environment=environment,
            train_args=args,
            config={
                "experiment_type": "culling-only",
                "winner": winner,
                "iterations": iterations,
                "changed_parameter": None if value is None else parameter,
                "value": value,
            },
            timeout=timeout,
        )
        summary["experiments"].append(
            {
                "changed_parameter": None if value is None else parameter,
                "value": value,
                **entry,
            }
        )
        results.append(backend_result)

    comparison_path = root / "comparison.json"
    summary["comparison"] = write_comparison(comparison_path, results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["status"] = (
        "success"
        if all(entry["status"] == "success" for entry in summary["experiments"])
        else "failed"
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
    """Change only training iteration budget after #43 fixes the winner."""
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
        entry, backend_result = _run_experiment(
            name=f"splatfacto-{winner}-iterations-{budget}",
            dataset=dataset,
            split_transforms=split,
            output_root=root / f"iterations-{budget}",
            environment=environment,
            train_args=winner_train_args(winner=winner, iterations=budget),
            config={
                "experiment_type": "budget-only",
                "winner": winner,
                "iterations": budget,
            },
            timeout=timeout,
        )
        summary["experiments"].append({"iterations": budget, **entry})
        results.append(backend_result)

    comparison_path = root / "comparison.json"
    summary["comparison"] = write_comparison(comparison_path, results, dataset)
    summary["comparison_path"] = str(comparison_path)
    summary["status"] = (
        "success"
        if all(entry["status"] == "success" for entry in summary["experiments"])
        else "failed"
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
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
