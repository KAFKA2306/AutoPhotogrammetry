from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from processing.provenance import write_json
from processing.quality_followup import CULLING_FLAGS, WINNERS, run_culling_sweep

DEFAULT_POLICY = {
    "max_psnr_drop": 0.5,
    "max_ssim_drop": 0.01,
    "max_lpips_increase": 0.02,
    "max_runtime_ratio": 1.5,
    "max_vram_ratio": 1.25,
    "max_artifact_regression": 0.002,
    "min_artifact_improvement": 0.001,
}
ARTIFACT_METRICS = ("low_opacity_primitive_ratio", "scale_anisotropy_above_10_ratio")


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def evaluate_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    active_policy = DEFAULT_POLICY if policy is None else policy
    reasons: list[str] = []
    if baseline.get("status") != "success":
        reasons.append("baseline_failed")
    if candidate.get("status") != "success":
        reasons.append("candidate_failed")

    b_psnr, c_psnr = _number(baseline, "psnr"), _number(candidate, "psnr")
    b_ssim, c_ssim = _number(baseline, "ssim"), _number(candidate, "ssim")
    b_lpips, c_lpips = _number(baseline, "lpips"), _number(candidate, "lpips")
    for key, before, after in (
        ("psnr", b_psnr, c_psnr),
        ("ssim", b_ssim, c_ssim),
        ("lpips", b_lpips, c_lpips),
    ):
        if before is None or after is None:
            reasons.append(f"unmeasured_{key}")

    if (
        b_psnr is not None
        and c_psnr is not None
        and c_psnr < b_psnr - active_policy["max_psnr_drop"]
    ):
        reasons.append("psnr_regression")
    if (
        b_ssim is not None
        and c_ssim is not None
        and c_ssim < b_ssim - active_policy["max_ssim_drop"]
    ):
        reasons.append("ssim_regression")
    if (
        b_lpips is not None
        and c_lpips is not None
        and c_lpips > b_lpips + active_policy["max_lpips_increase"]
    ):
        reasons.append("lpips_regression")

    runtime_ratio = _ratio(
        _number(candidate, "wall_clock_seconds"),
        _number(baseline, "wall_clock_seconds"),
    )
    vram_ratio = _ratio(
        _number(candidate, "peak_gpu_memory_bytes"),
        _number(baseline, "peak_gpu_memory_bytes"),
    )
    if (
        runtime_ratio is not None
        and runtime_ratio > active_policy["max_runtime_ratio"]
    ):
        reasons.append("runtime_regression")
    if vram_ratio is not None and vram_ratio > active_policy["max_vram_ratio"]:
        reasons.append("vram_regression")

    gains: dict[str, float] = {}
    for key in ARTIFACT_METRICS:
        before, after = _number(baseline, key), _number(candidate, key)
        if before is None or after is None:
            reasons.append(f"unmeasured_{key}")
            continue
        gain = before - after
        gains[key] = gain
        if gain < -active_policy["max_artifact_regression"]:
            reasons.append(f"{key}_regression")

    score = sum(gains.values())
    if (
        not gains
        or max(gains.values(), default=0.0)
        < active_policy["min_artifact_improvement"]
    ):
        reasons.append("no_measured_artifact_improvement")

    return {
        "eligible": not reasons,
        "score": score,
        "artifactGains": gains,
        "runtimeRatio": runtime_ratio,
        "vramRatio": vram_ratio,
        "reasons": sorted(set(reasons)),
    }


def _value_key(value: float) -> str:
    return format(float(value), ".12g")


def run_bounded_culling_loop(
    data_dir: str | Path,
    source_video: str | Path,
    output_root: str | Path,
    *,
    winner: str,
    parameter: str,
    values: Sequence[float],
    iterations: int = 30000,
    holdout_count: int | None = None,
    max_iterations: int = 4,
    no_improvement_limit: int = 2,
    policy: Mapping[str, float] | None = None,
    runner: Callable[..., dict] = run_culling_sweep,
) -> dict[str, Any]:
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}")
    if parameter not in CULLING_FLAGS:
        raise ValueError(f"unknown culling parameter: {parameter}")
    if max_iterations < 1 or no_improvement_limit < 1:
        raise ValueError("loop bounds must be positive")

    candidates = tuple(dict.fromkeys(float(value) for value in values))
    if not candidates:
        raise ValueError("at least one candidate value is required")

    active_policy = dict(DEFAULT_POLICY if policy is None else policy)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "quality-loop.json"

    prior: dict[str, Any] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("existing quality-loop manifest must be an object")
        prior = loaded
        if prior.get("winner") != winner or prior.get("parameter") != parameter:
            raise ValueError("existing loop manifest belongs to a different search")

    attempted = {
        _value_key(float(item["value"]))
        for item in prior.get("attempts", [])
        if isinstance(item, dict)
        and isinstance(item.get("value"), (int, float))
        and not isinstance(item.get("value"), bool)
    }
    previous_attempts = prior.get("attempts", [])
    if not isinstance(previous_attempts, list):
        raise ValueError("existing loop attempts must be a list")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_type": "bounded-culling-loop",
        "winner": winner,
        "parameter": parameter,
        "iterations": iterations,
        "max_iterations": max_iterations,
        "no_improvement_limit": no_improvement_limit,
        "policy": active_policy,
        "dataset_id": prior.get("dataset_id"),
        "attempts": list(previous_attempts),
        "last_good": prior.get(
            "last_good",
            {"kind": "baseline", "value": None, "score": 0.0, "metrics": None},
        ),
        "status": "running",
        "stop_reason": None,
    }
    last_good = summary["last_good"]
    if not isinstance(last_good, dict):
        raise ValueError("existing last_good must be an object")

    no_improvement = 0
    available = [value for value in candidates if _value_key(value) not in attempted]
    remaining_capacity = max(0, max_iterations - len(summary["attempts"]))
    scheduled = available[:remaining_capacity]

    for value in scheduled:
        index = len(summary["attempts"]) + 1
        result = runner(
            data_dir,
            source_video,
            root / f"iteration-{index:02d}-{_value_key(value).replace('.', 'p')}",
            winner=winner,
            parameter=parameter,
            values=[value],
            iterations=iterations,
            holdout_count=holdout_count,
        )
        dataset_id = result.get("dataset_id") or (
            result.get("comparison") or {}
        ).get("dataset_id")
        if not dataset_id:
            raise ValueError("sweep did not report dataset identity")
        if summary["dataset_id"] is None:
            summary["dataset_id"] = dataset_id
        elif summary["dataset_id"] != dataset_id:
            raise ValueError("dataset identity changed inside bounded loop")

        comparison = result.get("comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("sweep comparison is missing")
        rows = comparison.get("results")
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError(
                "bounded loop requires baseline + one candidate comparison"
            )
        baseline, candidate = rows
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("comparison rows must be objects")

        decision = evaluate_candidate(baseline, candidate, active_policy)
        last_good_score = last_good.get("score")
        if isinstance(last_good_score, bool) or not isinstance(
            last_good_score, (int, float)
        ):
            last_good_score = 0.0
        improved = (
            decision["eligible"]
            and float(decision["score"])
            > float(last_good_score) + active_policy["min_artifact_improvement"]
        )
        if improved:
            last_good = {
                "kind": "candidate",
                "value": value,
                "score": decision["score"],
                "metrics": dict(candidate),
            }
            summary["last_good"] = last_good
            no_improvement = 0
        else:
            no_improvement += 1

        summary["attempts"].append(
            {
                "value": value,
                "dataset_id": dataset_id,
                "comparison_path": result.get("comparison_path"),
                "candidate": dict(candidate),
                "decision": decision,
                "adoptedAsLastGood": improved,
            }
        )
        write_json(manifest_path, summary)

        if no_improvement >= no_improvement_limit:
            summary["status"] = "stopped"
            summary["stop_reason"] = "no_improvement"
            break

    if summary["status"] == "running":
        if len(summary["attempts"]) >= max_iterations:
            summary["status"] = "stopped"
            summary["stop_reason"] = "max_iterations"
        elif not available:
            summary["status"] = "stopped"
            summary["stop_reason"] = "candidate_space_exhausted"
        else:
            summary["status"] = "complete"
            summary["stop_reason"] = "candidate_space_exhausted"

    summary["manifest_path"] = str(manifest_path)
    write_json(manifest_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded one-parameter quality loop using same-holdout culling sweeps."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--winner", choices=WINNERS, required=True)
    parser.add_argument("--parameter", choices=tuple(CULLING_FLAGS), required=True)
    parser.add_argument("--value", action="append", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--no-improvement-limit", type=int, default=2)
    args = parser.parse_args()
    result = run_bounded_culling_loop(
        args.data,
        args.source_video,
        args.output_root,
        winner=args.winner,
        parameter=args.parameter,
        values=args.value,
        iterations=args.iterations,
        holdout_count=args.holdout_count,
        max_iterations=args.max_iterations,
        no_improvement_limit=args.no_improvement_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
