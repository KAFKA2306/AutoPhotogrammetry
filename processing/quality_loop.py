from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from processing.provenance import sha256_file, write_json
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
MONTAGE_CELL_SIZE = (320, 180)


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
    if runtime_ratio is not None and runtime_ratio > active_policy["max_runtime_ratio"]:
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
    if not gains or max(gains.values(), default=0.0) < active_policy["min_artifact_improvement"]:
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


def _verified_render_records(
    experiment: Mapping[str, Any],
) -> list[tuple[str, Path, dict[str, Any]]]:
    if experiment.get("status") != "success":
        return []
    manifest_value = experiment.get("evaluation_manifest_path")
    records = experiment.get("renders")
    if not isinstance(manifest_value, str) or not manifest_value:
        return []
    if not isinstance(records, list) or not records:
        return []

    manifest_path = Path(manifest_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"evaluation manifest does not exist: {manifest_path}")
    render_root = (manifest_path.parent / "renders").resolve()
    verified: list[tuple[str, Path, dict[str, Any]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"render record {index} must be an object")
        relative = record.get("path")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"render record {index} is missing path")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"render record {index} is missing SHA-256")
        path = (render_root / relative).resolve()
        if path != render_root and render_root not in path.parents:
            raise ValueError(f"render path escapes evaluation root: {relative}")
        if not path.is_file():
            raise ValueError(f"hold-out render does not exist: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"hold-out render hash mismatch: {path}")
        verified.append((relative, path, dict(record)))
    return sorted(verified, key=lambda item: item[0])


def _write_holdout_montage(
    sweep: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Any] | None:
    experiments = sweep.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 2:
        return None
    baseline_entry, candidate_entry = experiments
    if not isinstance(baseline_entry, Mapping) or not isinstance(candidate_entry, Mapping):
        raise ValueError("sweep experiments must be objects")

    baseline = _verified_render_records(baseline_entry)
    candidate = _verified_render_records(candidate_entry)
    if not baseline or not candidate:
        return None

    baseline_paths = [item[0] for item in baseline]
    candidate_paths = [item[0] for item in candidate]
    if baseline_paths != candidate_paths:
        raise ValueError(
            "baseline/candidate hold-out render sets differ: "
            f"baseline={baseline_paths}, candidate={candidate_paths}"
        )

    cell_width, cell_height = MONTAGE_CELL_SIZE
    canvas = Image.new(
        "RGB",
        (cell_width * len(baseline_paths), cell_height * 2),
        "white",
    )
    for row_index, records in enumerate((baseline, candidate)):
        for column_index, (_, path, _) in enumerate(records):
            with Image.open(path) as source:
                image = ImageOps.contain(
                    source.convert("RGB"),
                    MONTAGE_CELL_SIZE,
                    method=Image.Resampling.LANCZOS,
                )
            x = column_index * cell_width + (cell_width - image.width) // 2
            y = row_index * cell_height + (cell_height - image.height) // 2
            canvas.paste(image, (x, y))

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return {
        "schema_version": 1,
        "path": str(output),
        "sha256": sha256_file(output),
        "layout": {
            "rows": ["baseline", "candidate"],
            "columns": baseline_paths,
            "cell_size": [cell_width, cell_height],
        },
        "baseline_renders": [item[2] for item in baseline],
        "candidate_renders": [item[2] for item in candidate],
    }


def _all_candidates_failed(
    attempts: Sequence[Mapping[str, Any]],
    candidates: Sequence[float],
) -> bool:
    expected = {_value_key(value) for value in candidates}
    by_value: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        value = attempt.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        by_value[_value_key(float(value))] = attempt
    if not expected or not expected.issubset(by_value):
        return False
    return all(
        isinstance(by_value[key].get("candidate"), Mapping)
        and by_value[key]["candidate"].get("status") == "failed"
        for key in expected
    )


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
        iteration_root = root / f"iteration-{index:02d}-{_value_key(value).replace('.', 'p')}"
        result = runner(
            data_dir,
            source_video,
            iteration_root,
            winner=winner,
            parameter=parameter,
            values=[value],
            iterations=iterations,
            holdout_count=holdout_count,
        )
        dataset_id = result.get("dataset_id") or (result.get("comparison") or {}).get("dataset_id")
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
            raise ValueError("bounded loop requires baseline + one candidate comparison")
        baseline, candidate = rows
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("comparison rows must be objects")

        decision = evaluate_candidate(baseline, candidate, active_policy)
        render_evidence = _write_holdout_montage(
            result,
            iteration_root / "holdout-before-after.png",
        )
        last_good_score = last_good.get("score")
        if isinstance(last_good_score, bool) or not isinstance(last_good_score, (int, float)):
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
                "renderEvidence": render_evidence,
                "adoptedAsLastGood": improved,
            }
        )
        write_json(manifest_path, summary)

        if no_improvement >= no_improvement_limit:
            summary["status"] = "stopped"
            summary["stop_reason"] = (
                "all_failed"
                if _all_candidates_failed(summary["attempts"], candidates)
                else "no_improvement"
            )
            break

    if summary["status"] == "running":
        if _all_candidates_failed(summary["attempts"], candidates):
            summary["status"] = "stopped"
            summary["stop_reason"] = "all_failed"
        elif len(summary["attempts"]) >= max_iterations:
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
        description=("Bounded one-parameter quality loop using same-holdout culling sweeps.")
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
