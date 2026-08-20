from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.provenance import write_json
from processing.quality_sweep import run_quality_sweep


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_quality_pair(
    bad_scene: str,
    good_control_scene: str,
    *,
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    iterations: int = 30000,
    holdout_count: int | None = None,
    timeout: float | None = None,
) -> dict:
    """Run the explicit bad/control pair while preserving one shared execution authority.

    Scene roles must be supplied by the operator/evidence owner. The repository never
    infers which existing scene is visually bad or good. Both scene sweeps are attempted
    even when the first one fails, and their own quality-sweep manifests remain the
    detailed metric authority.
    """
    bad = bad_scene.strip()
    good = good_control_scene.strip()
    if not bad or not good:
        raise ValueError("bad_scene and good_control_scene are required")
    if bad == good:
        raise ValueError("bad_scene and good_control_scene must be different")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    pair_root = output_path / "quality-pair" / f"{bad}--{good}"
    pair_root.mkdir(parents=True, exist_ok=True)
    manifest_path = pair_root / "quality-pair.json"

    summary = {
        "schema_version": 1,
        "status": "running",
        "iterations": iterations,
        "holdout_count": holdout_count,
        "roles": {
            "bad_scene": bad,
            "good_control": good,
        },
        "scene_runs": [],
        "winner_decision_inputs": [],
        "selected_winner": None,
        "selection_policy": (
            "Do not select a production strategy automatically. Compare the bad scene and good "
            "control using their common comparison rows, same-holdout metrics, hashed renders, "
            "artifact metrics, runtime, and measured peak GPU memory."
        ),
    }
    write_json(manifest_path, summary)

    for role, scene in (("bad_scene", bad), ("good_control", good)):
        data = output_path / scene / "nerfstudio-data"
        source = input_path / scene / "source.webm"
        sweep_root = output_path / scene / "quality-sweep"
        sweep_manifest_path = sweep_root / "quality-sweep.json"
        record = {
            "role": role,
            "scene_id": scene,
            "data_path": str(data),
            "source_video_path": str(source),
            "quality_sweep_manifest": str(sweep_manifest_path),
            "comparison_path": None,
            "status": "running",
            "error": None,
        }
        summary["scene_runs"].append(record)
        write_json(manifest_path, summary)

        try:
            run_quality_sweep(
                data,
                source,
                sweep_root,
                iterations=iterations,
                holdout_count=holdout_count,
                timeout=timeout,
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["status"] = "success"

        sweep = _read_json(sweep_manifest_path)
        if sweep:
            record["status"] = str(sweep.get("status") or record["status"])
            record["comparison_path"] = sweep.get("comparison_path")
            record["dataset_id"] = sweep.get("dataset_id")
            record["failed_variants"] = sweep.get("failed_variants")
            if sweep.get("comparison_path"):
                summary["winner_decision_inputs"].append(
                    {
                        "role": role,
                        "scene_id": scene,
                        "comparison_path": sweep["comparison_path"],
                    }
                )
        write_json(manifest_path, summary)

    summary["status"] = (
        "success"
        if all(record["status"] == "success" for record in summary["scene_runs"])
        else "failed"
    )
    summary["manifest_path"] = str(manifest_path)
    write_json(manifest_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an explicitly named bad scene and good control through the canonical "
            "Splatfacto quality sweep and persist one shared execution manifest."
        )
    )
    parser.add_argument("--bad-scene", required=True)
    parser.add_argument("--good-control", required=True)
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()

    result = run_quality_pair(
        args.bad_scene,
        args.good_control,
        input_root=args.input_root,
        output_root=args.output_root,
        iterations=args.iterations,
        holdout_count=args.holdout_count,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
