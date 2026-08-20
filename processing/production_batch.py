from __future__ import annotations

import argparse
import json

from processing.batch import run_all_videos
from processing.exhibition_manifest import build_final_exhibition_manifest


def run_production_batch(
    *,
    registry_path: str = "sources/videos.json",
    input_root: str = "input",
    output_root: str = "output",
    train_iterations: int = 2000,
    timeout: float | None = None,
    fresh: bool = False,
) -> dict:
    """Run the full registry and finalize the downstream handoff only at exact 20/20 success."""
    batch = run_all_videos(
        registry_path=registry_path,
        input_root=input_root,
        output_root=output_root,
        ids=None,
        train_iterations=train_iterations,
        timeout=timeout,
        fresh=fresh,
    )
    if batch.get("status") != "success":
        return {
            "status": "failed",
            "failed_phase": "batch",
            "batch": batch,
            "final_exhibition_manifest": None,
        }

    try:
        final_manifest = build_final_exhibition_manifest(
            registry_path,
            output_root,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "failed_phase": "final-exhibition-manifest",
            "error": f"{type(exc).__name__}: {exc}",
            "batch": batch,
            "final_exhibition_manifest": None,
        }

    return {
        "status": "success",
        "failed_phase": None,
        "batch": batch,
        "final_exhibition_manifest": {
            "path": final_manifest["manifest_path"],
            "entry_count": final_manifest["entry_count"],
            "status": final_manifest["status"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all registered production videos and emit the final exactly-20 exhibition manifest."
        )
    )
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--train-iterations", type=int, default=2000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_production_batch(
        registry_path=args.registry,
        input_root=args.input_root,
        output_root=args.output_root,
        train_iterations=args.train_iterations,
        timeout=args.timeout,
        fresh=args.fresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
