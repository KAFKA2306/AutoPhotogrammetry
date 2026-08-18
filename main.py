from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.backend_evaluation import build_dataset_contract, write_comparison
from processing.collection import collect_images
from processing.batch import run_all_videos
from processing.huejotzingo import run_huejotzingo
from processing.image_selection import select_images
from processing.provenance import write_json
from processing.readiness_report import build_readiness_report


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move trusted visual input through processing/ into auditable output/."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--page-url", action="append", required=True)
    collect_parser.add_argument("--keyword", action="append", default=[])
    collect_parser.add_argument("--dataset", required=True)
    collect_parser.add_argument("--input-root", default="input")

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--dataset", required=True)
    select_parser.add_argument("--input-root", default="input")
    select_parser.add_argument("--output-root", default="output")
    select_parser.add_argument("--sharpness-threshold", type=float, default=0.0001)
    select_parser.add_argument("--similarity-threshold", type=float, default=0.92)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Select a collected image set and generate JSON/HTML input audit reports.",
    )
    audit_parser.add_argument("--dataset", required=True)
    audit_parser.add_argument("--input-root", default="input")
    audit_parser.add_argument("--output-root", default="output")
    audit_parser.add_argument("--sharpness-threshold", type=float, default=0.0001)
    audit_parser.add_argument("--similarity-threshold", type=float, default=0.92)

    huejotzingo_parser = subparsers.add_parser(
        "huejotzingo",
        help="Run the verified Huejotzingo video through COLMAP and Splatfacto.",
    )
    huejotzingo_parser.add_argument("--input-root", default="input")
    huejotzingo_parser.add_argument("--output-root", default="output")

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run every registry video through COLMAP and Gaussian Splatting.",
    )
    batch_parser.add_argument("--registry", default="sources/videos.json")
    batch_parser.add_argument("--id", action="append", dest="ids")
    batch_parser.add_argument("--input-root", default="input")
    batch_parser.add_argument("--output-root", default="output")
    batch_parser.add_argument("--timeout", type=float, default=None)
    batch_parser.add_argument("--train-iterations", type=int, default=2000)
    batch_parser.add_argument("--fresh", action="store_true")

    dataset_parser = subparsers.add_parser(
        "evaluation-dataset",
        help="Freeze one source video and exact train/hold-out frame hashes for backend comparison.",
    )
    dataset_parser.add_argument("--source-video", required=True)
    dataset_parser.add_argument("--frames", required=True)
    dataset_parser.add_argument("--holdout-count", type=int, required=True)
    dataset_parser.add_argument("--output", required=True)

    compare_parser = subparsers.add_parser(
        "evaluation-compare",
        help="Validate backend result manifests against one dataset contract and write comparison rows.",
    )
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--result", action="append", required=True)
    compare_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "collect":
        destination = Path(args.input_root) / args.dataset / "images"
        records = collect_images(args.keyword, args.page_url, destination)
        result = {
            "dataset": args.dataset,
            "input_dir": str(destination),
            "collected": len(records),
        }
    elif args.command == "select":
        image_dir = Path(args.input_root) / args.dataset / "images"
        if not image_dir.is_dir():
            raise SystemExit(f"Input image directory does not exist: {image_dir}")
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        )
        output_dir = Path(args.output_root) / args.dataset / "selected"
        selected = select_images(
            image_paths,
            sharpness_threshold=args.sharpness_threshold,
            similarity_threshold=args.similarity_threshold,
            output_dir=output_dir,
        )
        result = {
            "dataset": args.dataset,
            "input": len(image_paths),
            "selected": len(selected),
            "output_dir": str(output_dir),
        }
    elif args.command == "audit":
        result = build_readiness_report(
            args.dataset,
            input_root=args.input_root,
            output_root=args.output_root,
            sharpness_threshold=args.sharpness_threshold,
            similarity_threshold=args.similarity_threshold,
        )
    elif args.command == "huejotzingo":
        result = run_huejotzingo(
            input_root=args.input_root,
            output_root=args.output_root,
        )
    elif args.command == "evaluation-dataset":
        result = build_dataset_contract(
            args.source_video,
            args.frames,
            holdout_count=args.holdout_count,
        )
        write_json(args.output, result)
        result = {**result, "manifest_path": str(Path(args.output))}
    elif args.command == "evaluation-compare":
        dataset = _read_json(args.dataset)
        backend_results = [_read_json(path) for path in args.result]
        result = write_comparison(args.output, backend_results, dataset)
        result = {**result, "comparison_path": str(Path(args.output))}
    else:
        result = run_all_videos(
            registry_path=args.registry,
            input_root=args.input_root,
            output_root=args.output_root,
            ids=args.ids,
            train_iterations=args.train_iterations,
            timeout=args.timeout,
            fresh=args.fresh,
        )
        if result["status"] != "success":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
