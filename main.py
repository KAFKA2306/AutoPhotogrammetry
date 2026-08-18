from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.collection import collect_images
from processing.huejotzingo import run_huejotzingo
from processing.image_selection import select_images


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

    huejotzingo_parser = subparsers.add_parser(
        "huejotzingo",
        help="Run the verified Huejotzingo video through COLMAP and Splatfacto.",
    )
    huejotzingo_parser.add_argument("--input-root", default="input")
    huejotzingo_parser.add_argument("--output-root", default="output")

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
    else:
        result = run_huejotzingo(
            input_root=args.input_root,
            output_root=args.output_root,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
