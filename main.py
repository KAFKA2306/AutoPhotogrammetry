from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.collection import collect_images
from processing.image_selection import select_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect explicit-source images into input/ and write selected images to output/."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Download images from explicitly supplied HTML pages into input/<dataset>/images.",
    )
    collect_parser.add_argument("--page-url", action="append", required=True)
    collect_parser.add_argument("--keyword", action="append", default=[])
    collect_parser.add_argument("--dataset", required=True)
    collect_parser.add_argument("--input-root", default="input")

    select_parser = subparsers.add_parser(
        "select",
        help="Copy sharp, non-duplicate images from input/<dataset>/images to output/<dataset>/selected.",
    )
    select_parser.add_argument("--dataset", required=True)
    select_parser.add_argument("--input-root", default="input")
    select_parser.add_argument("--output-root", default="output")
    select_parser.add_argument("--sharpness-threshold", type=float, default=0.0001)
    select_parser.add_argument("--similarity-threshold", type=float, default=0.92)

    args = parser.parse_args()

    if args.command == "collect":
        destination = Path(args.input_root) / args.dataset / "images"
        records = collect_images(
            args.keyword,
            args.page_url,
            destination,
        )
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "input_dir": str(destination),
                    "collected": len(records),
                },
                ensure_ascii=False,
            )
        )
        return

    image_dir = Path(args.input_root) / args.dataset / "images"
    output_dir = Path(args.output_root) / args.dataset / "selected"
    if not image_dir.is_dir():
        raise SystemExit(f"Input image directory does not exist: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    selected = select_images(
        image_paths,
        sharpness_threshold=args.sharpness_threshold,
        similarity_threshold=args.similarity_threshold,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "input": len(image_paths),
                "selected": len(selected),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
