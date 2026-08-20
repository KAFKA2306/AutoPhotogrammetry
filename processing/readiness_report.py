from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from processing.image_selection import calculate_sharpness, select_images
from processing.provenance import IMAGE_SUFFIXES, sha256_file, write_json

REQUIRED_PROVENANCE_FIELDS = (
    "source_page",
    "image_url",
    "sha256",
    "width",
    "height",
    "content_type",
)


def _read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Collection manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Collection manifest must be a JSON array of objects: {path}")
    return payload


def _render_html(report: dict) -> str:
    reason_counts = report["selection"]["reason_counts"]
    dimensions = report["dimensions"]
    rows = [
        ("Input images", report["input"]["count"]),
        ("Selected images", report["selection"]["selected_count"]),
        ("Low sharpness", reason_counts["LOW_SHARPNESS"]),
        ("Near duplicate", reason_counts["NEAR_DUPLICATE"]),
        ("Exact duplicate manifest rows", reason_counts["EXACT_DUPLICATE"]),
        ("Missing provenance", reason_counts["PROVENANCE_MISSING"]),
        (
            "Provenance coverage",
            f"{report['provenance']['covered_count']}/{report['input']['count']}",
        ),
        ("Distinct image sizes", dimensions["distinct_size_count"]),
        ("Backend execution", report["backend"]["status"]),
    ]
    table = "\n".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Photogrammetry input audit — {html.escape(report["asset_id"])}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: .6rem; text-align: left; }}
    .notice {{ background: #f6f6f6; padding: 1rem; border-radius: .5rem; }}
  </style>
</head>
<body>
  <h1>Photogrammetry input audit</h1>
  <p><strong>Asset:</strong> {html.escape(report["asset_id"])}</p>
  <table><tbody>{table}</tbody></table>
  <div class="notice">
    <p>This report describes input selection and provenance only.</p>
    <p>Registration rate, reprojection error, geometry completeness, and texture quality are not measured here, so this report does not guarantee 3D reconstruction quality.</p>
  </div>
</body>
</html>
"""


def build_readiness_report(
    dataset: str,
    *,
    input_root: str | Path = "input",
    output_root: str | Path = "output",
    sharpness_threshold: float = 0.0001,
    similarity_threshold: float = 0.92,
) -> dict:
    image_dir = Path(input_root) / dataset / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Input image directory does not exist: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    source_manifest_path = image_dir / "manifest.json"
    source_records = _read_manifest(source_manifest_path)
    output_dir = Path(output_root) / dataset
    selected_dir = output_dir / "selected"
    selected_paths = select_images(
        image_paths,
        sharpness_threshold=sharpness_threshold,
        similarity_threshold=similarity_threshold,
        output_dir=selected_dir,
    )
    selected_names = {path.name for path in selected_paths}

    low_sharpness = 0
    near_duplicate = 0
    for image_path in image_paths:
        if image_path.name in selected_names:
            continue
        if calculate_sharpness(image_path) < sharpness_threshold:
            low_sharpness += 1
        else:
            near_duplicate += 1

    hashes = [str(record.get("sha256", "")) for record in source_records if record.get("sha256")]
    exact_duplicates = len(hashes) - len(set(hashes))
    records_by_name = {
        Path(str(record.get("local_path", ""))).name: record for record in source_records
    }
    covered = 0
    dimensions: list[tuple[int, int]] = []
    content_types: Counter[str] = Counter()
    for image_path in image_paths:
        record = records_by_name.get(image_path.name)
        if record and all(
            record.get(field) not in (None, "") for field in REQUIRED_PROVENANCE_FIELDS
        ):
            covered += 1
        if record:
            width = record.get("width")
            height = record.get("height")
            if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
                dimensions.append((width, height))
            content_type = record.get("content_type")
            if isinstance(content_type, str) and content_type:
                content_types[content_type] += 1

    missing_provenance = len(image_paths) - covered
    distinct_sizes = sorted(set(dimensions))
    report = {
        "schema_version": 1,
        "asset_id": dataset,
        "generated_views_used": False,
        "input": {
            "count": len(image_paths),
            "source_manifest": str(source_manifest_path),
        },
        "selection": {
            "selected_count": len(selected_paths),
            "selected_dir": str(selected_dir),
            "sharpness_threshold": sharpness_threshold,
            "similarity_threshold": similarity_threshold,
            "reason_counts": {
                "LOW_SHARPNESS": low_sharpness,
                "NEAR_DUPLICATE": near_duplicate,
                "EXACT_DUPLICATE": exact_duplicates,
                "PROVENANCE_MISSING": missing_provenance,
            },
        },
        "provenance": {
            "covered_count": covered,
            "coverage_ratio": covered / len(image_paths),
        },
        "dimensions": {
            "distinct_size_count": len(distinct_sizes),
            "sizes": [[width, height] for width, height in distinct_sizes],
            "content_types": dict(sorted(content_types.items())),
        },
        "backend": {
            "status": "not_run",
            "run_manifest_path": None,
        },
        "quality_measurements": {
            "registration_rate": None,
            "reprojection_error": None,
            "mesh_completeness": None,
            "quality_guarantee": False,
        },
    }

    selected_manifest = {
        "schema_version": 1,
        "asset_id": dataset,
        "generated_views_used": False,
        "source_manifest": str(source_manifest_path),
        "images": [
            {
                "filename": path.name,
                "sha256": sha256_file(path),
            }
            for path in selected_paths
        ],
    }
    selected_manifest_path = output_dir / "selected-manifest.json"
    report_json_path = output_dir / "readiness-report.json"
    report_html_path = output_dir / "readiness-report.html"
    write_json(selected_manifest_path, selected_manifest)
    write_json(report_json_path, report)
    report_html_path.write_text(_render_html(report), encoding="utf-8")

    return {
        "asset_id": dataset,
        "report_json": str(report_json_path),
        "report_html": str(report_html_path),
        "selected_manifest": str(selected_manifest_path),
        "input": len(image_paths),
        "selected": len(selected_paths),
    }
