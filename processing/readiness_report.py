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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPORT_CSS_ENTRY = _REPOSITORY_ROOT / "assets" / "readiness-report.css"
_MANAGED_CSS_FILES = (
    _REPOSITORY_ROOT / "assets" / ".kafka-design" / "kafka-tokens.css",
    _REPOSITORY_ROOT / "assets" / ".kafka-design" / "kafka-globals.css",
    _REPOSITORY_ROOT / "assets" / ".kafka-design" / "kafka-components.css",
)
_MANAGED_START = "/* kafka-design:managed-start */"
_MANAGED_END = "/* kafka-design:managed-end */"


def _read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Collection manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Collection manifest must be a JSON array of objects: {path}")
    return payload


def _records_by_filename(source_records: list[dict], manifest_path: Path) -> dict[str, dict]:
    records_by_name: dict[str, dict] = {}
    for record in source_records:
        local_path = record.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            continue
        name = Path(local_path).name
        if name in records_by_name:
            raise ValueError(
                f"Collection manifest has duplicate local_path filename {name!r}: {manifest_path}"
            )
        records_by_name[name] = record
    return records_by_name


def _backend_evidence(dataset: str, manifest_path: str | Path | None) -> dict:
    if manifest_path is None:
        return {
            "asset_id": dataset,
            "status": "not_run",
            "run_manifest_path": None,
            "run_manifest_sha256": None,
            "return_code": None,
        }

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Backend run manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Backend run manifest must be a JSON object: {path}")

    identity_fields = ("asset_id", "dataset", "scene", "source_id")
    identities = {
        field: str(payload[field])
        for field in identity_fields
        if payload.get(field) not in (None, "")
    }
    mismatches = {field: value for field, value in identities.items() if value != dataset}
    if mismatches:
        raise ValueError(
            f"Backend run manifest identity does not match asset_id {dataset!r}: {mismatches}"
        )

    return_code = payload.get("return_code")
    declared_status = payload.get("status")
    if isinstance(declared_status, str) and declared_status:
        status = declared_status
    elif return_code == 0:
        status = "success"
    elif isinstance(return_code, int):
        status = "failed"
    else:
        status = "recorded"

    return {
        "asset_id": dataset,
        "status": status,
        "run_manifest_path": str(path),
        "run_manifest_sha256": sha256_file(path),
        "return_code": return_code if isinstance(return_code, int) else None,
    }



def _audit_decision(report: dict) -> dict[str, str]:
    selected = report["selection"]["selected_count"]
    input_count = report["input"]["count"]
    missing_provenance = report["selection"]["reason_counts"]["PROVENANCE_MISSING"]
    backend = report["backend"]
    backend_status = backend["status"]
    return_code = backend["return_code"]

    if selected <= 0:
        return {
            "status": "BLOCKED",
            "reason": "No input image passed selection, so there is nothing to hand to a reconstruction backend.",
            "next_action": "Review the sharpness and similarity thresholds, then rerun input selection.",
        }

    if missing_provenance > 0:
        return {
            "status": "ACTION_REQUIRED",
            "reason": (
                f"{missing_provenance} of {input_count} input images are missing required provenance fields."
            ),
            "next_action": (
                "Complete source_page, image_url, sha256, width, height, and content_type in the input manifest."
            ),
        }

    if backend_status == "failed" or (
        isinstance(return_code, int) and return_code != 0
    ):
        return {
            "status": "BLOCKED",
            "reason": "The recorded reconstruction backend execution failed.",
            "next_action": "Inspect the backend run manifest, resolve the failure, and rerun the backend.",
        }

    if backend_status == "not_run":
        return {
            "status": "READY_FOR_BACKEND",
            "reason": "Input selection and provenance checks have evidence, but no reconstruction backend run is attached.",
            "next_action": "Run a reconstruction backend and attach its run manifest to this audit.",
        }

    if backend_status == "success":
        return {
            "status": "REVIEW_READY",
            "reason": "Input selection, provenance, and a successful backend run are recorded.",
            "next_action": (
                "Evaluate registration, reprojection, geometry completeness, and texture quality before claiming 3D quality."
            ),
        }

    return {
        "status": "REVIEW_REQUIRED",
        "reason": f"Backend evidence is recorded with status {backend_status!r}, but success is not established.",
        "next_action": "Inspect the backend run manifest and establish an explicit success or failure state.",
    }

def _report_css() -> str:
    if not _REPORT_CSS_ENTRY.is_file():
        raise FileNotFoundError(f"Report CSS entry does not exist: {_REPORT_CSS_ENTRY}")

    entry = _REPORT_CSS_ENTRY.read_text(encoding="utf-8")
    start = entry.find(_MANAGED_START)
    end = entry.find(_MANAGED_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Managed design import block is invalid: {_REPORT_CSS_ENTRY}")

    local_css = (entry[:start] + entry[end + len(_MANAGED_END) :]).strip()
    managed_parts: list[str] = []
    for css_path in _MANAGED_CSS_FILES:
        if not css_path.is_file():
            raise FileNotFoundError(f"Managed design CSS does not exist: {css_path}")
        managed_parts.append(css_path.read_text(encoding="utf-8").strip())
    return "\n\n".join([*managed_parts, local_css])


def _render_html(report: dict) -> str:
    reason_counts = report["selection"]["reason_counts"]
    dimensions = report["dimensions"]
    decision = report["decision"]
    selected = report["selection"]["selected_count"]
    input_count = report["input"]["count"]
    covered = report["provenance"]["covered_count"]
    rows = [
        ("Input images", input_count),
        ("Selected images", selected),
        ("Low sharpness", reason_counts["LOW_SHARPNESS"]),
        ("Near duplicate", reason_counts["NEAR_DUPLICATE"]),
        ("Exact duplicate manifest rows", reason_counts["EXACT_DUPLICATE"]),
        ("Missing provenance", reason_counts["PROVENANCE_MISSING"]),
        ("Provenance coverage", f"{covered}/{input_count}"),
        ("Distinct image sizes", dimensions["distinct_size_count"]),
        ("Backend execution", report["backend"]["status"]),
        ("Backend manifest", report["backend"]["run_manifest_path"] or "not recorded"),
    ]
    table = "\n".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    report_css = _report_css()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Photogrammetry input audit — {html.escape(report["asset_id"])}</title>
  <style>
{report_css}
  </style>
</head>
<body>
  <header class="audit-header">
    <p class="audit-kicker">AutoPhotogrammetry / input audit</p>
    <div class="audit-title-row">
      <div>
        <h1>Photogrammetry input audit</h1>
        <p class="audit-asset"><strong>Asset:</strong> {html.escape(report["asset_id"])}</p>
      </div>
      <strong class="audit-status" data-status="{html.escape(decision["status"])}">{html.escape(decision["status"])}</strong>
    </div>
  </header>
  <main>
    <section class="decision-surface" aria-labelledby="decision-heading">
      <div>
        <p class="surface-kicker">Current decision</p>
        <h2 id="decision-heading">{html.escape(decision["reason"])}</h2>
      </div>
      <div class="next-action">
        <span>Next action</span>
        <strong>{html.escape(decision["next_action"])}</strong>
      </div>
    </section>

    <section class="kpi-grid" aria-label="Readiness summary">
      <div class="kpi">
        <span>Selected</span>
        <strong>{selected}/{input_count}</strong>
        <small>input images</small>
      </div>
      <div class="kpi">
        <span>Provenance</span>
        <strong>{covered}/{input_count}</strong>
        <small>fully covered</small>
      </div>
      <div class="kpi">
        <span>Backend</span>
        <strong>{html.escape(str(report["backend"]["status"]))}</strong>
        <small>execution evidence</small>
      </div>
      <div class="kpi">
        <span>Image sizes</span>
        <strong>{dimensions["distinct_size_count"]}</strong>
        <small>distinct sizes</small>
      </div>
    </section>

    <details class="evidence-details">
      <summary>Evidence details</summary>
      <table><tbody>{table}</tbody></table>
    </details>

    <div class="notice">
      <p><strong>Scope boundary</strong></p>
      <p>This report describes input selection and provenance only.</p>
      <p>Registration rate, reprojection error, geometry completeness, and texture quality are not measured here, so this report does not guarantee 3D reconstruction quality.</p>
    </div>
  </main>
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
    backend_run_manifest: str | Path | None = None,
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
    records_by_name = _records_by_filename(source_records, source_manifest_path)

    current_manifest_hashes: list[str] = []
    for image_path in image_paths:
        record = records_by_name.get(image_path.name)
        if not record:
            continue
        recorded_hash = record.get("sha256")
        if isinstance(recorded_hash, str) and recorded_hash:
            actual_hash = sha256_file(image_path)
            if recorded_hash != actual_hash:
                raise ValueError(
                    f"Collection manifest SHA-256 does not match input file {image_path.name!r}: "
                    f"recorded={recorded_hash} actual={actual_hash}"
                )
            current_manifest_hashes.append(recorded_hash)

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

    exact_duplicates = len(current_manifest_hashes) - len(set(current_manifest_hashes))
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
    backend = _backend_evidence(dataset, backend_run_manifest)
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
        "backend": backend,
        "quality_measurements": {
            "registration_rate": None,
            "reprojection_error": None,
            "mesh_completeness": None,
            "quality_guarantee": False,
        },
    }
    report["decision"] = _audit_decision(report)

    selected_manifest = {
        "schema_version": 1,
        "asset_id": dataset,
        "generated_views_used": False,
        "source_manifest": str(source_manifest_path),
        "backend_run_manifest": backend["run_manifest_path"],
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
