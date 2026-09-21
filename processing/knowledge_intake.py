from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY = "KAFKA2306/AutoPhotogrammetry"

REQUIRED_METRICS = [
    "runtime_s",
    "peak_vram_mb",
    "artifact_size_bytes",
    "artifact_readable",
    "downstream_readiness",
]

OPTIONAL_IMAGE_METRICS = ["psnr", "ssim", "lpips"]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Knowledge event at line {line_number} is not an object")
        events.append(value)
    return events


def _experiment_kind(event: dict[str, Any]) -> str:
    tags = set(event.get("claim", {}).get("tags", []))
    statement = str(event.get("claim", {}).get("statement", "")).lower()
    if "reconstruction" in tags or any(word in statement for word in ("gaussian", "splat", "colmap")):
        return "reconstruction_backend"
    if "performance" in tags:
        return "performance"
    if "rendering" in tags:
        return "appearance"
    return "reconstruction_method"


def build_experiment_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        event_id = str(event.get("id", ""))
        if not event_id or event_id in seen:
            continue
        repositories = event.get("applies_to", {}).get("repositories", [])
        if REPOSITORY not in repositories:
            continue
        seen.add(event_id)
        evidence = event.get("evidence", {})
        numeric_claim = bool(evidence.get("numeric_claim_present", False))
        candidates.append(
            {
                "schema_version": 1,
                "knowledge_event_id": event_id,
                "status": "CANDIDATE",
                "experiment_kind": _experiment_kind(event),
                "source": event.get("source", {}),
                "claim": event.get("claim", {}),
                "source_evidence_status": evidence.get("status", "CLAIMED"),
                "verification_contract": {
                    "fixed_dataset_identity_required": True,
                    "baseline_required": True,
                    "one_variable_only": True,
                    "same_evaluation_semantics_required": True,
                    "actual_artifact_hash_required": True,
                    "required_metrics": REQUIRED_METRICS,
                    "optional_holdout_metrics": OPTIONAL_IMAGE_METRICS,
                    "metric_scale_is_independent_gate": True,
                    "physical_up_is_independent_gate": True,
                },
                "source_numeric_claim": {
                    "present": numeric_claim,
                    "may_populate_actual_metrics": False,
                    "requires_local_measurement": numeric_claim,
                },
                "promotion": {
                    "allowed_after": [
                        "baseline_replay",
                        "candidate_run",
                        "artifact_validation",
                        "same_contract_comparison",
                    ],
                    "unavailable_backend_result": "UNVERIFIED",
                },
            }
        )
    return candidates


def write_experiment_candidates(events_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    events = _load_jsonl(events_path)
    candidates = build_experiment_candidates(events)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_events": str(events_path),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "success",
        "candidate_count": len(candidates),
        "output": str(output),
    }\n