from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

import yaml

from processing.orientation import OrientationContractError, write_orientation_evidence
from processing.provenance import sha256_file, write_json


class ArtifactPublishError(RuntimeError):
    pass


def _validate_revision(value: str, label: str) -> str:
    revision = value.strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ArtifactPublishError(f"{label} must be a full lowercase 40-character Git commit SHA")
    return revision


def _recorded_revision(run_manifest: dict, explicit_revision: str | None) -> str:
    recorded = run_manifest.get("source_revision")
    if recorded is not None:
        revision = _validate_revision(str(recorded), "run manifest source_revision")
        if explicit_revision is not None:
            explicit = _validate_revision(explicit_revision, "explicit source_revision")
            if explicit != revision:
                raise ArtifactPublishError(
                    "explicit source_revision does not match the generation-time revision recorded in the run manifest"
                )
        return revision
    if explicit_revision is None:
        raise ArtifactPublishError(
            "run manifest has no generation-time source_revision; provide --source-revision only for an audited legacy run"
        )
    return _validate_revision(explicit_revision, "explicit source_revision")


def _resolve_run_artifact_path(run_manifest_path: Path, raw_path: str) -> Path:
    """Resolve host and container paths without changing the recorded artifact identity."""
    run_manifest_path = run_manifest_path.expanduser().resolve()
    raw = Path(raw_path)
    if raw.is_absolute() and raw.is_file():
        return raw

    output_root = run_manifest_path.parent.parent
    repo_root = output_root.parent
    candidates: list[Path] = []
    if raw.is_absolute():
        posix = PurePosixPath(raw_path)
        workspace_output = PurePosixPath("/workspace/output")
        try:
            relative = posix.relative_to(workspace_output)
        except ValueError:
            relative = None
        if relative is not None:
            candidates.append(output_root.joinpath(*relative.parts))
    else:
        candidates.extend((run_manifest_path.parent / raw, repo_root / raw))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ArtifactPublishError(
        f"Gaussian Splat PLY cannot be resolved from run manifest path: {raw_path}"
    )


def _resolve_transforms_path(run_manifest_path: Path, run_manifest: dict) -> Path:
    declared = run_manifest.get("orientation_transforms_path")
    if declared:
        candidate = Path(str(declared)).expanduser()
        if not candidate.is_absolute():
            candidate = run_manifest_path.parent / candidate
        candidate = candidate.resolve()
    else:
        candidate = (run_manifest_path.parent / "nerfstudio-data" / "transforms.json").resolve()
    if not candidate.is_file():
        raise ArtifactPublishError(
            "Nerfstudio transforms.json is required before Gaussian artifact publish; "
            f"expected {candidate}"
        )
    return candidate


def _resolve_physical_up_path(run_manifest_path: Path, run_manifest: dict) -> Path | None:
    declared = run_manifest.get("physical_up_evidence_path")
    if declared is not None:
        if not isinstance(declared, str) or not declared.strip():
            raise ArtifactPublishError("physical_up_evidence_path must be a non-empty path when declared")
        candidate = Path(declared).expanduser()
        if not candidate.is_absolute():
            candidate = run_manifest_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ArtifactPublishError(f"declared physical-up evidence file is missing: {candidate}")
        return candidate

    candidate = (run_manifest_path.parent / "physical-up-evidence.json").resolve()
    return candidate if candidate.is_file() else None


def _hf_cache_command(hf_cache_hub_root: str | Path | None) -> list[str]:
    root_value = hf_cache_hub_root or os.environ.get("HF_CACHE_HUB_ROOT")
    if not root_value:
        raise ArtifactPublishError("HF_CACHE_HUB_ROOT or --hf-cache-hub-root is required")
    root = Path(root_value).expanduser()
    script = root / "scripts" / "artifact_manager.py"
    if not script.is_file():
        raise ArtifactPublishError(f"hf-cache-hub artifact CLI not found: {script}")
    python_executable = os.environ.get("HF_CACHE_HUB_PYTHON") or sys.executable
    return [python_executable, str(script)]


def build_artifact_manifest(
    *,
    dataset: str,
    ply_path: Path,
    bucket: str,
    source_revision: str,
    run_id: str,
    orientation: dict,
    source_url: str | None = None,
    license_url: str | None = None,
) -> tuple[dict, str]:
    if not ply_path.is_file() or ply_path.stat().st_size == 0:
        raise ArtifactPublishError(f"Gaussian Splat PLY is missing or empty: {ply_path}")
    sha256 = sha256_file(ply_path)
    if orientation.get("ply_sha256") != sha256:
        raise ArtifactPublishError("orientation evidence does not match artifact PLY SHA-256")
    if orientation.get("status") != "accepted":
        raise ArtifactPublishError("only accepted orientation basis evidence can be published")

    artifact_id = f"autophotogrammetry/{dataset}/splat"
    remote_path = f"autophotogrammetry/gaussian-splats/{dataset}/{sha256}.ply"
    artifact = {
        "id": artifact_id,
        "kind": "gaussian-splat",
        "format": "ply",
        "storage": {
            "type": "huggingface-bucket",
            "bucket": bucket,
            "path": remote_path,
        },
        "size_bytes": ply_path.stat().st_size,
        "sha256": sha256,
        "orientation": orientation,
        "provenance": {
            "repository": "KAFKA2306/AutoPhotogrammetry",
            "revision": source_revision,
            "run_id": run_id,
        },
    }
    if source_url:
        artifact["source_url"] = source_url
    if license_url:
        artifact["license_url"] = license_url
    return {"schema_version": 1, "artifacts": [artifact]}, artifact_id


def publish_run_splat(
    run_manifest_path: str | Path,
    *,
    bucket: str,
    hf_cache_hub_root: str | Path | None = None,
    source_revision: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    run_manifest_path = Path(run_manifest_path).expanduser().resolve()
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "success":
        raise ArtifactPublishError("run manifest is not a successful local reconstruction")
    try:
        splat = run_manifest["splatfacto"]
        ply_path = _resolve_run_artifact_path(run_manifest_path, str(splat["ply_path"]))
        expected_sha = splat["ply_sha256"]
        expected_size = splat["ply_size_bytes"]
    except KeyError as exc:
        raise ArtifactPublishError(
            f"run manifest is missing Gaussian Splat metadata: {exc}"
        ) from exc

    actual_sha = sha256_file(ply_path)
    actual_size = ply_path.stat().st_size
    if actual_sha != expected_sha or actual_size != expected_size:
        raise ArtifactPublishError("local PLY no longer matches the successful run manifest")

    transforms_path = _resolve_transforms_path(run_manifest_path, run_manifest)
    physical_up_path = _resolve_physical_up_path(run_manifest_path, run_manifest)
    orientation_path = run_manifest_path.parent / "orientation-evidence.json"
    try:
        orientation = write_orientation_evidence(
            transforms_path,
            ply_path,
            orientation_path,
            physical_up_path=physical_up_path,
        )
    except OrientationContractError as exc:
        raise ArtifactPublishError(f"orientation gate failed: {exc}") from exc
    if orientation["ply_sha256"] != expected_sha:
        raise ArtifactPublishError("orientation evidence was generated for a different PLY SHA-256")

    # Keep the artifact manifest portable: hashes are authoritative and local paths
    # are reduced to sidecar file names inside the run directory.
    orientation_for_manifest = copy.deepcopy(orientation)
    orientation_for_manifest["evidence_path"] = orientation_path.name
    physical = orientation_for_manifest.get("physical_up", {})
    if physical.get("status") == "accepted" and physical.get("evidence_path"):
        physical["evidence_path"] = Path(str(physical["evidence_path"])).name
    run_manifest["orientation"] = orientation_for_manifest

    revision = _recorded_revision(run_manifest, source_revision)
    registry = run_manifest.get("registry", {})
    license_info = registry.get("license") or {}
    source_url = registry.get("source_page")
    license_url = license_info.get("url") if isinstance(license_info, dict) else None
    run_id = f"{run_manifest.get('dataset', 'unknown')}:{run_manifest.get('started_at', 'unknown')}"
    artifact_manifest, artifact_id = build_artifact_manifest(
        dataset=str(run_manifest["dataset"]),
        ply_path=ply_path,
        bucket=bucket,
        source_revision=revision,
        run_id=run_id,
        orientation=orientation_for_manifest,
        source_url=source_url,
        license_url=license_url,
    )
    artifact_manifest_path = run_manifest_path.parent / "artifact-manifest.yaml"
    artifact_manifest_path.write_text(
        yaml.safe_dump(artifact_manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    command: Sequence[str] = [
        *_hf_cache_command(hf_cache_hub_root),
        "--manifest",
        str(artifact_manifest_path),
        "publish",
        str(ply_path),
        "--id",
        artifact_id,
    ]
    completed = runner(command, check=False, capture_output=True, text=True)
    try:
        publish_result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        publish_result = {"status": "FAILED", "error": "hf-cache-hub returned non-JSON output"}
        run_manifest["artifact_publish"] = publish_result
        write_json(run_manifest_path, run_manifest)
        raise ArtifactPublishError(publish_result["error"]) from exc

    if (
        completed.returncode != 0
        or publish_result.get("status") != "PUBLISHED"
        or publish_result.get("remote_verified") is not True
        or publish_result.get("sha256") != expected_sha
        or publish_result.get("size_bytes") != expected_size
    ):
        failure = {
            "status": "failed",
            "artifact_id": artifact_id,
            "local_sha256": expected_sha,
            "local_size_bytes": expected_size,
            "publish_result": publish_result,
        }
        run_manifest["artifact_publish"] = failure
        write_json(run_manifest_path, run_manifest)
        raise ArtifactPublishError(f"remote publish verification failed for {artifact_id}")

    physical_up = orientation_for_manifest.get("physical_up", {})
    success = {
        "status": "published",
        "artifact_id": artifact_id,
        "manifest_path": str(artifact_manifest_path),
        "remote_uri": publish_result["remote_uri"],
        "size_bytes": expected_size,
        "sha256": expected_sha,
        "source_revision": revision,
        "run_id": run_id,
        "orientation_status": orientation_for_manifest["status"],
        "orientation_scope": orientation_for_manifest["scope"],
        "physical_up_status": physical_up.get("status"),
        "physical_up_authority_type": physical_up.get("authority_type"),
        "orientation_evidence_sha256": orientation_for_manifest["evidence_sha256"],
        "remote_verified": True,
    }
    run_manifest["artifact_publish"] = success
    write_json(run_manifest_path, run_manifest)
    return success
