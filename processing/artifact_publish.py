from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import yaml

from processing.provenance import sha256_file, write_json


class ArtifactPublishError(RuntimeError):
    pass


def git_revision(runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> str:
    result = runner(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ArtifactPublishError(f"git rev-parse did not return a full lowercase commit SHA: {revision!r}")
    return revision


def _hf_cache_command(hf_cache_hub_root: str | Path | None) -> list[str]:
    root = Path(hf_cache_hub_root or os.environ.get("HF_CACHE_HUB_ROOT", "")).expanduser()
    if not str(root):
        raise ArtifactPublishError("HF_CACHE_HUB_ROOT or --hf-cache-hub-root is required")
    script = root / "scripts" / "artifact_manager.py"
    if not script.is_file():
        raise ArtifactPublishError(f"hf-cache-hub artifact CLI not found: {script}")
    return [sys.executable, str(script)]


def build_artifact_manifest(
    *,
    dataset: str,
    ply_path: Path,
    bucket: str,
    source_revision: str,
    run_id: str,
    source_url: str | None = None,
    license_url: str | None = None,
) -> tuple[dict, str]:
    if not ply_path.is_file() or ply_path.stat().st_size == 0:
        raise ArtifactPublishError(f"Gaussian Splat PLY is missing or empty: {ply_path}")
    sha256 = sha256_file(ply_path)
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
        "provenance": {
            "repository": "KAFKA2306/AutoPhotogrammetry",
            "revision": source_revision,
            "source_path": str(ply_path),
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
    run_manifest_path = Path(run_manifest_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "success":
        raise ArtifactPublishError("run manifest is not a successful local reconstruction")
    try:
        splat = run_manifest["splatfacto"]
        ply_path = Path(splat["ply_path"])
        expected_sha = splat["ply_sha256"]
        expected_size = splat["ply_size_bytes"]
    except KeyError as exc:
        raise ArtifactPublishError(f"run manifest is missing Gaussian Splat metadata: {exc}") from exc

    actual_sha = sha256_file(ply_path)
    actual_size = ply_path.stat().st_size
    if actual_sha != expected_sha or actual_size != expected_size:
        raise ArtifactPublishError("local PLY no longer matches the successful run manifest")

    revision = source_revision or git_revision(runner)
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

    success = {
        "status": "published",
        "artifact_id": artifact_id,
        "manifest_path": str(artifact_manifest_path),
        "remote_uri": publish_result["remote_uri"],
        "size_bytes": expected_size,
        "sha256": expected_sha,
        "source_revision": revision,
        "run_id": run_id,
        "remote_verified": True,
    }
    run_manifest["artifact_publish"] = success
    write_json(run_manifest_path, run_manifest)
    return success
