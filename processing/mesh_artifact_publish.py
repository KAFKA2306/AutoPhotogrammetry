from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from processing.artifact_publish import ArtifactPublishError, _hf_cache_command
from processing.provenance import sha256_file

SUPPORTED_SINGLE_FILE_FORMATS = {"glb", "stl"}


def _revision(value: str) -> str:
    value = value.strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ArtifactPublishError(
            "source_revision must be a full lowercase 40-character Git commit SHA"
        )
    return value


def _sha256(value: str, label: str) -> str:
    value = value.strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ArtifactPublishError(f"{label} must be a lowercase SHA-256")
    return value


def _verified_export(export_manifest_path: Path) -> tuple[dict, Path, str, int]:
    manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ArtifactPublishError("mesh export manifest is not successful")
    output = manifest.get("output") or {}
    output_format = str(output.get("format") or "").lower()
    if output_format not in SUPPORTED_SINGLE_FILE_FORMATS:
        raise ArtifactPublishError(
            "mesh artifact publish currently requires a single-file GLB or STL export"
        )
    files = output.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ArtifactPublishError("single-file mesh export must declare exactly one output file")
    declared = files[0]
    path = Path(str(declared.get("path") or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ArtifactPublishError(f"mesh artifact is missing or empty: {path}")
    actual_sha = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_sha != declared.get("sha256") or actual_size != declared.get("size_bytes"):
        raise ArtifactPublishError("local mesh no longer matches its export manifest")
    return manifest, path, actual_sha, actual_size


def build_mesh_artifact_manifest(
    export_manifest_path: str | Path,
    *,
    dataset: str,
    bucket: str,
    source_revision: str,
    source_gaussian_sha256: str,
    raw_mesh_sha256: str,
    source_url: str | None = None,
    license_url: str | None = None,
) -> tuple[dict, str, Path]:
    export_manifest_path = Path(export_manifest_path).expanduser().resolve()
    if not export_manifest_path.is_file():
        raise ArtifactPublishError(f"mesh export manifest is missing: {export_manifest_path}")
    export_manifest, mesh_path, mesh_sha, mesh_size = _verified_export(export_manifest_path)
    source_revision = _revision(source_revision)
    source_gaussian_sha256 = _sha256(source_gaussian_sha256, "source_gaussian_sha256")
    raw_mesh_sha256 = _sha256(raw_mesh_sha256, "raw_mesh_sha256")

    output_format = export_manifest["output"]["format"]
    parent_mesh_sha = _sha256(str(export_manifest["input"]["sha256"]), "parent mesh SHA-256")
    transform = export_manifest.get("transform") or {}
    metric_scale = transform.get("metric_scale") or {"status": "unverified", "unit": None}
    if metric_scale.get("status") not in {"accepted", "unverified", "unavailable"}:
        raise ArtifactPublishError("mesh export metric scale status is unsupported")

    artifact_id = f"autophotogrammetry/{dataset}/mesh/{output_format}"
    remote_path = f"autophotogrammetry/meshes/{dataset}/{mesh_sha}.{output_format}"
    artifact = {
        "id": artifact_id,
        "kind": "mesh",
        "format": output_format,
        "generated": False,
        "storage": {
            "type": "huggingface-bucket",
            "bucket": bucket,
            "path": remote_path,
        },
        "size_bytes": mesh_size,
        "sha256": mesh_sha,
        "provenance": {
            "repository": "KAFKA2306/AutoPhotogrammetry",
            "revision": source_revision,
            "source_gaussian_sha256": source_gaussian_sha256,
            "raw_mesh_sha256": raw_mesh_sha256,
            "parent_mesh_sha256": parent_mesh_sha,
            "export_manifest_sha256": sha256_file(export_manifest_path),
        },
        "geometry": {
            "coordinate_frame": transform.get("coordinate_frame", "unknown"),
            "metric_scale": metric_scale,
            "readback": export_manifest["output"].get("readback"),
        },
    }
    if source_url:
        artifact["source_url"] = source_url
    if license_url:
        artifact["license_url"] = license_url
    return {"schema_version": 1, "artifacts": [artifact]}, artifact_id, mesh_path


def publish_mesh_export(
    export_manifest_path: str | Path,
    *,
    dataset: str,
    bucket: str,
    source_revision: str,
    source_gaussian_sha256: str,
    raw_mesh_sha256: str,
    source_url: str | None = None,
    license_url: str | None = None,
    hf_cache_hub_root: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    export_manifest_path = Path(export_manifest_path).expanduser().resolve()
    manifest, artifact_id, mesh_path = build_mesh_artifact_manifest(
        export_manifest_path,
        dataset=dataset,
        bucket=bucket,
        source_revision=source_revision,
        source_gaussian_sha256=source_gaussian_sha256,
        raw_mesh_sha256=raw_mesh_sha256,
        source_url=source_url,
        license_url=license_url,
    )
    artifact = manifest["artifacts"][0]
    artifact_manifest_path = (
        export_manifest_path.parent / f"{artifact['format']}-artifact-manifest.yaml"
    )
    artifact_manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    command: Sequence[str] = [
        *_hf_cache_command(hf_cache_hub_root),
        "--manifest",
        str(artifact_manifest_path),
        "publish",
        str(mesh_path),
        "--id",
        artifact_id,
    ]
    completed = runner(command, check=False, capture_output=True, text=True)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ArtifactPublishError("hf-cache-hub returned non-JSON output") from exc
    if (
        completed.returncode != 0
        or result.get("status") != "PUBLISHED"
        or result.get("remote_verified") is not True
        or result.get("sha256") != artifact["sha256"]
        or result.get("size_bytes") != artifact["size_bytes"]
    ):
        raise ArtifactPublishError(f"remote publish verification failed for {artifact_id}")
    return {
        "status": "published",
        "artifact_id": artifact_id,
        "manifest_path": str(artifact_manifest_path),
        "remote_uri": result["remote_uri"],
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "source_gaussian_sha256": source_gaussian_sha256,
        "raw_mesh_sha256": raw_mesh_sha256,
        "remote_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a verified single-file mesh export.")
    parser.add_argument("export_manifest")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-gaussian-sha256", required=True)
    parser.add_argument("--raw-mesh-sha256", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--license-url")
    parser.add_argument("--hf-cache-hub-root")
    args = parser.parse_args()
    result = publish_mesh_export(
        args.export_manifest,
        dataset=args.dataset,
        bucket=args.bucket,
        source_revision=args.source_revision,
        source_gaussian_sha256=args.source_gaussian_sha256,
        raw_mesh_sha256=args.raw_mesh_sha256,
        source_url=args.source_url,
        license_url=args.license_url,
        hf_cache_hub_root=args.hf_cache_hub_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
