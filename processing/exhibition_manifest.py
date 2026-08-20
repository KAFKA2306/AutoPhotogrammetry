from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from processing.gaussian_ply import gaussian_ply_metrics
from processing.provenance import image_records, sha256_file, write_json
from processing.video_sources import load_video_registry

FINAL_EXHIBITION_COUNT = 20


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _required_text(value: object, *, field: str, scene_id: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {
        "unknown",
        "unverified",
        "needs_review",
        "none",
        "null",
    }:
        raise ValueError(f"{scene_id}: verified {field} is required")
    return text


def _license_record(source: Mapping, *, scene_id: str) -> dict:
    value = source.get("license") or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{scene_id}: license must be an object")
    return {
        "name": _required_text(value.get("name"), field="license.name", scene_id=scene_id),
        "url": _required_text(value.get("url"), field="license.url", scene_id=scene_id),
    }


def _resolve_existing_path(value: object, *, output_root: Path, scene_root: Path) -> Path:
    path = Path(str(value or ""))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        # Container manifests often preserve /workspace/output/... paths. The
        # portable fallback below re-roots the suffix at the caller's output_root.
        parts = list(path.parts)
        if "output" in parts:
            index = parts.index("output")
            candidates.append(output_root.joinpath(*parts[index + 1 :]))
    else:
        candidates.extend((scene_root / path, output_root / path))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    raise ValueError(f"referenced path does not exist: {value}")


def _splatfacto_manifest(
    outer: Mapping,
    *,
    output_root: Path,
    scene_root: Path,
) -> tuple[dict, Path]:
    splat = outer.get("splatfacto") or {}
    if not isinstance(splat, Mapping):
        raise ValueError(f"{scene_root.name}: splatfacto result is missing")
    manifest_path = _resolve_existing_path(
        splat.get("manifest_path"),
        output_root=output_root,
        scene_root=scene_root,
    )
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "success":
        raise ValueError(f"{scene_root.name}: Splatfacto manifest is not success")
    return manifest, manifest_path


def _ply_record(
    outer: Mapping,
    child: Mapping,
    child_manifest_path: Path,
    *,
    output_root: Path,
    scene_root: Path,
) -> dict:
    splat = outer.get("splatfacto") or {}
    child_output = child.get("output") or {}
    declared_path = splat.get("ply_path") or child_output.get("ply_path")
    if not declared_path:
        raise ValueError(f"{scene_root.name}: PLY path is missing")

    path = Path(str(declared_path))
    if path.is_absolute():
        ply = _resolve_existing_path(path, output_root=output_root, scene_root=scene_root)
    else:
        child_candidate = (child_manifest_path.parent / path).resolve()
        if child_candidate.is_file():
            ply = child_candidate
        else:
            ply = _resolve_existing_path(path, output_root=output_root, scene_root=scene_root)
    if not ply.is_file() or ply.stat().st_size <= 0:
        raise ValueError(f"{scene_root.name}: PLY is missing or empty: {ply}")

    expected_sha = str(splat.get("ply_sha256") or child_output.get("sha256") or "").lower()
    expected_size = splat.get("ply_size_bytes") or child_output.get("size_bytes")
    if len(expected_sha) != 64:
        raise ValueError(f"{scene_root.name}: PLY SHA-256 is missing")
    actual_sha = sha256_file(ply)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{scene_root.name}: PLY SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"{scene_root.name}: positive PLY size is required")
    if ply.stat().st_size != expected_size:
        raise ValueError(
            f"{scene_root.name}: PLY size mismatch: expected {expected_size}, got {ply.stat().st_size}"
        )

    # Parsing representation-level metrics is also the final readability gate.
    metrics = gaussian_ply_metrics(ply)
    try:
        relative = ply.relative_to(output_root)
        materialization_path = f"output/{relative.as_posix()}"
    except ValueError:
        materialization_path = str(ply)
    return {
        "path": materialization_path,
        "size_bytes": expected_size,
        "sha256": actual_sha,
        "primitive_count": metrics["primitive_count"],
    }


def _playback_record(source: Mapping, outer: Mapping, *, scene_id: str) -> dict:
    explicit = source.get("playback_url")
    if explicit:
        if "requires_untrusted_urls" not in source:
            raise ValueError(
                f"{scene_id}: explicit playback_url requires explicit requires_untrusted_urls"
            )
        return {
            "playback_url": str(explicit),
            "requires_untrusted_urls": bool(source["requires_untrusted_urls"]),
            "authority": "registry",
        }

    resolution = outer.get("source_resolution") or {}
    original_media = source.get("media_url") or (
        resolution.get("media_url") if isinstance(resolution, Mapping) else None
    )
    original_media = _required_text(
        original_media,
        field="original media URL for playback fallback",
        scene_id=scene_id,
    )
    return {
        "playback_url": original_media,
        "requires_untrusted_urls": True,
        "authority": "original-media-fallback",
    }


def build_final_exhibition_manifest(
    registry_path: str | Path = "sources/videos.json",
    output_root: str | Path = "output",
    *,
    expected_count: int = FINAL_EXHIBITION_COUNT,
    output_path: str | Path | None = None,
) -> dict:
    """Build the sole final-20 downstream handoff, failing on any missing real evidence."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    registry = load_video_registry(registry_path)
    sources = registry["videos"]
    if len(sources) != expected_count:
        raise ValueError(
            f"final exhibition requires exactly {expected_count} registry entries, got {len(sources)}"
        )
    ids = [str(source.get("id") or "") for source in sources]
    if any(not scene_id for scene_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("final exhibition registry requires unique non-empty ids")

    root = Path(output_root).expanduser().resolve()
    entries = []
    for order, source in enumerate(sources, start=1):
        scene_id = str(source["id"])
        scene_root = root / scene_id
        outer_path = scene_root / "manifest.json"
        if not outer_path.is_file():
            raise ValueError(f"{scene_id}: production manifest is missing: {outer_path}")
        outer = _read_json(outer_path)
        if outer.get("status") != "success":
            raise ValueError(f"{scene_id}: production manifest is not success")
        if str(outer.get("dataset") or "") != scene_id:
            raise ValueError(f"{scene_id}: production manifest dataset identity mismatch")

        source_page = _required_text(
            source.get("source_page"), field="source_page", scene_id=scene_id
        )
        author = _required_text(source.get("author"), field="author", scene_id=scene_id)
        license_record = _license_record(source, scene_id=scene_id)
        source_record = outer.get("source") or {}
        source_sha = str(
            source_record.get("sha256") if isinstance(source_record, Mapping) else ""
        ).lower()
        if len(source_sha) != 64:
            raise ValueError(f"{scene_id}: source video SHA-256 is missing")

        child, child_path = _splatfacto_manifest(
            outer,
            output_root=root,
            scene_root=scene_root,
        )
        ply = _ply_record(
            outer,
            child,
            child_path,
            output_root=root,
            scene_root=scene_root,
        )
        selected_dir = scene_root / "selected"
        selected = image_records(selected_dir)
        if not selected:
            raise ValueError(f"{scene_id}: selected frame evidence is missing")

        training = child.get("training") or {}
        export = child.get("export") or {}
        versions = child.get("versions") or {}
        if not isinstance(training, Mapping) or not isinstance(export, Mapping):
            raise ValueError(f"{scene_id}: train/export evidence is missing")
        train_command = training.get("command")
        export_command = export.get("command")
        if not isinstance(train_command, list) or not train_command:
            raise ValueError(f"{scene_id}: exact training command is missing")
        if not isinstance(export_command, list) or not export_command:
            raise ValueError(f"{scene_id}: exact export command is missing")
        if training.get("return_code") != 0 or export.get("return_code") != 0:
            raise ValueError(f"{scene_id}: train/export return code is not zero")

        playback = _playback_record(source, outer, scene_id=scene_id)
        probe = outer.get("probe") or {}
        entries.append(
            {
                "id": scene_id,
                "display_order": order,
                "title": str(source.get("title") or scene_id),
                "target": str(source.get("target") or source.get("title") or scene_id),
                "source_page": source_page,
                "author": author,
                "license": license_record,
                "source_video": {
                    "sha256": source_sha,
                    "duration_seconds": (
                        probe.get("duration_seconds") if isinstance(probe, Mapping) else None
                    ),
                    "width": probe.get("width") if isinstance(probe, Mapping) else None,
                    "height": probe.get("height") if isinstance(probe, Mapping) else None,
                },
                "selected_frames": selected,
                "reconstruction": {
                    "backend": "nerfstudio-splatfacto",
                    "nerfstudio_version": (
                        versions.get("nerfstudio") if isinstance(versions, Mapping) else None
                    ),
                    "gsplat_version": (
                        versions.get("gsplat") if isinstance(versions, Mapping) else None
                    ),
                    "training_command": list(train_command),
                    "export_command": list(export_command),
                },
                "ply": ply,
                **playback,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "ready",
        "entry_count": len(entries),
        "entries": entries,
        "contract": {
            "expected_count": expected_count,
            "all_entries_require_real_ply": True,
            "unknown_playback_urls_are_not_fabricated": True,
        },
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "final-exhibition-manifest.json"
    )
    write_json(destination, manifest)
    return {**manifest, "manifest_path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed exactly-20 AutoPhotogrammetry downstream exhibition manifest."
    )
    parser.add_argument("--registry", default="sources/videos.json")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--expected-count", type=int, default=FINAL_EXHIBITION_COUNT)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build_final_exhibition_manifest(
        args.registry,
        args.output_root,
        expected_count=args.expected_count,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
