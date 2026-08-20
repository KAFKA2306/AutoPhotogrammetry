from __future__ import annotations

import json
import math
from pathlib import Path

from processing.provenance import sha256_file, write_json

PINNED_NERFSTUDIO_REVISION = "50e0e3c70c775e89333256213363badbf074f29d"
PINNED_VRCHAT_RENDERER_REVISION = "f96c0117cba518ff84d059d36f16909b873e23aa"
ORIENTATION_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "nerfstudio-z-up-to-unity-y-up-v1"
_SQRT_HALF = math.sqrt(0.5)

# Nerfstudio model/world coordinates are +X right, +Y back, +Z up.
# A pure canonical basis rotation Rx(-90 deg) maps +Z(up) -> +Y(up)
# and +Y(back) -> -Z(back), i.e. +Z is forward in the target semantic frame.
SOURCE_TO_CANONICAL_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
SOURCE_TO_CANONICAL_QUATERNION_XYZW = (-_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)

# The pinned VRChat importer always performs a Y reflection AFTER horizon
# alignment. Therefore the pre-reflection horizon quaternion must map source
# +Z to -Y so the mandatory reflection produces final +Y.
VRCHAT_HORIZON_QUATERNION_XYZW = (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)
VRCHAT_HORIZON_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
VRCHAT_POST_REFLECTION_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class OrientationContractError(RuntimeError):
    pass


def _mat_vec(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)  # type: ignore[return-value]


def _mat_mul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[r][k] * right[k][c] for k in range(3)) for c in range(3))
        for r in range(3)
    )


def _det3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _close_vector(actual: tuple[float, float, float], expected: tuple[float, float, float], tol: float = 1e-9) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(actual, expected, strict=True))


def _orientation_method(transforms: dict) -> str:
    override = transforms.get("orientation_override")
    if override is None:
        return "up"
    if not isinstance(override, str) or not override.strip():
        raise OrientationContractError("transforms.json orientation_override must be a non-empty string when present")
    return override.strip().lower()


def build_orientation_evidence(
    transforms_path: str | Path,
    ply_path: str | Path,
    *,
    nerfstudio_revision: str = PINNED_NERFSTUDIO_REVISION,
) -> dict:
    transforms_file = Path(transforms_path).expanduser().resolve()
    ply_file = Path(ply_path).expanduser().resolve()
    if not transforms_file.is_file():
        raise OrientationContractError(f"Nerfstudio transforms.json is missing: {transforms_file}")
    if not ply_file.is_file() or ply_file.stat().st_size <= 0:
        raise OrientationContractError(f"Gaussian Splat PLY is missing or empty: {ply_file}")
    if nerfstudio_revision != PINNED_NERFSTUDIO_REVISION:
        raise OrientationContractError(
            "orientation contract is pinned to Nerfstudio revision "
            f"{PINNED_NERFSTUDIO_REVISION}; got {nerfstudio_revision}"
        )

    try:
        transforms = json.loads(transforms_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrientationContractError(f"cannot read Nerfstudio transforms.json: {exc}") from exc
    if not isinstance(transforms, dict):
        raise OrientationContractError("Nerfstudio transforms.json must contain a JSON object")

    method = _orientation_method(transforms)
    accepted_methods = {"up", "vertical"}
    status = "accepted" if method in accepted_methods else "review_required"
    reason = (
        "pinned Nerfstudio parser orients the reconstruction to +Z up before training"
        if status == "accepted"
        else f"orientation_method={method!r} does not establish the +Z-up gravity contract"
    )

    evidence = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "algorithm_version": ALGORITHM_VERSION,
        "nerfstudio_revision": nerfstudio_revision,
        "orientation_method": method,
        "source_frame": {
            "name": "nerfstudio-model",
            "x_axis": "+right",
            "y_axis": "+back",
            "z_axis": "+up",
            "up_vector": [0.0, 0.0, 1.0],
        },
        "canonical_frame": {
            "name": "unity-semantic-y-up",
            "x_axis": "+right",
            "y_axis": "+up",
            "z_axis": "+forward",
            "up_vector": [0.0, 1.0, 0.0],
        },
        "source_to_canonical": {
            "matrix3x3": [list(row) for row in SOURCE_TO_CANONICAL_MATRIX],
            "quaternion_xyzw": list(SOURCE_TO_CANONICAL_QUATERNION_XYZW),
            "pivot": [0.0, 0.0, 0.0],
        },
        "consumer_application": {
            "consumer": "MichaelMoroz/VRChatGaussianSplatting",
            "revision": PINNED_VRCHAT_RENDERER_REVISION,
            "mode": "horizon_alignment_pre_y_reflection",
            "quaternion_xyzw": list(VRCHAT_HORIZON_QUATERNION_XYZW),
            "matrix3x3": [list(row) for row in VRCHAT_HORIZON_MATRIX],
            "pivot": [0.0, 0.0, 0.0],
            "mandatory_post_transform": "reflect-y",
            "representation_aware": ["position", "gaussian_rotation", "spherical_harmonics"],
        },
        "transforms_sha256": sha256_file(transforms_file),
        "ply_sha256": sha256_file(ply_file),
        "derivation_method": "pinned Nerfstudio coordinate convention plus explicit consumer import transform",
    }
    if status == "accepted":
        validate_orientation_evidence(evidence, expected_ply_sha256=evidence["ply_sha256"])
    return evidence


def validate_orientation_evidence(evidence: dict, *, expected_ply_sha256: str) -> None:
    if evidence.get("schema_version") != ORIENTATION_SCHEMA_VERSION:
        raise OrientationContractError("unsupported orientation evidence schema_version")
    if evidence.get("ply_sha256") != expected_ply_sha256:
        raise OrientationContractError("orientation evidence does not match the exact PLY SHA-256")
    if evidence.get("status") != "accepted":
        raise OrientationContractError(
            f"orientation evidence is not accepted: {evidence.get('status')} ({evidence.get('reason')})"
        )
    if evidence.get("nerfstudio_revision") != PINNED_NERFSTUDIO_REVISION:
        raise OrientationContractError("orientation evidence Nerfstudio revision is stale or unsupported")
    if evidence.get("algorithm_version") != ALGORITHM_VERSION:
        raise OrientationContractError("orientation evidence algorithm_version is stale or unsupported")

    canonical = tuple(tuple(float(v) for v in row) for row in evidence["source_to_canonical"]["matrix3x3"])
    consumer = tuple(tuple(float(v) for v in row) for row in evidence["consumer_application"]["matrix3x3"])
    if abs(_det3(canonical) - 1.0) > 1e-9 or abs(_det3(consumer) - 1.0) > 1e-9:
        raise OrientationContractError("orientation rotation matrix must be a proper rotation with determinant +1")
    if not _close_vector(_mat_vec(canonical, (0.0, 0.0, 1.0)), (0.0, 1.0, 0.0)):
        raise OrientationContractError("source_to_canonical does not map Nerfstudio +Z up to canonical +Y up")

    final_matrix = _mat_mul(VRCHAT_POST_REFLECTION_MATRIX, consumer)
    if not _close_vector(_mat_vec(final_matrix, (0.0, 0.0, 1.0)), (0.0, 1.0, 0.0)):
        raise OrientationContractError("consumer transform plus mandatory Y reflection does not produce final +Y up")

    quat = [float(v) for v in evidence["consumer_application"]["quaternion_xyzw"]]
    if len(quat) != 4 or abs(sum(v * v for v in quat) - 1.0) > 1e-9:
        raise OrientationContractError("consumer quaternion must be normalized")


def write_orientation_evidence(
    transforms_path: str | Path,
    ply_path: str | Path,
    output_path: str | Path,
    *,
    nerfstudio_revision: str = PINNED_NERFSTUDIO_REVISION,
) -> dict:
    evidence = build_orientation_evidence(
        transforms_path,
        ply_path,
        nerfstudio_revision=nerfstudio_revision,
    )
    output = Path(output_path).expanduser().resolve()
    write_json(output, evidence)
    evidence["evidence_path"] = str(output)
    evidence["evidence_sha256"] = sha256_file(output)
    return evidence
