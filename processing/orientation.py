from __future__ import annotations

import json
import math
from pathlib import Path

from processing.physical_up import (
    PhysicalUpContractError,
    load_physical_up_evidence,
    mat_mul,
    quaternion_multiply,
)
from processing.provenance import sha256_file, write_json

PINNED_NERFSTUDIO_REVISION = "50e0e3c70c775e89333256213363badbf074f29d"
PINNED_VRCHAT_RENDERER_REVISION = "f96c0117cba518ff84d059d36f16909b873e23aa"
ORIENTATION_SCHEMA_VERSION = 2
ALGORITHM_VERSION = "nerfstudio-basis-plus-external-physical-up-v3"
_SQRT_HALF = math.sqrt(0.5)
Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]
Quaternion = tuple[float, float, float, float]

SOURCE_TO_CANONICAL_MATRIX: Matrix = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
SOURCE_TO_CANONICAL_QUATERNION_XYZW: Quaternion = (-_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)

# The pinned VRChat importer performs a Y reflection after horizon alignment.
# This pre-reflection rotation maps model +Z to -Y so the mandatory reflection
# produces final +Y while rotating position, Gaussian covariance/rotation and
# spherical harmonics consistently.
VRCHAT_HORIZON_QUATERNION_XYZW: Quaternion = (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)
VRCHAT_HORIZON_MATRIX: Matrix = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
VRCHAT_POST_REFLECTION_MATRIX: Matrix = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class OrientationContractError(RuntimeError):
    pass


def _mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)


def _det3(matrix: Matrix) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _close_vector(actual: Vector, expected: Vector, tol: float = 1e-8) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(actual, expected, strict=True))


def _orientation_method(transforms: dict) -> str:
    override = transforms.get("orientation_override")
    if override is None:
        return "up"
    if not isinstance(override, str) or not override.strip():
        raise OrientationContractError(
            "transforms.json orientation_override must be a non-empty string when present"
        )
    return override.strip().lower()


def _physical_up_decision(method: str) -> dict:
    if method == "up":
        reason = (
            "Nerfstudio `up` aligns the mean camera-up vector to model +Z. "
            "SfM has a global rotation gauge, so this is a camera-orientation heuristic, "
            "not an independently observed gravity vector."
        )
    elif method == "vertical":
        reason = (
            "Nerfstudio `vertical` estimates an image-vertical 3D direction and uses camera-up "
            "for sign disambiguation. It is a geometric/camera heuristic, not an external gravity observation."
        )
    elif method == "pca":
        reason = "Nerfstudio `pca` aligns camera-center principal axes and does not establish physical gravity."
    else:
        reason = "Nerfstudio orientation is disabled; physical gravity is not established."
    return {
        "status": "review_required",
        "observable_from_sfm_alone": False,
        "authority": None,
        "reason": reason,
        "required_external_authority": [
            "IMU/gravity telemetry",
            "surveyed control points or known ground plane",
            "another independently auditable gravity reference",
        ],
    }


def _load_external_physical_up(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    try:
        return load_physical_up_evidence(path)
    except PhysicalUpContractError as exc:
        raise OrientationContractError(
            f"external physical-up evidence failed validation: {exc}"
        ) from exc


def _matrix_from_payload(value: object, label: str) -> Matrix:
    if not isinstance(value, list) or len(value) != 3:
        raise OrientationContractError(f"{label} must be a 3x3 matrix")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise OrientationContractError(f"{label} must be a 3x3 matrix")
        rows.append(tuple(float(component) for component in row))
    return tuple(rows)


def _quaternion_from_payload(value: object, label: str) -> Quaternion:
    if not isinstance(value, list) or len(value) != 4:
        raise OrientationContractError(f"{label} must contain four values")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def build_orientation_evidence(
    transforms_path: str | Path,
    ply_path: str | Path,
    *,
    nerfstudio_revision: str = PINNED_NERFSTUDIO_REVISION,
    physical_up_path: str | Path | None = None,
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
    external_physical_up = _load_external_physical_up(physical_up_path)
    basis_status = (
        "accepted"
        if method in {"up", "vertical"} or external_physical_up is not None
        else "review_required"
    )
    basis_reason = (
        "Nerfstudio model basis has a reproducible Unity conversion"
        if basis_status == "accepted"
        else f"orientation_method={method!r} lacks a physical-up authority for a usable upright frame"
    )

    source_to_canonical_matrix = SOURCE_TO_CANONICAL_MATRIX
    source_to_canonical_quaternion = SOURCE_TO_CANONICAL_QUATERNION_XYZW
    consumer_matrix = VRCHAT_HORIZON_MATRIX
    consumer_quaternion = VRCHAT_HORIZON_QUATERNION_XYZW
    physical_up = _physical_up_decision(method)
    scope = "coordinate_basis_only"
    derivation_method = "pinned Nerfstudio basis convention plus explicit consumer import transform"

    if external_physical_up is not None:
        correction = _matrix_from_payload(
            external_physical_up["model_to_gravity_aligned"]["matrix3x3"],
            "physical-up correction",
        )
        correction_quaternion = _quaternion_from_payload(
            external_physical_up["model_to_gravity_aligned"]["quaternion_xyzw"],
            "physical-up correction quaternion",
        )
        source_to_canonical_matrix = mat_mul(SOURCE_TO_CANONICAL_MATRIX, correction)
        source_to_canonical_quaternion = quaternion_multiply(
            SOURCE_TO_CANONICAL_QUATERNION_XYZW,
            correction_quaternion,
        )
        consumer_matrix = mat_mul(VRCHAT_HORIZON_MATRIX, correction)
        consumer_quaternion = quaternion_multiply(
            VRCHAT_HORIZON_QUATERNION_XYZW,
            correction_quaternion,
        )
        physical_up = external_physical_up
        scope = "coordinate_basis_plus_physical_up"
        derivation_method = (
            "pinned Nerfstudio basis convention plus validated external physical-up authority "
            "plus explicit consumer import transform"
        )

    evidence = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "status": basis_status,
        "scope": scope,
        "reason": basis_reason,
        "algorithm_version": ALGORITHM_VERSION,
        "nerfstudio_revision": nerfstudio_revision,
        "orientation_method": method,
        "source_frame": {
            "name": "nerfstudio-model-basis",
            "x_axis": "+model-x",
            "y_axis": "+model-y",
            "z_axis": "+model-z",
            "basis_up_vector": [0.0, 0.0, 1.0],
            "physical_gravity_claimed": external_physical_up is not None,
        },
        "canonical_frame": {
            "name": "unity-basis-y-up",
            "x_axis": "+right",
            "y_axis": "+basis-up",
            "z_axis": "+forward",
            "basis_up_vector": [0.0, 1.0, 0.0],
            "physical_gravity_claimed": external_physical_up is not None,
        },
        "source_to_canonical": {
            "matrix3x3": [list(row) for row in source_to_canonical_matrix],
            "quaternion_xyzw": list(source_to_canonical_quaternion),
            "pivot": [0.0, 0.0, 0.0],
        },
        "consumer_application": {
            "consumer": "MichaelMoroz/VRChatGaussianSplatting",
            "revision": PINNED_VRCHAT_RENDERER_REVISION,
            "mode": "horizon_alignment_pre_y_reflection",
            "quaternion_xyzw": list(consumer_quaternion),
            "matrix3x3": [list(row) for row in consumer_matrix],
            "pivot": [0.0, 0.0, 0.0],
            "mandatory_post_transform": "reflect-y",
            "representation_aware": ["position", "gaussian_rotation", "spherical_harmonics"],
        },
        "physical_up": physical_up,
        "transforms_sha256": sha256_file(transforms_file),
        "ply_sha256": sha256_file(ply_file),
        "derivation_method": derivation_method,
    }
    if basis_status == "accepted":
        validate_orientation_evidence(evidence, expected_ply_sha256=evidence["ply_sha256"])
    return evidence


def validate_orientation_evidence(evidence: dict, *, expected_ply_sha256: str) -> None:
    if evidence.get("schema_version") != ORIENTATION_SCHEMA_VERSION:
        raise OrientationContractError("unsupported orientation evidence schema_version")
    if evidence.get("ply_sha256") != expected_ply_sha256:
        raise OrientationContractError("orientation evidence does not match the exact PLY SHA-256")
    if evidence.get("status") != "accepted":
        raise OrientationContractError(
            f"orientation basis evidence is not accepted: {evidence.get('status')} ({evidence.get('reason')})"
        )
    scope = evidence.get("scope")
    if scope not in {"coordinate_basis_only", "coordinate_basis_plus_physical_up"}:
        raise OrientationContractError("orientation evidence scope is unsupported")
    if evidence.get("nerfstudio_revision") != PINNED_NERFSTUDIO_REVISION:
        raise OrientationContractError(
            "orientation evidence Nerfstudio revision is stale or unsupported"
        )
    if evidence.get("algorithm_version") != ALGORITHM_VERSION:
        raise OrientationContractError(
            "orientation evidence algorithm_version is stale or unsupported"
        )
    if evidence.get("canonical_frame", {}).get("name") != "unity-basis-y-up":
        raise OrientationContractError("orientation evidence canonical frame is unsupported")
    physical_up = evidence.get("physical_up", {})
    if physical_up.get("status") not in {"accepted", "review_required", "unavailable"}:
        raise OrientationContractError("physical_up status is missing or unsupported")
    if scope == "coordinate_basis_plus_physical_up" and physical_up.get("status") != "accepted":
        raise OrientationContractError(
            "physical-up composition scope requires accepted external authority"
        )

    canonical = _matrix_from_payload(
        evidence["source_to_canonical"]["matrix3x3"],
        "source_to_canonical",
    )
    consumer = _matrix_from_payload(
        evidence["consumer_application"]["matrix3x3"],
        "consumer_application",
    )
    if abs(_det3(canonical) - 1.0) > 1e-7 or abs(_det3(consumer) - 1.0) > 1e-7:
        raise OrientationContractError(
            "orientation rotation matrix must be a proper rotation with determinant +1"
        )

    model_up: Vector = (0.0, 0.0, 1.0)
    if physical_up.get("status") == "accepted":
        value = physical_up.get("model_up_vector")
        if not isinstance(value, list) or len(value) != 3:
            raise OrientationContractError("accepted physical_up requires model_up_vector")
        model_up = (float(value[0]), float(value[1]), float(value[2]))

    if not _close_vector(_mat_vec(canonical, model_up), (0.0, 1.0, 0.0)):
        raise OrientationContractError(
            "source_to_canonical does not map the authoritative model up to Unity +Y"
        )

    final_matrix = mat_mul(VRCHAT_POST_REFLECTION_MATRIX, consumer)
    if not _close_vector(_mat_vec(final_matrix, model_up), (0.0, 1.0, 0.0)):
        raise OrientationContractError(
            "consumer transform plus mandatory Y reflection does not produce final authoritative +Y"
        )

    quat = [float(v) for v in evidence["consumer_application"]["quaternion_xyzw"]]
    if len(quat) != 4 or abs(sum(v * v for v in quat) - 1.0) > 1e-7:
        raise OrientationContractError("consumer quaternion must be normalized")


def write_orientation_evidence(
    transforms_path: str | Path,
    ply_path: str | Path,
    output_path: str | Path,
    *,
    nerfstudio_revision: str = PINNED_NERFSTUDIO_REVISION,
    physical_up_path: str | Path | None = None,
) -> dict:
    evidence = build_orientation_evidence(
        transforms_path,
        ply_path,
        nerfstudio_revision=nerfstudio_revision,
        physical_up_path=physical_up_path,
    )
    output = Path(output_path).expanduser().resolve()
    write_json(output, evidence)
    evidence["evidence_path"] = str(output)
    evidence["evidence_sha256"] = sha256_file(output)
    return evidence
