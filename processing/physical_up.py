from __future__ import annotations

import json
import math
from pathlib import Path

from processing.provenance import sha256_file

PHYSICAL_UP_SCHEMA_VERSION = 1
ALLOWED_AUTHORITY_TYPES = {
    "imu_gravity",
    "surveyed_up_vector",
    "surveyed_ground_plane",
    "independent_gravity_reference",
}
ALLOWED_VECTOR_SEMANTICS = {"up", "gravity_down"}
Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]
Quaternion = tuple[float, float, float, float]


class PhysicalUpContractError(RuntimeError):
    pass


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _vector3(value: object, label: str) -> Vector:
    if not isinstance(value, list) or len(value) != 3 or not all(_finite(v) for v in value):
        raise PhysicalUpContractError(f"{label} must contain exactly three finite numbers")
    return tuple(float(v) for v in value)


def _matrix3(value: object, label: str) -> Matrix:
    if not isinstance(value, list) or len(value) != 3:
        raise PhysicalUpContractError(f"{label} must be a 3x3 matrix")
    rows = tuple(_vector3(row, f"{label} row") for row in value)
    return rows


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vector, right: Vector) -> Vector:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Vector) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vector, label: str) -> Vector:
    magnitude = _norm(value)
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise PhysicalUpContractError(f"{label} must have non-zero finite norm")
    return tuple(component / magnitude for component in value)


def _mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)


def mat_mul(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(sum(left[r][k] * right[k][c] for k in range(3)) for c in range(3)) for r in range(3)
    )


def _det3(matrix: Matrix) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_rotation_matrix(matrix: Matrix, label: str) -> None:
    if abs(_det3(matrix) - 1.0) > 1e-6:
        raise PhysicalUpContractError(f"{label} must be a proper rotation with determinant +1")
    for row in matrix:
        if abs(_dot(row, row) - 1.0) > 1e-6:
            raise PhysicalUpContractError(f"{label} rows must be unit length")
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(_dot(matrix[i], matrix[j])) > 1e-6:
                raise PhysicalUpContractError(f"{label} rows must be orthogonal")


def rotation_from_to(source: Vector, target: Vector) -> tuple[Matrix, Quaternion]:
    source_n = _normalize(source, "source direction")
    target_n = _normalize(target, "target direction")
    cosine = max(-1.0, min(1.0, _dot(source_n, target_n)))
    if cosine > 1.0 - 1e-12:
        return (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0, 1.0),
        )
    if cosine < -1.0 + 1e-12:
        candidates = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        basis = min(candidates, key=lambda axis: abs(_dot(source_n, axis)))
        axis = _normalize(_cross(source_n, basis), "180-degree rotation axis")
        x, y, z = axis
        matrix: Matrix = (
            (2.0 * x * x - 1.0, 2.0 * x * y, 2.0 * x * z),
            (2.0 * y * x, 2.0 * y * y - 1.0, 2.0 * y * z),
            (2.0 * z * x, 2.0 * z * y, 2.0 * z * z - 1.0),
        )
        return matrix, (x, y, z, 0.0)

    cross = _cross(source_n, target_n)
    sine = _norm(cross)
    axis = tuple(component / sine for component in cross)
    x, y, z = axis
    one_minus_cosine = 1.0 - cosine
    matrix = (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )
    half_angle = math.acos(cosine) * 0.5
    half_sine = math.sin(half_angle)
    quaternion = (
        x * half_sine,
        y * half_sine,
        z * half_sine,
        math.cos(half_angle),
    )
    return matrix, quaternion


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= 1e-12 or not math.isfinite(magnitude):
        raise PhysicalUpContractError("composed quaternion is invalid")
    return tuple(component / magnitude for component in value)  # type: ignore[return-value]


def load_physical_up_evidence(path: str | Path) -> dict:
    evidence_path = Path(path).expanduser().resolve()
    if not evidence_path.is_file():
        raise PhysicalUpContractError(f"physical-up evidence file is missing: {evidence_path}")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicalUpContractError(f"cannot read physical-up evidence: {exc}") from exc
    if not isinstance(payload, dict):
        raise PhysicalUpContractError("physical-up evidence must contain a JSON object")
    if payload.get("schema_version") != PHYSICAL_UP_SCHEMA_VERSION:
        raise PhysicalUpContractError("unsupported physical-up evidence schema_version")

    authority_type = payload.get("authority_type")
    if authority_type not in ALLOWED_AUTHORITY_TYPES:
        raise PhysicalUpContractError(f"unsupported physical-up authority_type: {authority_type!r}")
    authority_source = payload.get("authority_source")
    if not isinstance(authority_source, str) or not authority_source.strip():
        raise PhysicalUpContractError("physical-up authority_source must be a non-empty string")
    authority_sha = payload.get("authority_source_sha256")
    if (
        not isinstance(authority_sha, str)
        or len(authority_sha) != 64
        or any(character not in "0123456789abcdef" for character in authority_sha)
    ):
        raise PhysicalUpContractError(
            "authority_source_sha256 must be a lowercase SHA-256 hex digest"
        )

    semantics = payload.get("vector_semantics")
    if semantics not in ALLOWED_VECTOR_SEMANTICS:
        raise PhysicalUpContractError(f"unsupported vector_semantics: {semantics!r}")
    source_frame = payload.get("source_frame")
    if not isinstance(source_frame, str) or not source_frame.strip():
        raise PhysicalUpContractError("source_frame must be a non-empty string")
    source_vector = _normalize(
        _vector3(payload.get("source_vector"), "source_vector"), "source_vector"
    )
    if semantics == "gravity_down":
        source_up = tuple(-component for component in source_vector)
    else:
        source_up = source_vector

    source_to_model = _matrix3(
        payload.get("source_to_model_matrix3x3"), "source_to_model_matrix3x3"
    )
    _validate_rotation_matrix(source_to_model, "source_to_model_matrix3x3")
    model_up = _normalize(_mat_vec(source_to_model, source_up), "model up vector")
    correction_matrix, correction_quaternion = rotation_from_to(model_up, (0.0, 0.0, 1.0))
    _validate_rotation_matrix(correction_matrix, "model physical-up correction")
    corrected = _mat_vec(correction_matrix, model_up)
    if any(abs(a - b) > 1e-6 for a, b in zip(corrected, (0.0, 0.0, 1.0), strict=True)):
        raise PhysicalUpContractError("physical-up correction does not map model up to +Z")

    uncertainty = payload.get("angular_uncertainty_deg")
    if uncertainty is not None and (not _finite(uncertainty) or float(uncertainty) < 0.0):
        raise PhysicalUpContractError("angular_uncertainty_deg must be finite and non-negative")

    resolved = {
        "status": "accepted",
        "observable_from_sfm_alone": False,
        "authority_type": authority_type,
        "authority_source": authority_source.strip(),
        "authority_source_sha256": authority_sha,
        "source_frame": source_frame.strip(),
        "vector_semantics": semantics,
        "source_vector": list(source_vector),
        "source_to_model_matrix3x3": [list(row) for row in source_to_model],
        "model_up_vector": list(model_up),
        "model_to_gravity_aligned": {
            "matrix3x3": [list(row) for row in correction_matrix],
            "quaternion_xyzw": list(correction_quaternion),
        },
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
    }
    if uncertainty is not None:
        resolved["angular_uncertainty_deg"] = float(uncertainty)
    return resolved
