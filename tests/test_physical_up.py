from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from processing.physical_up import PhysicalUpContractError, load_physical_up_evidence


class PhysicalUpContractTest(unittest.TestCase):
    def _write(
        self,
        root: Path,
        *,
        authority_type: str = "imu_gravity",
        semantics: str = "up",
        source_vector: list[float] | None = None,
        matrix: list[list[float]] | None = None,
    ) -> Path:
        path = root / "physical-up-evidence.json"
        payload = {
            "schema_version": 1,
            "authority_type": authority_type,
            "authority_source": "fixture://telemetry.json",
            "authority_source_sha256": "a" * 64,
            "source_frame": "fixture-imu",
            "vector_semantics": semantics,
            "source_vector": source_vector or [0.0, 0.0, 1.0],
            "source_to_model_matrix3x3": matrix
            or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "angular_uncertainty_deg": 0.25,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_known_tilt_is_rotated_to_model_z(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            angle = math.radians(30.0)
            path = self._write(
                root,
                source_vector=[0.0, math.sin(angle), math.cos(angle)],
            )
            evidence = load_physical_up_evidence(path)
            self.assertEqual("accepted", evidence["status"])
            self.assertEqual("imu_gravity", evidence["authority_type"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), evidence["evidence_sha256"]
            )
            matrix = evidence["model_to_gravity_aligned"]["matrix3x3"]
            up = evidence["model_up_vector"]
            corrected = [sum(matrix[r][c] * up[c] for c in range(3)) for r in range(3)]
            self.assertAlmostEqual(0.0, corrected[0], places=7)
            self.assertAlmostEqual(0.0, corrected[1], places=7)
            self.assertAlmostEqual(1.0, corrected[2], places=7)

    def test_gravity_down_semantics_are_inverted(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), semantics="gravity_down", source_vector=[0.0, 0.0, -2.0])
            evidence = load_physical_up_evidence(path)
            self.assertEqual([0.0, 0.0, 1.0], evidence["model_up_vector"])

    def test_unknown_authority_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), authority_type="dominant_plane_pca")
            with self.assertRaisesRegex(PhysicalUpContractError, "authority_type"):
                load_physical_up_evidence(path)

    def test_zero_vector_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), source_vector=[0.0, 0.0, 0.0])
            with self.assertRaisesRegex(PhysicalUpContractError, "non-zero"):
                load_physical_up_evidence(path)

    def test_non_rotation_frame_transform_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(
                Path(d),
                matrix=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
            with self.assertRaisesRegex(PhysicalUpContractError, "proper rotation"):
                load_physical_up_evidence(path)

    def test_malformed_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["authority_source_sha256"] = "not-a-sha"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PhysicalUpContractError, "SHA-256"):
                load_physical_up_evidence(path)


if __name__ == "__main__":
    unittest.main()
