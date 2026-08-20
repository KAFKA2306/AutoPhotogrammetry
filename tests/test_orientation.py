from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from processing.orientation import (
    ALGORITHM_VERSION,
    OrientationContractError,
    build_orientation_evidence,
    validate_orientation_evidence,
    write_orientation_evidence,
)


class OrientationContractTest(unittest.TestCase):
    def _fixture(self, root: Path, *, orientation_override: str | None = None):
        transforms = root / "nerfstudio-data" / "transforms.json"
        transforms.parent.mkdir(parents=True)
        payload = {
            "frames": [
                {
                    "file_path": "images/frame-000001.jpg",
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                }
            ]
        }
        if orientation_override is not None:
            payload["orientation_override"] = orientation_override
        transforms.write_text(json.dumps(payload), encoding="utf-8")
        ply = root / "export" / "splat.ply"
        ply.parent.mkdir()
        ply.write_bytes(b"ply\nsynthetic-gaussian")
        return transforms, ply

    def test_default_nerfstudio_up_maps_to_canonical_y_up(self):
        with tempfile.TemporaryDirectory() as d:
            transforms, ply = self._fixture(Path(d))
            evidence = build_orientation_evidence(transforms, ply)
            self.assertEqual("accepted", evidence["status"])
            self.assertEqual("up", evidence["orientation_method"])
            self.assertEqual(ALGORITHM_VERSION, evidence["algorithm_version"])
            self.assertEqual([0.0, 1.0, 0.0], evidence["canonical_frame"]["up_vector"])
            self.assertEqual(
                [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                evidence["source_to_canonical"]["quaternion_xyzw"],
            )
            self.assertEqual(
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                evidence["consumer_application"]["quaternion_xyzw"],
            )
            validate_orientation_evidence(
                evidence, expected_ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest()
            )

    def test_vertical_is_explicitly_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            transforms, ply = self._fixture(Path(d), orientation_override="vertical")
            evidence = build_orientation_evidence(transforms, ply)
            self.assertEqual("accepted", evidence["status"])
            self.assertEqual("vertical", evidence["orientation_method"])

    def test_pca_is_preserved_as_review_required_and_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            transforms, ply = self._fixture(Path(d), orientation_override="pca")
            evidence = build_orientation_evidence(transforms, ply)
            self.assertEqual("review_required", evidence["status"])
            with self.assertRaisesRegex(OrientationContractError, "not accepted"):
                validate_orientation_evidence(evidence, expected_ply_sha256=evidence["ply_sha256"])

    def test_none_is_preserved_as_review_required_and_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            transforms, ply = self._fixture(Path(d), orientation_override="none")
            evidence = build_orientation_evidence(transforms, ply)
            self.assertEqual("review_required", evidence["status"])
            with self.assertRaisesRegex(OrientationContractError, "not accepted"):
                validate_orientation_evidence(evidence, expected_ply_sha256=evidence["ply_sha256"])

    def test_exact_ply_hash_is_a_gate(self):
        with tempfile.TemporaryDirectory() as d:
            transforms, ply = self._fixture(Path(d))
            evidence = build_orientation_evidence(transforms, ply)
            with self.assertRaisesRegex(OrientationContractError, "exact PLY"):
                validate_orientation_evidence(evidence, expected_ply_sha256="0" * 64)

    def test_evidence_is_persisted_and_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            transforms, ply = self._fixture(root)
            out = root / "orientation-evidence.json"
            evidence = write_orientation_evidence(transforms, ply, out)
            self.assertTrue(out.is_file())
            self.assertEqual(str(out.resolve()), evidence["evidence_path"])
            self.assertEqual(
                hashlib.sha256(out.read_bytes()).hexdigest(), evidence["evidence_sha256"]
            )

    def test_raw_ply_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            transforms, ply = self._fixture(root)
            before = ply.read_bytes()
            write_orientation_evidence(transforms, ply, root / "orientation-evidence.json")
            self.assertEqual(before, ply.read_bytes())


if __name__ == "__main__":
    unittest.main()
