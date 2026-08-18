import hashlib
import tempfile
import unittest
from pathlib import Path

from processing.huejotzingo import _colmap_metrics, colmap_commands, ensure_source


class HuejotzingoTests(unittest.TestCase):
    def test_existing_verified_source_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.webm"
            source.write_bytes(b"verified")
            expected = hashlib.sha256(b"verified").hexdigest()
            self.assertEqual(
                ensure_source(source, url="https://invalid.example/", expected_sha256=expected),
                source,
            )

    def test_existing_source_with_wrong_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.webm"
            source.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                ensure_source(source, expected_sha256="0" * 64)

    def test_colmap_commands_use_one_sparse_workspace(self):
        commands = dict(colmap_commands("selected images", "output/colmap"))
        self.assertEqual(commands["colmap-feature-extractor"][0:2], ["colmap", "feature_extractor"])
        self.assertIn("selected images", commands["colmap-feature-extractor"])
        self.assertEqual(commands["colmap-sequential-matcher"][0:2], ["colmap", "sequential_matcher"])
        self.assertEqual(commands["colmap-mapper"][-2:], ["--output_path", "output/colmap/sparse"])

    def test_colmap_metrics_parse_analyzer_output(self):
        metrics = _colmap_metrics(
            "Registered images: 78\nPoints: 32782\nMean reprojection error: 0.370830px\n"
        )
        self.assertEqual(metrics["registered_images"], 78)
        self.assertEqual(metrics["points"], 32782)
        self.assertAlmostEqual(metrics["mean_reprojection_error_px"], 0.370830)


if __name__ == "__main__":
    unittest.main()
