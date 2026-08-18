import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from processing.provenance import sha256_file, write_json
from processing.readiness_report import build_readiness_report


class ReadinessReportTests(unittest.TestCase):
    def _write_checkerboard(self, path: Path) -> None:
        pattern = np.indices((128, 128)).sum(axis=0) % 2
        array = np.repeat((pattern * 255).astype(np.uint8)[..., None], 3, axis=2)
        Image.fromarray(array, mode="RGB").save(path)

    def _write_flat(self, path: Path) -> None:
        array = np.full((128, 128, 3), 127, dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)

    def test_report_uses_existing_selection_and_preserves_quality_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "input" / "artifact" / "images"
            image_dir.mkdir(parents=True)
            first = image_dir / "first.jpg"
            duplicate = image_dir / "duplicate.jpg"
            blurry = image_dir / "blurry.jpg"
            self._write_checkerboard(first)
            duplicate.write_bytes(first.read_bytes())
            self._write_flat(blurry)

            records = []
            for path in (first, duplicate, blurry):
                records.append(
                    {
                        "source_page": "https://example.test/object",
                        "image_url": f"https://example.test/{path.name}",
                        "local_path": str(path),
                        "sha256": sha256_file(path),
                        "width": 128,
                        "height": 128,
                        "content_type": "image/jpeg",
                    }
                )
            write_json(image_dir / "manifest.json", records)

            result = build_readiness_report(
                "artifact",
                input_root=root / "input",
                output_root=root / "output",
                sharpness_threshold=0.0001,
                similarity_threshold=0.92,
            )

            report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
            selected_manifest = json.loads(
                Path(result["selected_manifest"]).read_text(encoding="utf-8")
            )
            report_html = Path(result["report_html"]).read_text(encoding="utf-8")

            self.assertEqual(report["asset_id"], "artifact")
            self.assertEqual(report["input"]["count"], 3)
            self.assertEqual(report["selection"]["selected_count"], 1)
            self.assertEqual(report["selection"]["reason_counts"]["LOW_SHARPNESS"], 1)
            self.assertEqual(report["selection"]["reason_counts"]["NEAR_DUPLICATE"], 1)
            self.assertEqual(report["selection"]["reason_counts"]["EXACT_DUPLICATE"], 1)
            self.assertEqual(report["selection"]["reason_counts"]["PROVENANCE_MISSING"], 0)
            self.assertEqual(report["provenance"]["covered_count"], 3)
            self.assertEqual(report["provenance"]["coverage_ratio"], 1.0)
            self.assertFalse(report["generated_views_used"])
            self.assertFalse(report["quality_measurements"]["quality_guarantee"])
            self.assertIsNone(report["quality_measurements"]["registration_rate"])
            self.assertIsNone(report["quality_measurements"]["reprojection_error"])
            self.assertIsNone(report["quality_measurements"]["mesh_completeness"])
            self.assertEqual(selected_manifest["asset_id"], "artifact")
            self.assertEqual(len(selected_manifest["images"]), 1)
            self.assertIn("does not guarantee 3D reconstruction quality", report_html)


if __name__ == "__main__":
    unittest.main()
