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

    def _write_input(self, root: Path) -> Path:
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
        return image_dir

    def test_report_uses_existing_selection_and_preserves_quality_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root)

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
            self.assertEqual(report["backend"]["status"], "not_run")
            self.assertIsNone(report["backend"]["run_manifest_path"])
            self.assertEqual(report["decision"]["status"], "READY_FOR_BACKEND")
            self.assertIn("no reconstruction backend run", report["decision"]["reason"])
            self.assertIn("Run a reconstruction backend", report["decision"]["next_action"])
            self.assertFalse(report["quality_measurements"]["quality_guarantee"])
            self.assertIsNone(report["quality_measurements"]["registration_rate"])
            self.assertIsNone(report["quality_measurements"]["reprojection_error"])
            self.assertIsNone(report["quality_measurements"]["mesh_completeness"])
            self.assertEqual(selected_manifest["asset_id"], "artifact")
            self.assertEqual(len(selected_manifest["images"]), 1)
            self.assertIn("does not guarantee 3D reconstruction quality", report_html)
            self.assertIn('class="decision-surface"', report_html)
            self.assertIn('data-status="READY_FOR_BACKEND"', report_html)
            self.assertIn("Next action", report_html)
            self.assertIn("Evidence details", report_html)
            self.assertLess(report_html.index("Next action"), report_html.index("Evidence details"))

    def test_report_links_existing_backend_manifest_by_asset_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root)
            backend_manifest = root / "backend-run.json"
            write_json(
                backend_manifest,
                {
                    "dataset": "artifact",
                    "status": "success",
                    "return_code": 0,
                    "command": ["example-backend", "artifact"],
                },
            )

            result = build_readiness_report(
                "artifact",
                input_root=root / "input",
                output_root=root / "output",
                backend_run_manifest=backend_manifest,
            )
            report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
            selected_manifest = json.loads(
                Path(result["selected_manifest"]).read_text(encoding="utf-8")
            )

            self.assertEqual(report["backend"]["asset_id"], "artifact")
            self.assertEqual(report["backend"]["status"], "success")
            self.assertEqual(report["backend"]["return_code"], 0)
            self.assertEqual(report["backend"]["run_manifest_path"], str(backend_manifest))
            self.assertEqual(report["decision"]["status"], "REVIEW_READY")
            self.assertIn("successful backend run", report["decision"]["reason"])
            self.assertEqual(
                report["backend"]["run_manifest_sha256"],
                sha256_file(backend_manifest),
            )
            self.assertEqual(
                selected_manifest["backend_run_manifest"],
                str(backend_manifest),
            )

    def test_report_rejects_backend_manifest_for_another_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root)
            backend_manifest = root / "backend-run.json"
            write_json(backend_manifest, {"dataset": "other", "return_code": 0})

            with self.assertRaisesRegex(ValueError, "does not match asset_id"):
                build_readiness_report(
                    "artifact",
                    input_root=root / "input",
                    output_root=root / "output",
                    backend_run_manifest=backend_manifest,
                )

    def test_missing_provenance_requires_action_before_backend_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = self._write_input(root)
            manifest_path = image_dir / "manifest.json"
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            records[0].pop("source_page")
            write_json(manifest_path, records)

            result = build_readiness_report(
                "artifact",
                input_root=root / "input",
                output_root=root / "output",
            )
            report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
            report_html = Path(result["report_html"]).read_text(encoding="utf-8")

            self.assertEqual(
                report["selection"]["reason_counts"]["PROVENANCE_MISSING"], 1
            )
            self.assertEqual(report["decision"]["status"], "ACTION_REQUIRED")
            self.assertIn("missing required provenance", report["decision"]["reason"])
            self.assertIn("source_page", report["decision"]["next_action"])
            self.assertIn('data-status="ACTION_REQUIRED"', report_html)

    def test_report_ignores_stale_manifest_rows_when_counting_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = self._write_input(root)
            manifest_path = image_dir / "manifest.json"
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "source_page": "https://example.test/removed",
                    "image_url": "https://example.test/removed.jpg",
                    "local_path": str(image_dir / "removed.jpg"),
                    "sha256": records[0]["sha256"],
                    "width": 128,
                    "height": 128,
                    "content_type": "image/jpeg",
                }
            )
            write_json(manifest_path, records)

            result = build_readiness_report(
                "artifact",
                input_root=root / "input",
                output_root=root / "output",
            )
            report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))

            self.assertEqual(report["input"]["count"], 3)
            self.assertEqual(report["selection"]["reason_counts"]["EXACT_DUPLICATE"], 1)

    def test_report_rejects_manifest_hash_that_does_not_match_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = self._write_input(root)
            manifest_path = image_dir / "manifest.json"
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            records[0]["sha256"] = "0" * 64
            write_json(manifest_path, records)

            with self.assertRaisesRegex(ValueError, "SHA-256 does not match input file"):
                build_readiness_report(
                    "artifact",
                    input_root=root / "input",
                    output_root=root / "output",
                )

    def test_report_rejects_duplicate_manifest_filename_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = self._write_input(root)
            manifest_path = image_dir / "manifest.json"
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            records.append({**records[0], "local_path": str(Path("elsewhere") / "first.jpg")})
            write_json(manifest_path, records)

            with self.assertRaisesRegex(ValueError, "duplicate local_path filename"):
                build_readiness_report(
                    "artifact",
                    input_root=root / "input",
                    output_root=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
