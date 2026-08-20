import unittest
from unittest.mock import patch

from processing.production_batch import run_production_batch


class ProductionBatchTests(unittest.TestCase):
    def test_batch_failure_never_attempts_final_ready_manifest(self):
        with (
            patch(
                "processing.production_batch.run_all_videos",
                return_value={"status": "failed", "results": []},
            ),
            patch("processing.production_batch.build_final_exhibition_manifest") as finalizer,
        ):
            result = run_production_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_phase"], "batch")
        finalizer.assert_not_called()

    def test_full_batch_success_requires_final_twenty_manifest_success(self):
        with (
            patch(
                "processing.production_batch.run_all_videos",
                return_value={"status": "success", "results": [object()] * 20},
            ),
            patch(
                "processing.production_batch.build_final_exhibition_manifest",
                return_value={
                    "status": "ready",
                    "entry_count": 20,
                    "manifest_path": "output/final-exhibition-manifest.json",
                },
            ),
        ):
            result = run_production_batch()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["final_exhibition_manifest"]["entry_count"], 20)

    def test_finalizer_failure_is_not_relabelled_as_batch_success(self):
        with (
            patch(
                "processing.production_batch.run_all_videos",
                return_value={"status": "success", "results": [object()] * 20},
            ),
            patch(
                "processing.production_batch.build_final_exhibition_manifest",
                side_effect=ValueError("scene-20: PLY missing"),
            ),
        ):
            result = run_production_batch()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_phase"], "final-exhibition-manifest")
        self.assertIn("PLY missing", result["error"])


if __name__ == "__main__":
    unittest.main()
