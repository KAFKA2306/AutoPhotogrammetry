import unittest
from unittest.mock import patch

from processing.backend_evaluation import cleanup_metrics, empty_metrics, gaussian_artifact_metrics


class BackendArtifactMetricTests(unittest.TestCase):
    def test_empty_metrics_exposes_direct_artifact_fields_as_unmeasured(self):
        metrics = empty_metrics()
        for field in (
            "low_opacity_primitive_count",
            "low_opacity_primitive_ratio",
            "scale_anisotropy_above_10_count",
            "scale_anisotropy_above_10_ratio",
            "cleanup_removed_primitive_count",
            "cleanup_removed_primitive_ratio",
        ):
            self.assertIn(field, metrics)
            self.assertIsNone(metrics[field])

    def test_gaussian_artifact_metrics_maps_measured_ply_values(self):
        measured = {
            "primitive_count": 100,
            "size_bytes": 1234,
            "opacity": {"below_0_1_count": 20, "below_0_1_ratio": 0.2},
            "scale_anisotropy_ratio": {"above_10_count": 5, "above_10_ratio": 0.05},
        }
        with patch("processing.backend_evaluation.gaussian_ply_metrics", return_value=measured):
            metrics = gaussian_artifact_metrics("model.ply")
        self.assertEqual(metrics["primitive_count"], 100)
        self.assertEqual(metrics["output_size_bytes"], 1234)
        self.assertEqual(metrics["low_opacity_primitive_count"], 20)
        self.assertEqual(metrics["scale_anisotropy_above_10_ratio"], 0.05)

    def test_cleanup_metrics_reports_removed_primitives_only(self):
        before = {"primitive_count": 100}
        after = {"primitive_count": 75}
        with patch(
            "processing.backend_evaluation.gaussian_ply_metrics",
            side_effect=[before, after],
        ):
            metrics = cleanup_metrics("before.ply", "after.ply")
        self.assertEqual(metrics["cleanup_removed_primitive_count"], 25)
        self.assertEqual(metrics["cleanup_removed_primitive_ratio"], 0.25)

    def test_cleanup_metrics_rejects_added_primitives(self):
        with patch(
            "processing.backend_evaluation.gaussian_ply_metrics",
            side_effect=[{"primitive_count": 10}, {"primitive_count": 11}],
        ):
            with self.assertRaisesRegex(ValueError, "more primitives"):
                cleanup_metrics("before.ply", "after.ply")


if __name__ == "__main__":
    unittest.main()
