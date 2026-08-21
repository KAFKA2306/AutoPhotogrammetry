import unittest

from processing.colmap_evaluation import aggregate_models, parse_model_analyzer


class ColmapEvaluationTests(unittest.TestCase):
    def test_model_analyzer_parser_captures_stage_c_fields(self):
        metrics = parse_model_analyzer(
            "Cameras: 1\n"
            "Images: 20\n"
            "Registered images: 19\n"
            "Points: 1771\n"
            "Observations: 5313\n"
            "Mean track length: 3.0\n"
            "Mean observations per image: 279.6316\n"
            "Mean reprojection error: 0.455544 px\n"
        )
        self.assertEqual(metrics["registered_images"], 19)
        self.assertEqual(metrics["points"], 1771)
        self.assertEqual(metrics["observations"], 5313)
        self.assertAlmostEqual(metrics["mean_track_length"], 3.0)
        self.assertAlmostEqual(metrics["mean_reprojection_error_px"], 0.455544)

    def test_aggregate_models_exposes_fragmentation_and_largest_model_quality(self):
        result = aggregate_models(
            [
                {
                    "model_path": "sparse/0",
                    "metrics": {
                        "registered_images": 15,
                        "points": 1000,
                        "observations": 3000,
                        "mean_track_length": 3.0,
                        "mean_reprojection_error_px": 0.4,
                    },
                },
                {
                    "model_path": "sparse/1",
                    "metrics": {
                        "registered_images": 4,
                        "points": 200,
                        "mean_track_length": 2.0,
                        "mean_reprojection_error_px": 0.7,
                    },
                },
            ],
            input_images=20,
        )
        self.assertEqual(result["registered_images"], 19)
        self.assertAlmostEqual(result["registration_ratio"], 0.95)
        self.assertEqual(result["submodel_count"], 2)
        self.assertAlmostEqual(result["largest_model_ratio"], 15 / 19)
        self.assertEqual(result["sparse_points"], 1000)
        self.assertEqual(result["largest_model_path"], "sparse/0")

    def test_aggregate_models_rejects_impossible_duplicate_registration(self):
        with self.assertRaisesRegex(ValueError, "registered 22 images from 20"):
            aggregate_models(
                [
                    {"model_path": "0", "metrics": {"registered_images": 12}},
                    {"model_path": "1", "metrics": {"registered_images": 10}},
                ],
                input_images=20,
            )


if __name__ == "__main__":
    unittest.main()
