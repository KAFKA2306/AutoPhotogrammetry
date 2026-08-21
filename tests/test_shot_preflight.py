import unittest

import numpy as np

from processing.shot_preflight import _essential_pose, select_shot, shot_intervals


class ShotPreflightTests(unittest.TestCase):
    def test_shot_intervals_are_bounded_sorted_and_drop_tiny_segments(self):
        shots = shot_intervals(20.0, [10.0, 5.0, 10.0, 19.5, -1.0, 30.0], minimum_seconds=2.0)
        self.assertEqual(
            [(shot["start_seconds"], shot["end_seconds"]) for shot in shots],
            [(0.0, 5.0), (5.0, 10.0), (10.0, 19.5)],
        )

    def test_known_two_view_translation_has_positive_pose_and_triangulation(self):
        points = np.array(
            [
                [-1.0, -0.5, 4.0],
                [-0.5, 0.2, 5.0],
                [0.0, -0.4, 6.0],
                [0.4, 0.6, 5.5],
                [0.8, -0.2, 7.0],
                [1.2, 0.5, 8.0],
                [-1.2, 0.4, 6.5],
                [0.2, 0.1, 4.5],
                [0.7, 0.3, 5.2],
                [-0.8, -0.1, 7.5],
            ],
            dtype=np.float64,
        )
        translation = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        first = points[:, :2] / points[:, 2:3]
        moved = points + translation
        second = moved[:, :2] / moved[:, 2:3]
        essential = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        pose = _essential_pose(essential, first, second)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertGreaterEqual(pose["pose_cheirality_ratio"], 0.9)
        self.assertAlmostEqual(pose["rotation_degrees"], 0.0, places=5)
        self.assertGreater(pose["triangulation_angle_degrees"], 0.0)
        self.assertAlmostEqual(np.linalg.norm(pose["translation_direction"]), 1.0, places=6)

    def test_selection_uses_measured_geometry_without_score_or_rank(self):
        shots = [
            {
                "id": "shot-0000",
                "duration_seconds": 100.0,
                "geometry": {
                    "geometry_pair_count": 2,
                    "essential_inlier_ratio_median": 0.7,
                    "triangulation_angle_degrees_median": 2.0,
                    "feature_overlap_ratio_median": 0.5,
                },
            },
            {
                "id": "shot-0001",
                "duration_seconds": 20.0,
                "geometry": {
                    "geometry_pair_count": 8,
                    "essential_inlier_ratio_median": 0.6,
                    "triangulation_angle_degrees_median": 1.5,
                    "feature_overlap_ratio_median": 0.4,
                },
            },
        ]
        self.assertEqual(select_shot(shots), "shot-0001")
        self.assertFalse(any("score" in shot or "rank" in shot for shot in shots))

    def test_selection_fails_closed_without_valid_geometry(self):
        self.assertIsNone(
            select_shot(
                [
                    {
                        "id": "shot-0000",
                        "duration_seconds": 300.0,
                        "geometry": {"geometry_pair_count": 0},
                    }
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
