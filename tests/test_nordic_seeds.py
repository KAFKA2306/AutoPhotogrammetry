import copy
import unittest

from processing.nordic_seeds import (
    EXPECTED_REGION_IDS,
    enabled_seed_categories,
    load_nordic_seeds,
    validate_nordic_seeds,
)


class NordicSeedTests(unittest.TestCase):
    def test_exact_eight_region_coverage_and_explicit_missing_aland(self) -> None:
        payload = load_nordic_seeds()
        self.assertEqual(tuple(region["id"] for region in payload["regions"]), EXPECTED_REGION_IDS)
        self.assertEqual(len(payload["regions"]), 8)
        aland = next(region for region in payload["regions"] if region["id"] == "aland")
        self.assertEqual(aland["status"], "missing")
        self.assertEqual(aland["seeds"], [])
        self.assertTrue(aland["missing_reason"])

    def test_enabled_seeds_are_exact_declared_categories(self) -> None:
        payload = load_nordic_seeds()
        seeds = enabled_seed_categories(payload)
        self.assertGreaterEqual(len(seeds), 7)
        self.assertEqual(len({seed["category_title"] for seed in seeds}), len(seeds))
        self.assertTrue(
            all(
                seed["source_url"].startswith("https://commons.wikimedia.org/wiki/Category:")
                for seed in seeds
            )
        )

    def test_duplicate_category_is_rejected(self) -> None:
        payload = load_nordic_seeds()
        duplicate = copy.deepcopy(payload["regions"][0]["seeds"][0])
        payload["regions"][1]["seeds"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate Commons seed category"):
            validate_nordic_seeds(payload)

    def test_missing_region_cannot_hide_a_guessed_seed(self) -> None:
        payload = load_nordic_seeds()
        aland = next(region for region in payload["regions"] if region["id"] == "aland")
        aland["seeds"] = [
            {
                "category_title": "Category:Drone videos from Åland",
                "seed_type": "drone",
                "enabled": True,
                "source_url": "https://commons.wikimedia.org/wiki/Category:Drone_videos_from_%C3%85land",
            }
        ]
        with self.assertRaisesRegex(ValueError, "Missing region cannot have active seeds"):
            validate_nordic_seeds(payload)


if __name__ == "__main__":
    unittest.main()
