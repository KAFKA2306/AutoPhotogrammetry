from __future__ import annotations

import unittest

from processing.wikimedia_360 import (
    ROOT_CATEGORY,
    build_pool,
    candidate_record,
    projection_state,
    validate_pool,
)


class Wikimedia360Test(unittest.TestCase):
    def test_two_to_one_raster_does_not_prove_equirectangular(self) -> None:
        state = projection_state(
            {
                "resolution": [7680, 3840],
                "commons_categories": [ROOT_CATEGORY],
                "discovered_categories": [ROOT_CATEGORY],
            }
        )
        self.assertEqual(state["projection_type"], "unknown")
        self.assertTrue(state["projection_review_required"])
        self.assertFalse(state["equirectangular_processing_ready"])

    def test_eac_category_is_explicitly_classified_but_not_sent_to_equirectangular_path(
        self,
    ) -> None:
        state = projection_state(
            {
                "commons_categories": ["Category:EAC Video"],
                "discovered_categories": [ROOT_CATEGORY, "Category:EAC Video"],
            }
        )
        self.assertEqual(state["projection_type"], "eac")
        self.assertFalse(state["projection_review_required"])
        self.assertFalse(state["equirectangular_processing_ready"])

    def test_stage_a_is_rights_and_identity_gate_not_quality_score(self) -> None:
        candidate = candidate_record(
            {
                "canonical_title": "File:Moving 360.webm",
                "source_page": "https://commons.wikimedia.org/wiki/File:Moving_360.webm",
                "media_url": "https://upload.wikimedia.org/example.webm",
                "source_sha1": "a" * 40,
                "source_size_bytes": 1234,
                "mime": "video/webm",
                "media_type": "VIDEO",
                "confirmed_video": True,
                "width": 7680,
                "height": 3840,
                "duration_seconds": 231.0,
                "duration_authority": "test",
                "author": "Example Author",
                "license": {
                    "name": "CC BY 3.0",
                    "url": "https://creativecommons.org/licenses/by/3.0/",
                    "status": "verified",
                },
                "commons_categories": [ROOT_CATEGORY],
                "discovered_categories": [ROOT_CATEGORY],
                "discovery_paths": [[ROOT_CATEGORY]],
                "metadata_authority": "test",
            }
        )
        self.assertTrue(candidate["stage_a"]["eligible_for_projection_review"])
        self.assertEqual(candidate["projection_type"], "unknown")
        self.assertNotIn("score", candidate)
        self.assertEqual(candidate["camera_motion"], "unknown")
        self.assertEqual(candidate["static_scene"], "unknown")

    def test_build_pool_recurses_category_and_normalizes_metadata(self) -> None:
        root_file = "File:Walkthrough.webm"
        eac_file = "File:EAC sample.webm"

        def request_json(params):
            if params.get("list") == "categorymembers":
                category = params["cmtitle"]
                if category == ROOT_CATEGORY:
                    return {
                        "query": {
                            "categorymembers": [
                                {"ns": 6, "title": root_file, "type": "file"},
                                {"ns": 14, "title": "Category:EAC Video", "type": "subcat"},
                            ]
                        }
                    }
                if category == "Category:EAC Video":
                    return {
                        "query": {"categorymembers": [{"ns": 6, "title": eac_file, "type": "file"}]}
                    }
                raise AssertionError(category)

            title = params["titles"]
            categories = [ROOT_CATEGORY]
            if title == eac_file:
                categories.append("Category:EAC Video")
            return {
                "query": {
                    "pages": [
                        {
                            "title": title,
                            "categories": [{"title": value} for value in categories],
                            "videoinfo": [
                                {
                                    "url": f"https://upload.wikimedia.org/{title[5:]}",
                                    "sha1": ("b" if title == root_file else "c") * 40,
                                    "size": 4096,
                                    "width": 4096,
                                    "height": 2048,
                                    "duration": 60.0,
                                    "mime": "video/webm",
                                    "mediatype": "VIDEO",
                                    "commonmetadata": {},
                                    "extmetadata": {
                                        "Artist": {"value": "Example Author"},
                                        "LicenseShortName": {"value": "CC BY 4.0"},
                                        "LicenseUrl": {
                                            "value": "https://creativecommons.org/licenses/by/4.0/"
                                        },
                                    },
                                }
                            ],
                        }
                    ]
                }
            }

        pool = build_pool(request_json=request_json)
        self.assertEqual(pool["discovered_file_count"], 2)
        self.assertEqual(pool["candidate_count"], 2)
        by_title = {item["canonical_title"]: item for item in pool["candidates"]}
        self.assertEqual(by_title[root_file]["projection_type"], "unknown")
        self.assertEqual(by_title[eac_file]["projection_type"], "eac")
        validate_pool(pool)

    def test_validation_rejects_metadata_quality_ranking(self) -> None:
        pool = {
            "schema_version": 1,
            "root_category": ROOT_CATEGORY,
            "candidates": [
                {
                    "canonical_title": "File:Bad.webm",
                    "evaluation_stage": "metadata",
                    "stage_a": {},
                    "projection_type": "unknown",
                    "equirectangular_processing_ready": False,
                    "score": 0.99,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "quality ranking"):
            validate_pool(pool)


if __name__ == "__main__":
    unittest.main()
