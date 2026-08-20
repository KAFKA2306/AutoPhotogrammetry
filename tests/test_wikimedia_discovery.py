import unittest

from processing.wikimedia_discovery import (
    discover_from_seed_config,
    normalize_discovery,
    normalize_videoinfo,
    traverse_category,
)


class WikimediaDiscoveryTests(unittest.TestCase):
    def test_category_traversal_follows_pagination_subcategories_and_cycles(self) -> None:
        calls = []

        def fake_request(params):
            calls.append(dict(params))
            category = params["cmtitle"]
            continuation = params.get("cmcontinue")
            if category == "Category:Root" and continuation is None:
                return {
                    "query": {
                        "categorymembers": [
                            {"ns": 6, "type": "file", "title": "File:B.webm"},
                            {"ns": 14, "type": "subcat", "title": "Category:Child"},
                        ]
                    },
                    "continue": {"cmcontinue": "page-2", "continue": "-||"},
                }
            if category == "Category:Root" and continuation == "page-2":
                return {
                    "query": {
                        "categorymembers": [
                            {"ns": 6, "type": "file", "title": "File:A.webm"},
                            {"ns": 0, "type": "page", "title": "Article:Ignore"},
                        ]
                    }
                }
            if category == "Category:Child":
                return {
                    "query": {
                        "categorymembers": [
                            {"ns": 6, "type": "file", "title": "File:B.webm"},
                            {"ns": 6, "type": "file", "title": "File:C.webm"},
                            {"ns": 14, "type": "subcat", "title": "Category:Root"},
                        ]
                    }
                }
            raise AssertionError(params)

        result = traverse_category("Category:Root", request_json=fake_request)
        self.assertEqual(
            [item["canonical_title"] for item in result],
            ["File:A.webm", "File:B.webm", "File:C.webm"],
        )
        b = next(item for item in result if item["canonical_title"] == "File:B.webm")
        self.assertEqual(
            b["discovered_categories"],
            ["Category:Child", "Category:Root"],
        )
        self.assertEqual(len([call for call in calls if call["cmtitle"] == "Category:Root"]), 2)
        self.assertEqual(len([call for call in calls if call["cmtitle"] == "Category:Child"]), 1)

    def test_region_failure_does_not_erase_other_region(self) -> None:
        seed_config = {
            "schema_version": 1,
            "regions": [
                {
                    "id": "norway",
                    "display_name": "Norway",
                    "status": "configured",
                    "seeds": [
                        {
                            "category_title": "Category:Norway",
                            "seed_type": "drone",
                            "enabled": True,
                            "source_url": "https://commons.wikimedia.org/wiki/Category:Norway",
                        }
                    ],
                },
                {
                    "id": "sweden",
                    "display_name": "Sweden",
                    "status": "configured",
                    "seeds": [
                        {
                            "category_title": "Category:Sweden",
                            "seed_type": "drone",
                            "enabled": True,
                            "source_url": "https://commons.wikimedia.org/wiki/Category:Sweden",
                        }
                    ],
                },
                {"id": "finland", "display_name": "Finland", "status": "missing", "missing_reason": "fixture", "seeds": []},
                {"id": "denmark", "display_name": "Denmark", "status": "missing", "missing_reason": "fixture", "seeds": []},
                {"id": "iceland", "display_name": "Iceland", "status": "missing", "missing_reason": "fixture", "seeds": []},
                {"id": "greenland", "display_name": "Greenland", "status": "missing", "missing_reason": "fixture", "seeds": []},
                {"id": "faroe-islands", "display_name": "Faroe Islands", "status": "missing", "missing_reason": "fixture", "seeds": []},
                {"id": "aland", "display_name": "Åland", "status": "missing", "missing_reason": "fixture", "seeds": []},
            ],
        }

        def fake_request(params):
            if params["cmtitle"] == "Category:Sweden":
                raise RuntimeError("fixture API failure")
            return {
                "query": {
                    "categorymembers": [
                        {"ns": 6, "type": "file", "title": "File:Norway.webm"}
                    ]
                }
            }

        result = discover_from_seed_config(seed_config, request_json=fake_request)
        self.assertEqual(
            [item["canonical_title"] for item in result["candidates"]],
            ["File:Norway.webm"],
        )
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["failures"][0]["region_id"], "sweden")

    def test_videoinfo_normalizes_rights_and_marks_review_needed(self) -> None:
        discovery = {
            "canonical_title": "File:Example.webm",
            "regions": ["norway"],
            "discovered_categories": ["Category:Drone videos from Norway"],
            "discovery_paths": [["Category:Drone videos from Norway"]],
        }
        response = {
            "page": {
                "categories": [
                    {"title": "Category:License review needed"},
                    {"title": "Category:Videos"},
                ]
            },
            "videoinfo": {
                "url": "https://upload.wikimedia.org/example.webm",
                "size": 1234,
                "width": 3840,
                "height": 2160,
                "duration": 180.5,
                "sha1": "abc123",
                "mime": "video/webm",
                "mediatype": "VIDEO",
                "extmetadata": {
                    "Artist": {"value": "<b>Example Author</b>"},
                    "LicenseShortName": {"value": "CC BY 3.0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/3.0/"},
                },
            },
        }
        normalized = normalize_videoinfo(discovery, response)
        self.assertTrue(normalized["confirmed_video"])
        self.assertEqual(normalized["author"], "Example Author")
        self.assertEqual(normalized["duration_seconds"], 180.5)
        self.assertEqual(normalized["license"]["status"], "needs_review")
        self.assertEqual(normalized["source_sha1"], "abc123")

    def test_missing_rights_remain_unknown_and_non_video_is_filtered(self) -> None:
        discovery = {
            "schema_version": 1,
            "candidates": [
                {
                    "canonical_title": "File:Image.jpg",
                    "regions": ["norway"],
                    "discovered_categories": ["Category:Mixed"],
                    "discovery_paths": [["Category:Mixed"]],
                }
            ],
            "failures": [],
        }

        def fake_request(params):
            self.assertEqual(params["prop"], "videoinfo|categories")
            return {
                "query": {
                    "pages": [
                        {
                            "title": "File:Image.jpg",
                            "categories": [],
                            "videoinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/image.jpg",
                                    "size": 42,
                                    "sha1": "same-sha1",
                                    "mime": "image/jpeg",
                                    "mediatype": "BITMAP",
                                    "extmetadata": {},
                                }
                            ],
                        }
                    ]
                }
            }

        result = normalize_discovery(discovery, request_json=fake_request)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["metadata_failures"], [])

    def test_same_sha1_merges_duplicate_file_titles(self) -> None:
        discovery = {
            "schema_version": 1,
            "candidates": [
                {
                    "canonical_title": "File:Alias A.webm",
                    "regions": ["norway"],
                    "discovered_categories": ["Category:A"],
                    "discovery_paths": [["Category:A"]],
                },
                {
                    "canonical_title": "File:Alias B.webm",
                    "regions": ["sweden"],
                    "discovered_categories": ["Category:B"],
                    "discovery_paths": [["Category:B"]],
                },
            ],
            "failures": [],
        }

        def fake_request(params):
            title = params["titles"]
            return {
                "query": {
                    "pages": [
                        {
                            "title": title,
                            "categories": [],
                            "videoinfo": [
                                {
                                    "url": f"https://upload.wikimedia.org/{title}.webm",
                                    "size": 100,
                                    "sha1": "shared-sha1",
                                    "mime": "video/webm",
                                    "mediatype": "VIDEO",
                                    "extmetadata": {
                                        "Artist": {"value": "Author"},
                                        "LicenseShortName": {"value": "CC0"},
                                        "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0/"},
                                    },
                                }
                            ],
                        }
                    ]
                }
            }

        result = normalize_discovery(discovery, request_json=fake_request)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["regions"], ["norway", "sweden"])
        self.assertEqual(
            result["candidates"][0]["discovered_categories"],
            ["Category:A", "Category:B"],
        )


if __name__ == "__main__":
    unittest.main()
