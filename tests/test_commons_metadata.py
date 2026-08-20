from __future__ import annotations

import unittest
from unittest.mock import patch

from processing.commons_metadata import refresh_source_metadata


class CommonsMetadataTests(unittest.TestCase):
    def test_missing_rights_and_media_are_filled_from_api(self) -> None:
        source = {
            "id": "scene",
            "source_page": "https://commons.wikimedia.org/wiki/File:Scene.webm",
            "author": None,
            "license": {"name": None, "status": "needs_review", "url": None},
        }
        resolved = {
            "media_url": "https://upload.wikimedia.org/scene.webm",
            "source_sha1": "a" * 40,
            "source_size_bytes": 123,
            "mime": "video/webm",
            "author": '<a href="/wiki/User:Example">Example</a>',
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "resolved_via": "wikimedia-api",
        }
        with patch("processing.commons_metadata.resolve_media_url", return_value=resolved):
            updated = refresh_source_metadata(source)

        self.assertEqual(updated["media_url"], resolved["media_url"])
        self.assertEqual(updated["author"], "Example")
        self.assertEqual(updated["license"]["status"], "verified")
        self.assertEqual(updated["metadata_evidence"]["source_sha1"], "a" * 40)
        self.assertTrue(updated["metadata_evidence"]["download_url_available"])

    def test_existing_pinned_transcode_and_sha256_are_not_replaced(self) -> None:
        source = {
            "id": "scene",
            "source_page": "https://commons.wikimedia.org/wiki/File:Scene.webm",
            "media_url": "https://upload.wikimedia.org/transcoded/scene.1080p.webm",
            "sha256": "b" * 64,
            "author": "Example",
            "license": {
                "name": "CC0 1.0 Universal",
                "status": "verified",
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            },
        }
        updated = refresh_source_metadata(source)
        self.assertEqual(updated["media_url"], source["media_url"])
        self.assertEqual(updated["sha256"], source["sha256"])
        self.assertEqual(updated["license"], source["license"])

    def test_rate_limit_redirect_does_not_upgrade_unverified_rights(self) -> None:
        source = {
            "id": "scene",
            "source_page": "https://commons.wikimedia.org/wiki/File:Scene.webm",
            "author": None,
            "license": {"name": None, "status": "needs_review", "url": None},
        }
        resolved = {
            "media_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Scene.webm",
            "author": None,
            "license": None,
            "license_url": None,
            "resolved_via": "wikimedia-direct-redirect-after-429",
        }
        with patch("processing.commons_metadata.resolve_media_url", return_value=resolved):
            with self.assertRaisesRegex(ValueError, "metadata unavailable"):
                refresh_source_metadata(source)


if __name__ == "__main__":
    unittest.main()
