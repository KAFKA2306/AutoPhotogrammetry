import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.exhibition_manifest import build_final_exhibition_manifest
from processing.provenance import sha256_file


class ExhibitionManifestTests(unittest.TestCase):
    def _source(self, index: int) -> dict:
        return {
            "id": f"scene-{index:02d}",
            "title": f"Scene {index:02d}",
            "target": f"Target {index:02d}",
            "source_page": f"https://example.test/source/{index}",
            "author": "Example Author",
            "license": {
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
        }

    def _registry(self, sources: list[dict]) -> dict:
        return {
            "schema_version": 2,
            "default": sources[0]["id"],
            "evaluation_policy": {
                "stages": {
                    "metadata": {},
                    "preflight": {},
                    "colmap": {},
                    "splat": {},
                }
            },
            "videos": sources,
        }

    def _write_scene(self, root: Path, source: dict) -> None:
        scene = root / source["id"]
        selected = scene / "selected"
        selected.mkdir(parents=True)
        (selected / "frame-000001.jpg").write_bytes(b"frame")

        run = scene / "runs" / "splatfacto" / "run"
        export = run / "export"
        export.mkdir(parents=True)
        ply = export / "splat.ply"
        ply.write_bytes(b"real-ply-bytes")
        child = {
            "status": "success",
            "versions": {"nerfstudio": "test-ns", "gsplat": "test-gsplat"},
            "training": {
                "command": ["ns-train", "splatfacto", "--data", "data"],
                "return_code": 0,
            },
            "export": {
                "command": ["ns-export", "gaussian-splat"],
                "return_code": 0,
            },
            "output": {
                "ply_path": "export/splat.ply",
                "size_bytes": ply.stat().st_size,
                "sha256": sha256_file(ply),
            },
        }
        child_path = run / "manifest.json"
        child_path.write_text(json.dumps(child), encoding="utf-8")
        outer = {
            "dataset": source["id"],
            "status": "success",
            "source_resolution": {
                "media_url": f"https://upload.wikimedia.org/example/{source['id']}.webm"
            },
            "source": {"sha256": "a" * 64},
            "probe": {"duration_seconds": 120.0, "width": 1920, "height": 1080},
            "splatfacto": {
                "manifest_path": str(child_path),
                "ply_path": str(ply),
                "ply_sha256": sha256_file(ply),
                "ply_size_bytes": ply.stat().st_size,
            },
        }
        (scene / "manifest.json").write_text(json.dumps(outer), encoding="utf-8")

    def test_exactly_twenty_real_entries_produce_one_ready_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [self._source(index) for index in range(1, 21)]
            for source in sources:
                self._write_scene(root, source)
            metrics = {"primitive_count": 123}
            with patch(
                "processing.exhibition_manifest.load_video_registry",
                return_value=self._registry(sources),
            ), patch(
                "processing.exhibition_manifest.gaussian_ply_metrics",
                return_value=metrics,
            ):
                result = build_final_exhibition_manifest(
                    "registry.json",
                    root,
                )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["entry_count"], 20)
            self.assertEqual(
                [entry["display_order"] for entry in result["entries"]],
                list(range(1, 21)),
            )
            self.assertTrue(all(entry["ply"]["sha256"] for entry in result["entries"]))
            self.assertTrue(
                all(entry["requires_untrusted_urls"] for entry in result["entries"])
            )
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_missing_twentieth_ply_fails_instead_of_emitting_partial_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [self._source(index) for index in range(1, 21)]
            for source in sources:
                self._write_scene(root, source)
            missing = (
                root
                / "scene-20"
                / "runs"
                / "splatfacto"
                / "run"
                / "export"
                / "splat.ply"
            )
            missing.unlink()
            with patch(
                "processing.exhibition_manifest.load_video_registry",
                return_value=self._registry(sources),
            ), patch(
                "processing.exhibition_manifest.gaussian_ply_metrics",
                return_value={"primitive_count": 1},
            ):
                with self.assertRaisesRegex(ValueError, "referenced path does not exist|PLY"):
                    build_final_exhibition_manifest("registry.json", root)
            self.assertFalse((root / "final-exhibition-manifest.json").exists())

    def test_registry_count_must_be_exactly_twenty(self):
        sources = [self._source(index) for index in range(1, 20)]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "processing.exhibition_manifest.load_video_registry",
            return_value=self._registry(sources),
        ):
            with self.assertRaisesRegex(ValueError, "exactly 20"):
                build_final_exhibition_manifest("registry.json", tmp)

    def test_explicit_playback_url_requires_explicit_trust_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [self._source(index) for index in range(1, 21)]
            sources[0]["playback_url"] = "https://example.test/video.mp4"
            for source in sources:
                self._write_scene(root, source)
            with patch(
                "processing.exhibition_manifest.load_video_registry",
                return_value=self._registry(sources),
            ), patch(
                "processing.exhibition_manifest.gaussian_ply_metrics",
                return_value={"primitive_count": 1},
            ):
                with self.assertRaisesRegex(ValueError, "requires_untrusted_urls"):
                    build_final_exhibition_manifest("registry.json", root)


if __name__ == "__main__":
    unittest.main()
