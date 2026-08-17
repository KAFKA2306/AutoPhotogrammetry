import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_pipeline import (
    VideoSource,
    extract_frames_command,
    gaussian_splat_export_command,
    nerfstudio_process_images_command,
    probe_video,
    scene_cut_times,
    splatfacto_train_command,
    write_source_manifest,
)


class VideoPipelineTests(unittest.TestCase):
    def test_command_builders_preserve_spaces(self):
        self.assertEqual(
            extract_frames_command("source video.webm", "frames dir", fps=3),
            [
                "ffmpeg", "-hide_banner", "-y", "-i", "source video.webm",
                "-vf", "fps=3", "-q:v", "2", "frames dir/frame-%06d.jpg",
            ],
        )
        self.assertEqual(
            nerfstudio_process_images_command("frames dir", "processed data"),
            [
                "ns-process-data", "images", "--data", "frames dir",
                "--output-dir", "processed data",
            ],
        )
        self.assertEqual(
            splatfacto_train_command("processed data"),
            ["ns-train", "splatfacto", "--data", "processed data"],
        )
        self.assertEqual(
            gaussian_splat_export_command("outputs/config.yml", "exports/splat"),
            [
                "ns-export", "gaussian-splat", "--load-config", "outputs/config.yml",
                "--output-dir", "exports/splat",
            ],
        )

    @patch("video_pipeline.run")
    def test_probe_video_parses_ffprobe_json(self, mocked_run):
        mocked_run.return_value.stdout = json.dumps(
            {"format": {"duration": "123.246"}, "streams": []}
        )
        result = probe_video("source.webm")
        self.assertEqual(result["format"]["duration"], "123.246")

    @patch("video_pipeline.run")
    def test_scene_cut_times_parses_showinfo(self, mocked_run):
        mocked_run.return_value.stderr = "x pts_time:12.5 y\nx pts_time:88.25 y\n"
        self.assertEqual(scene_cut_times("source.webm"), [12.5, 88.25])

    def test_source_manifest_hashes_exact_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.webm"
            video.write_bytes(b"video bytes")
            source = VideoSource(
                title="Example",
                source_page="https://example.org/file",
                media_url="https://example.org/file.webm",
                author="Author",
                license="CC BY 3.0",
                license_url="https://creativecommons.org/licenses/by/3.0/",
                target="Example target",
            )
            manifest_path = root / "manifest.json"
            manifest = write_source_manifest(
                video,
                source,
                {"format": {"duration": "123"}},
                manifest_path,
            )
            self.assertEqual(manifest["video"]["size_bytes"], 11)
            self.assertEqual(len(manifest["video"]["sha256"]), 64)
            self.assertEqual(
                json.loads(manifest_path.read_text())["source"]["license"],
                "CC BY 3.0",
            )


if __name__ == "__main__":
    unittest.main()
