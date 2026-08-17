import json
import unittest
from unittest.mock import patch

from processing.video import (
    extract_frames_command,
    frame_timestamp_records,
    probe_video,
    scene_cut_times,
)


class VideoTests(unittest.TestCase):
    def test_extract_command_preserves_spaces_and_supports_scaling(self):
        self.assertEqual(
            extract_frames_command(
                "source video.webm",
                "frames dir",
                fps=1 / 3,
                width=1024,
            ),
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                "source video.webm",
                "-vf",
                "fps=0.333333,scale=1024:-2",
                "-q:v",
                "2",
                "frames dir/frame-%06d.jpg",
            ],
        )

    def test_frame_timestamps_follow_sampling_rate(self):
        records = frame_timestamp_records(
            ["frame-000001.jpg", "frame-000002.jpg", "frame-000003.jpg"],
            fps=2,
        )
        self.assertEqual(
            records,
            [
                {"frame": "frame-000001.jpg", "source_time_seconds": 0.0},
                {"frame": "frame-000002.jpg", "source_time_seconds": 0.5},
                {"frame": "frame-000003.jpg", "source_time_seconds": 1.0},
            ],
        )

    @patch("processing.video.run")
    def test_probe_video_parses_ffprobe_json(self, mocked_run):
        mocked_run.return_value.stdout = json.dumps(
            {
                "format": {
                    "duration": "123.246",
                    "format_name": "matroska,webm",
                },
                "streams": [],
            }
        )
        result = probe_video("source.webm")
        self.assertEqual(result["format"]["duration"], "123.246")

    @patch("processing.video.run")
    def test_scene_cut_times_parses_showinfo(self, mocked_run):
        mocked_run.return_value.stderr = (
            "x pts_time:12.5 y\nx pts_time:88.25 y\n"
        )
        self.assertEqual(scene_cut_times("source.webm"), [12.5, 88.25])


if __name__ == "__main__":
    unittest.main()
