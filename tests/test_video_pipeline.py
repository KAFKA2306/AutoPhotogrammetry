import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_pipeline import (
    NerfstudioConfigurationError,
    VideoSource,
    extract_frames_command,
    frame_timestamp_records,
    gaussian_splat_export_command,
    nerfstudio_process_images_command,
    probe_video,
    run_splatfacto_export,
    scene_cut_times,
    select_video_frames,
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

    @patch("video_pipeline.run")
    def test_probe_video_parses_ffprobe_json(self, mocked_run):
        mocked_run.return_value.stdout = json.dumps(
            {"format": {"duration": "123.246", "format_name": "matroska,webm"}, "streams": []}
        )
        result = probe_video("source.webm")
        self.assertEqual(result["format"]["duration"], "123.246")
        self.assertEqual(result["format"]["format_name"], "matroska,webm")

    @patch("video_pipeline.run")
    def test_scene_cut_times_parses_showinfo(self, mocked_run):
        mocked_run.return_value.stderr = "x pts_time:12.5 y\nx pts_time:88.25 y\n"
        self.assertEqual(scene_cut_times("source.webm"), [12.5, 88.25])

    def test_select_video_frames_is_ordered_and_linear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(4):
                path = root / f"frame-{index}.jpg"
                path.write_bytes(str(index).encode())
                frames.append(path)

            sharpness = {frames[0]: 1.0, frames[1]: 0.0, frames[2]: 1.0, frames[3]: 1.0}
            similarity_calls = []

            def similarity(left, right):
                similarity_calls.append((Path(left), Path(right)))
                return 0.95 if Path(left) == frames[2] else 0.1

            result = select_video_frames(
                frames,
                root / "selected",
                sharpness_threshold=0.5,
                similarity_threshold=0.92,
                sharpness_fn=lambda path: sharpness[Path(path)],
                similarity_fn=similarity,
            )

            self.assertEqual(result["input"], 4)
            self.assertEqual(result["selected"], 2)
            self.assertEqual(result["rejected_blur"], 1)
            self.assertEqual(result["rejected_duplicate"], 1)
            self.assertEqual(len(similarity_calls), 2)

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
                {"format": {"duration": "123", "format_name": "matroska,webm"}},
                manifest_path,
                downloaded_at="2026-08-17T00:00:00+00:00",
            )
            self.assertEqual(manifest["video"]["size_bytes"], 11)
            self.assertEqual(len(manifest["video"]["sha256"]), 64)
            self.assertEqual(manifest["video"]["downloaded_at"], "2026-08-17T00:00:00+00:00")
            self.assertEqual(
                json.loads(manifest_path.read_text())["source"]["license"],
                "CC BY 3.0",
            )

    def test_splatfacto_runner_writes_auditable_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "processed data" / "images"
            data.mkdir(parents=True)
            (data / "frame 1.jpg").write_bytes(b"frame one")
            (data / "frame 2.jpg").write_bytes(b"frame two")

            def fake_run(command, **kwargs):
                cwd = Path(kwargs["cwd"])
                if command[1] == "splatfacto":
                    model = cwd / "outputs" / "example"
                    (model / "nerfstudio_models").mkdir(parents=True)
                    (model / "config.yml").write_text("method: splatfacto\n")
                    (model / "nerfstudio_models" / "step-000001.ckpt").write_bytes(b"checkpoint")
                    return subprocess.CompletedProcess(command, 0, "trained\n", "")
                if command[1] == "gaussian-splat":
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "splat.ply").write_bytes(b"ply data")
                    return subprocess.CompletedProcess(command, 0, "exported\n", "")
                raise AssertionError(command)

            with patch("video_pipeline._resolve_cli", side_effect=lambda name: Path("/fake") / name), \
                 patch("video_pipeline._package_version", side_effect=lambda name: {"nerfstudio": "1.2.3", "gsplat": "1.5.0"}[name]), \
                 patch("video_pipeline.subprocess.run", side_effect=fake_run):
                result = run_splatfacto_export(data.parent, root / "runs")

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["input"]["image_count"], 2)
            self.assertEqual(result["versions"], {"nerfstudio": "1.2.3", "gsplat": "1.5.0"})
            self.assertEqual(result["training"]["return_code"], 0)
            self.assertTrue(result["training"]["config_path"].endswith("config.yml"))
            self.assertTrue(result["training"]["checkpoint_path"].endswith(".ckpt"))
            self.assertEqual(result["export"]["return_code"], 0)
            self.assertEqual(result["output"]["size_bytes"], 8)
            self.assertEqual(len(result["output"]["sha256"]), 64)
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(manifest["output"]["ply_path"], "export/splat.ply")
            self.assertEqual(len(manifest["input"]["images"]), 2)

    def test_splatfacto_runner_records_training_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "frame.jpg").write_bytes(b"frame")

            with patch("video_pipeline._resolve_cli", side_effect=lambda name: Path("/fake") / name), \
                 patch("video_pipeline._package_version", side_effect=lambda name: "1.0"), \
                 patch(
                     "video_pipeline.subprocess.run",
                     return_value=subprocess.CompletedProcess(["ns-train"], 2, "", "training failed"),
                 ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_splatfacto_export(data, root / "runs")

            manifests = list((root / "runs").rglob("manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_phase"], "training")
            self.assertEqual(manifest["training"]["return_code"], 2)
            self.assertIsNone(manifest["export"])

    def test_splatfacto_runner_fails_when_cli_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            with patch("video_pipeline.shutil.which", return_value=None):
                with self.assertRaisesRegex(NerfstudioConfigurationError, "ns-train"):
                    run_splatfacto_export(data, Path(tmp) / "runs")


if __name__ == "__main__":
    unittest.main()
