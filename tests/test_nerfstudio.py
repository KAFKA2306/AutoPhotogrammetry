import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.nerfstudio import (
    NerfstudioConfigurationError,
    gaussian_splat_export_command,
    nerfstudio_process_images_command,
    run_splatfacto_export,
    splatfacto_train_command,
)


class NerfstudioTests(unittest.TestCase):
    def test_command_builders_preserve_spaces(self):
        self.assertEqual(
            nerfstudio_process_images_command("frames dir", "processed data"),
            [
                "ns-process-data",
                "images",
                "--data",
                "frames dir",
                "--output-dir",
                "processed data",
            ],
        )
        self.assertEqual(
            splatfacto_train_command("processed data"),
            ["ns-train", "splatfacto", "--data", "processed data"],
        )
        self.assertEqual(
            gaussian_splat_export_command(
                "outputs/config.yml",
                "exports/splat",
            ),
            [
                "ns-export",
                "gaussian-splat",
                "--load-config",
                "outputs/config.yml",
                "--output-dir",
                "exports/splat",
            ],
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
                    (
                        model
                        / "nerfstudio_models"
                        / "step-000001.ckpt"
                    ).write_bytes(b"checkpoint")
                    return subprocess.CompletedProcess(
                        command, 0, "trained\n", ""
                    )
                if command[1] == "gaussian-splat":
                    output_dir = Path(
                        command[command.index("--output-dir") + 1]
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "splat.ply").write_bytes(b"ply data")
                    return subprocess.CompletedProcess(
                        command, 0, "exported\n", ""
                    )
                raise AssertionError(command)

            with patch(
                "processing.nerfstudio._resolve_cli",
                side_effect=lambda name: Path("/fake") / name,
            ), patch(
                "processing.nerfstudio._package_version",
                side_effect=lambda name: {
                    "nerfstudio": "1.2.3",
                    "gsplat": "1.5.0",
                }[name],
            ), patch(
                "processing.nerfstudio.subprocess.run",
                side_effect=fake_run,
            ):
                result = run_splatfacto_export(
                    data.parent,
                    root / "runs",
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["input"]["image_count"], 2)
            self.assertEqual(
                result["versions"],
                {"nerfstudio": "1.2.3", "gsplat": "1.5.0"},
            )
            self.assertEqual(result["training"]["return_code"], 0)
            self.assertEqual(result["export"]["return_code"], 0)
            self.assertEqual(result["output"]["size_bytes"], 8)
            self.assertEqual(len(result["output"]["sha256"]), 64)
            manifest = json.loads(
                Path(result["manifest_path"]).read_text()
            )
            self.assertEqual(
                manifest["output"]["ply_path"],
                "export/splat.ply",
            )

    def test_splatfacto_runner_records_training_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "frame.jpg").write_bytes(b"frame")

            with patch(
                "processing.nerfstudio._resolve_cli",
                side_effect=lambda name: Path("/fake") / name,
            ), patch(
                "processing.nerfstudio._package_version",
                side_effect=lambda name: "1.0",
            ), patch(
                "processing.nerfstudio.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["ns-train"],
                    2,
                    "",
                    "training failed",
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_splatfacto_export(
                        data,
                        root / "runs",
                    )

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
            with patch(
                "processing.nerfstudio.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(
                    NerfstudioConfigurationError,
                    "ns-train",
                ):
                    run_splatfacto_export(
                        data,
                        Path(tmp) / "runs",
                    )


if __name__ == "__main__":
    unittest.main()
