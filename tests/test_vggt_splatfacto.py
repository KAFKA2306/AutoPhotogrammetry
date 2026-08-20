import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.backend_evaluation import (
    artifact_record,
    build_nerfstudio_dataset_contract,
    dataset_identity,
)
from processing.vggt_splatfacto import (
    prepare_vggt_nerfstudio_data,
    run_vggt_splatfacto,
)


class VggtSplatfactoTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source.webm"
        source.write_bytes(b"video")
        original = root / "original"
        images = original / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(5):
            image = images / f"image-{index:03d}.jpg"
            image.write_bytes(f"image-{index}".encode())
            frames.append(
                {
                    "file_path": f"images/{image.name}",
                    "transform_matrix": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
            )
        transforms = original / "transforms.json"
        transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
        dataset = build_nerfstudio_dataset_contract(source, transforms, holdout_count=1)
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        scene = root / "vggt" / "scene"
        vggt_images = scene / "images"
        sparse = scene / "sparse"
        vggt_images.mkdir(parents=True)
        sparse.mkdir(parents=True)
        by_hash = {record["sha256"]: record for record in dataset["frames"]}
        for index, (digest, _) in enumerate(sorted(by_hash.items()), start=1):
            source_image = next(
                path
                for path in images.iterdir()
                if __import__("processing.provenance", fromlist=["sha256_file"]).sha256_file(path)
                == digest
            )
            target = vggt_images / f"{index:04d}-{digest[:12]}.jpg"
            target.write_bytes(source_image.read_bytes())
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / name).write_bytes(name.encode())
        return source, dataset, dataset_path, transforms, scene

    def test_prepare_uses_existing_colmap_and_preserves_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, dataset, _, _, scene = self._fixture(root)

            def fake_run(command, *, cwd, stdout, stderr):
                self.assertIn("--skip-colmap", command)
                self.assertIn("--skip-image-processing", command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                images_dir = output_dir / "images"
                frames = [
                    {
                        "file_path": f"./images/{path.name}",
                        "transform_matrix": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                    for path in sorted(images_dir.iterdir())
                ]
                (output_dir / "transforms.json").write_text(
                    json.dumps({"frames": frames}), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with patch("processing.vggt_splatfacto._run_recorded", side_effect=fake_run):
                prepared = prepare_vggt_nerfstudio_data(dataset, scene, root / "prepared")

            self.assertEqual(prepared["return_code"], 0)
            self.assertEqual(len(prepared["images"]), 5)
            self.assertTrue(Path(prepared["transforms_path"]).is_file())

    def test_e2e_reuses_dataset_identity_and_writes_two_backend_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dataset, dataset_path, _, scene = self._fixture(root)
            prepared_root = root / "prepared-source"
            prepared_images = prepared_root / "images"
            prepared_images.mkdir(parents=True)
            for path in sorted((scene / "images").iterdir()):
                (prepared_images / path.name).write_bytes(path.read_bytes())
            prepared_frames = [
                {
                    "file_path": f"images/{path.name}",
                    "transform_matrix": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
                for path in sorted(prepared_images.iterdir())
            ]
            prepared_transforms = prepared_root / "transforms.json"
            prepared_transforms.write_text(
                json.dumps({"frames": prepared_frames}), encoding="utf-8"
            )

            baseline_artifact = root / "baseline.ply"
            baseline_artifact.write_bytes(b"baseline")
            baseline = {
                "schema_version": 2,
                "dataset_id": dataset_identity(dataset),
                "backend": {"name": "splatfacto-default", "upstream_revision": "ns-rev"},
                "command": ["ns-train", "splatfacto"],
                "config": {"variant": "default"},
                "return_code": 0,
                "status": "success",
                "failure_phase": None,
                "artifact": artifact_record(baseline_artifact, format="ply"),
                "metrics": {},
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            def fake_train(data, output_root, **kwargs):
                run_dir = Path(output_root) / "splatfacto" / "run"
                (run_dir / "export").mkdir(parents=True)
                ply = run_dir / "export" / "splat.ply"
                ply.write_bytes(b"vggt-ply")
                config = run_dir / "config.yml"
                config.write_text("config", encoding="utf-8")
                manifest = run_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "training": {
                        "command": ["ns-train", "splatfacto"],
                        "config_path": "config.yml",
                        "peak_gpu_memory_bytes": 123,
                    },
                    "output": {
                        "ply_path": "export/splat.ply",
                        "sha256": "b" * 64,
                        "size_bytes": 8,
                    },
                }

            def fake_eval(config, output_root, **kwargs):
                out = Path(output_root)
                out.mkdir(parents=True)
                manifest = out / "eval-manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "metrics": {"psnr": 24.0, "ssim": 0.88, "lpips": 0.12},
                    "render_count": 1,
                    "renders": [],
                }

            artifact_metrics = {
                "primitive_count": 10,
                "output_size_bytes": 8,
                "low_opacity_primitive_count": 1,
                "low_opacity_primitive_ratio": 0.1,
                "scale_anisotropy_above_10_count": 2,
                "scale_anisotropy_above_10_ratio": 0.2,
            }
            with (
                patch(
                    "processing.vggt_splatfacto.prepare_vggt_nerfstudio_data",
                    return_value={
                        "transforms_path": str(prepared_transforms),
                        "command": ["ns-process-data"],
                        "return_code": 0,
                    },
                ),
                patch(
                    "processing.vggt_splatfacto.verify_gpu_runtime",
                    return_value={"gpu": "test"},
                ),
                patch(
                    "processing.vggt_splatfacto.verify_research_environment",
                    return_value={"nerfstudio_revision": "ns-rev"},
                ),
                patch(
                    "processing.vggt_splatfacto.run_splatfacto_export",
                    side_effect=fake_train,
                ),
                patch(
                    "processing.vggt_splatfacto.run_nerfstudio_eval",
                    side_effect=fake_eval,
                ),
                patch(
                    "processing.vggt_splatfacto.gaussian_artifact_metrics",
                    return_value=artifact_metrics,
                ),
            ):
                result = run_vggt_splatfacto(
                    dataset_path,
                    source,
                    scene,
                    root / "out",
                    baseline_result_json=baseline_path,
                    iterations=30000,
                )

            self.assertEqual(result["dataset_id"], dataset_identity(dataset))
            self.assertEqual(result["metrics"]["psnr"], 24.0)
            self.assertEqual(result["metrics"]["peak_gpu_memory_bytes"], 123)
            comparison = json.loads(Path(result["comparison_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(comparison["results"]), 2)
            self.assertEqual(
                [row["backend"] for row in comparison["results"]],
                ["splatfacto-default", "vggt-colmap-splatfacto"],
            )

    def test_changed_frame_set_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, dataset_path, _, scene = self._fixture(root)
            prepared = root / "changed"
            images = prepared / "images"
            images.mkdir(parents=True)
            changed_frames = []
            for index in range(2):
                image = images / f"wrong-{index}.jpg"
                image.write_bytes(f"wrong-{index}".encode())
                changed_frames.append(
                    {
                        "file_path": f"images/{image.name}",
                        "transform_matrix": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                )
            transforms = prepared / "transforms.json"
            transforms.write_text(
                json.dumps({"frames": changed_frames}),
                encoding="utf-8",
            )
            with (
                patch(
                    "processing.vggt_splatfacto.prepare_vggt_nerfstudio_data",
                    return_value={"transforms_path": str(transforms)},
                ),
                patch("processing.vggt_splatfacto.verify_gpu_runtime", return_value={}),
                patch(
                    "processing.vggt_splatfacto.verify_research_environment",
                    return_value={"nerfstudio_revision": "ns-rev"},
                ),
                patch("processing.vggt_splatfacto.run_splatfacto_export") as train,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed the frozen dataset identity"):
                    run_vggt_splatfacto(dataset_path, source, scene, root / "out")
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
