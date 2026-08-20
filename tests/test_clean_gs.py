import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from processing.clean_gs import _git_checkout_revision, clean_gs_command, run_clean_gs


class CleanGsTests(unittest.TestCase):
    def _write_ply(self, path: Path, count: int) -> None:
        dtype = np.dtype(
            [
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        vertices = np.zeros(count, dtype=dtype)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {count}\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.tobytes())

    def test_command_uses_official_cli_flags(self):
        command = clean_gs_command(
            "clean-gs.py",
            scene="temple",
            masks_dir="masks",
            input_ply="input.ply",
            output_ply="output.ply",
            color_threshold=0.3,
            k_neighbors=5,
            neighbor_threshold=0.7,
        )
        self.assertEqual(command[:2], ["python", "clean-gs.py"])
        self.assertIn("--masks_dir", command)
        self.assertIn("--input_ply", command)
        self.assertIn("--output_ply", command)
        self.assertIn("--color_threshold", command)

    def test_checkout_revision_requires_requested_commit_at_head(self):
        script = Path("/tmp/clean-gs/clean-gs.py")
        head = "a" * 40

        def fake_run(command, **kwargs):
            args = command[4:]
            if args == ["rev-parse", "--show-toplevel"]:
                stdout = "/tmp/clean-gs\n"
            elif args == ["rev-parse", "HEAD"]:
                stdout = f"{head}\n"
            elif args == ["rev-parse", "--verify", "release^{commit}"]:
                stdout = f"{head}\n"
            else:
                self.fail(f"unexpected git command: {command}")
            return subprocess.CompletedProcess(command, 0, stdout, "")

        with patch("processing.clean_gs.subprocess.run", side_effect=fake_run):
            result = _git_checkout_revision(script, "release")

        self.assertEqual(result["head_revision"], head)
        self.assertEqual(result["resolved_revision"], head)
        self.assertEqual(result["requested_revision"], "release")

    def test_checkout_revision_rejects_mismatched_head(self):
        script = Path("/tmp/clean-gs/clean-gs.py")
        requested = "a" * 40
        head = "b" * 40

        def fake_run(command, **kwargs):
            args = command[4:]
            if args == ["rev-parse", "--show-toplevel"]:
                stdout = "/tmp/clean-gs\n"
            elif args == ["rev-parse", "HEAD"]:
                stdout = f"{head}\n"
            elif args == ["rev-parse", "--verify", f"{requested}^{{commit}}"]:
                stdout = f"{requested}\n"
            else:
                self.fail(f"unexpected git command: {command}")
            return subprocess.CompletedProcess(command, 0, stdout, "")

        with patch("processing.clean_gs.subprocess.run", side_effect=fake_run):
            with self.assertRaisesRegex(ValueError, "does not match"):
                _git_checkout_revision(script, requested)

    def test_runner_records_masks_and_primitive_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "clean-gs.py"
            script.write_text("# fixture\n", encoding="utf-8")
            masks = root / "masks"
            masks.mkdir()
            (masks / "000001.png").write_bytes(b"mask")
            source = root / "input.ply"
            output = root / "output.ply"
            manifest = root / "manifest.json"
            self._write_ply(source, 3)
            head = "d" * 40

            def fake_run(command, **kwargs):
                self._write_ply(output, 2)
                return subprocess.CompletedProcess(command, 0, "cleaned", "")

            checkout = {
                "checkout_root": str(root),
                "requested_revision": "deadbeef",
                "resolved_revision": head,
                "head_revision": head,
            }
            with (
                patch("processing.clean_gs._git_checkout_revision", return_value=checkout),
                patch("processing.clean_gs.subprocess.run", side_effect=fake_run),
            ):
                result = run_clean_gs(
                    script_path=script,
                    upstream_revision="deadbeef",
                    scene="temple",
                    masks_dir=masks,
                    input_ply=source,
                    output_ply=output,
                    manifest_path=manifest,
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["upstream_revision"], head)
            self.assertEqual(result["upstream_checkout"], checkout)
            self.assertEqual(result["input"]["primitive_count"], 3)
            self.assertEqual(result["output"]["primitive_count"], 2)
            self.assertEqual(result["removed_primitive_count"], 1)
            self.assertEqual(len(result["masks"]), 1)
            self.assertTrue(manifest.is_file())

    def test_runner_requires_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "clean-gs.py"
            script.write_text("# fixture\n", encoding="utf-8")
            masks = root / "masks"
            masks.mkdir()
            source = root / "input.ply"
            self._write_ply(source, 1)
            checkout = {
                "checkout_root": str(root),
                "requested_revision": "deadbeef",
                "resolved_revision": "d" * 40,
                "head_revision": "d" * 40,
            }
            with patch("processing.clean_gs._git_checkout_revision", return_value=checkout):
                with self.assertRaisesRegex(ValueError, "semantic masks"):
                    run_clean_gs(
                        script_path=script,
                        upstream_revision="deadbeef",
                        scene="temple",
                        masks_dir=masks,
                        input_ply=source,
                        output_ply=root / "output.ply",
                        manifest_path=root / "manifest.json",
                    )


if __name__ == "__main__":
    unittest.main()
