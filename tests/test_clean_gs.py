import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from processing.clean_gs import (
    CLEAN_GS_LICENSE,
    CLEAN_GS_LICENSE_URL,
    CLEAN_GS_REVISION,
    _git_checkout_revision,
    clean_gs_command,
    run_clean_gs,
)


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

    def test_canonical_revision_is_exact_commit_and_license_is_pinned(self):
        self.assertEqual(len(CLEAN_GS_REVISION), 40)
        int(CLEAN_GS_REVISION, 16)
        self.assertEqual(CLEAN_GS_LICENSE, "MIT")
        self.assertIn(CLEAN_GS_REVISION, CLEAN_GS_LICENSE_URL)

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
            args = command[3:]
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
            args = command[3:]
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

    def test_runner_uses_canonical_revision_by_default_and_records_license(self):
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

            def fake_run(command, **kwargs):
                self._write_ply(output, 2)
                return subprocess.CompletedProcess(command, 0, "cleaned", "")

            checkout = {
                "checkout_root": str(root),
                "requested_revision": CLEAN_GS_REVISION,
                "resolved_revision": CLEAN_GS_REVISION,
                "head_revision": CLEAN_GS_REVISION,
            }
            with (
                patch("processing.clean_gs._git_checkout_revision", return_value=checkout) as revision_check,
                patch("processing.clean_gs.subprocess.run", side_effect=fake_run),
            ):
                result = run_clean_gs(
                    script_path=script,
                    scene="temple",
                    masks_dir=masks,
                    input_ply=source,
                    output_ply=output,
                    manifest_path=manifest,
                )

            revision_check.assert_called_once_with(script.resolve(), CLEAN_GS_REVISION)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["upstream_revision"], CLEAN_GS_REVISION)
            self.assertEqual(result["upstream_license"], "MIT")
            self.assertEqual(result["upstream_license_url"], CLEAN_GS_LICENSE_URL)
            self.assertEqual(result["upstream_checkout"], checkout)
            self.assertEqual(result["input"]["primitive_count"], 3)
            self.assertEqual(result["output"]["primitive_count"], 2)
            self.assertEqual(result["removed_primitive_count"], 1)
            self.assertEqual(len(result["masks"]), 1)
            self.assertTrue(manifest.is_file())

    def test_runner_allows_explicit_revision_override_but_verifies_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "clean-gs.py"
            script.write_text("# fixture\n", encoding="utf-8")
            masks = root / "masks"
            masks.mkdir()
            (masks / "000001.png").write_bytes(b"mask")
            source = root / "input.ply"
            output = root / "output.ply"
            self._write_ply(source, 2)
            explicit = "d" * 40
            checkout = {
                "checkout_root": str(root),
                "requested_revision": explicit,
                "resolved_revision": explicit,
                "head_revision": explicit,
            }

            def fake_run(command, **kwargs):
                self._write_ply(output, 1)
                return subprocess.CompletedProcess(command, 0, "cleaned", "")

            with (
                patch("processing.clean_gs._git_checkout_revision", return_value=checkout) as revision_check,
                patch("processing.clean_gs.subprocess.run", side_effect=fake_run),
            ):
                result = run_clean_gs(
                    script_path=script,
                    upstream_revision=explicit,
                    scene="temple",
                    masks_dir=masks,
                    input_ply=source,
                    output_ply=output,
                    manifest_path=root / "manifest.json",
                )
            revision_check.assert_called_once_with(script.resolve(), explicit)
            self.assertEqual(result["upstream_revision"], explicit)

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
                "requested_revision": CLEAN_GS_REVISION,
                "resolved_revision": CLEAN_GS_REVISION,
                "head_revision": CLEAN_GS_REVISION,
            }
            with patch("processing.clean_gs._git_checkout_revision", return_value=checkout):
                with self.assertRaisesRegex(ValueError, "semantic masks"):
                    run_clean_gs(
                        script_path=script,
                        scene="temple",
                        masks_dir=masks,
                        input_ply=source,
                        output_ply=root / "output.ply",
                        manifest_path=root / "manifest.json",
                    )


if __name__ == "__main__":
    unittest.main()
