import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photogrammetry import (
    BackendConfig,
    BackendConfigurationError,
    build_command,
    resolve_executable,
    run_backend,
)


class PhotogrammetryRunnerTests(unittest.TestCase):
    def test_commands_are_argument_arrays_and_preserve_spaces(self):
        image_dir = Path("input images")
        output_dir = Path("output models")
        commands = {
            "meshroom": ["tool", "--input", str(image_dir), "--output", str(output_dir)],
            "visualsfm": ["tool", str(image_dir), str(output_dir)],
            "colmap": [
                "tool", "automatic_reconstructor", "--image_path", str(image_dir),
                "--workspace_path", str(output_dir),
            ],
        }
        for backend, expected in commands.items():
            with self.subTest(backend=backend):
                self.assertEqual(build_command(backend, "tool", image_dir, output_dir), expected)

    def test_missing_executable_fails_with_configuration_guidance(self):
        with patch("photogrammetry.shutil.which", return_value=None):
            with self.assertRaisesRegex(BackendConfigurationError, "AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE"):
                resolve_executable("colmap", env={})

    def test_run_writes_separate_manifest_logs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "input images"
            output_root = root / "output models"
            image_dir.mkdir()
            executable = root / "fake tool.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('fake 1.0')\n"
                "    raise SystemExit(0)\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'result.txt').write_text('ok', encoding='utf-8')\n"
                "print('completed')\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            result = run_backend(
                "meshroom",
                image_dir,
                output_root,
                BackendConfig(executable=str(executable)),
            )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(result.return_code, 0)
            self.assertEqual(manifest["backend"], "meshroom")
            self.assertEqual(manifest["version"], "fake 1.0")
            self.assertIn("model/result.txt", manifest["artifacts"])
            self.assertEqual(Path(result.stdout_log).read_text(encoding="utf-8"), "completed\n")
            self.assertTrue(any("input images" in arg for arg in result.command))
            self.assertTrue(Path(result.run_dir).is_relative_to(output_root / "meshroom"))


if __name__ == "__main__":
    unittest.main()
