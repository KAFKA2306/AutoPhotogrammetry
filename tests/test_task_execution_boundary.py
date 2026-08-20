import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task"


class TaskExecutionBoundaryTests(unittest.TestCase):
    def run_task(self, path):
        env = os.environ.copy()
        env["PATH"] = path
        return subprocess.run(
            ["/bin/bash", str(TASK), "doctor"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_missing_docker_cli_fails_closed_without_host_repair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_task(temp_dir)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker CLI unavailable", result.stderr)
        self.assertIn("execution is BLOCKED", result.stderr)
        self.assertIn("no host repair is attempted", result.stderr)

    def test_unavailable_docker_daemon_fails_closed_without_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docker = Path(temp_dir) / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                'if [ "${1:-}" = info ]; then exit 1; fi\n'
                "echo unexpected-docker-command >&2\n"
                "exit 99\n",
                encoding="utf-8",
            )
            docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
            result = self.run_task(temp_dir)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker daemon unavailable", result.stderr)
        self.assertIn("execution is BLOCKED", result.stderr)
        self.assertIn("no host repair is attempted", result.stderr)
        self.assertNotIn("unexpected-docker-command", result.stderr)


if __name__ == "__main__":
    unittest.main()
