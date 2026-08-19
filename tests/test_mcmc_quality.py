import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.mcmc_quality import (
    GSPLAT_MCMC_VERSION,
    NERFSTUDIO_MCMC_REVISION,
    mcmc_train_args,
    run_mcmc_comparison,
    verify_research_environment,
)


class McmcQualityTests(unittest.TestCase):
    def test_mcmc_args_change_only_strategy(self):
        default = mcmc_train_args(iterations=30000, enabled=False)
        mcmc = mcmc_train_args(iterations=30000, enabled=True)
        self.assertEqual(mcmc[: len(default)], default)
        self.assertEqual(mcmc[len(default) :], ("--pipeline.model.strategy", "mcmc"))

    def test_verify_research_environment_checks_exact_commit_and_gsplat(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "nerfstudio"
            (source / ".git").mkdir(parents=True)
            completed = subprocess.CompletedProcess(
                ["git"], 0, NERFSTUDIO_MCMC_REVISION + "\n", ""
            )
            with patch("processing.mcmc_quality.subprocess.run", return_value=completed), patch(
                "processing.mcmc_quality.metadata.version", return_value=GSPLAT_MCMC_VERSION
            ):
                result = verify_research_environment(source)
            self.assertEqual(result["nerfstudio_revision"], NERFSTUDIO_MCMC_REVISION)
            self.assertEqual(result["gsplat_version"], GSPLAT_MCMC_VERSION)

    def test_run_comparison_records_default_and_mcmc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            source = root / "nerfstudio"
            source.mkdir()
            calls = []

            def fake_run(data_dir, output_root, **kwargs):
                calls.append(kwargs["train_extra_args"])
                run_dir = Path(output_root) / "splatfacto" / "run"
                (run_dir / "export").mkdir(parents=True)
                (run_dir / "export" / "splat.ply").write_bytes(b"ply")
                manifest = run_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {
                    "manifest_path": str(manifest),
                    "output": {
                        "ply_path": "export/splat.ply",
                        "sha256": "b" * 64,
                        "size_bytes": 3,
                    },
                }

            environment = {
                "nerfstudio_repository": "https://github.com/nerfstudio-project/nerfstudio",
                "nerfstudio_revision": NERFSTUDIO_MCMC_REVISION,
                "gsplat_repository": "https://github.com/nerfstudio-project/gsplat",
                "gsplat_version": GSPLAT_MCMC_VERSION,
            }
            with patch("processing.mcmc_quality.verify_research_environment", return_value=environment), patch(
                "processing.mcmc_quality.run_splatfacto_export", side_effect=fake_run
            ):
                result = run_mcmc_comparison(
                    data,
                    root / "out",
                    nerfstudio_source=source,
                    iterations=30000,
                )

            self.assertEqual(len(calls), 2)
            self.assertNotIn("--pipeline.model.strategy", calls[0])
            self.assertIn("--pipeline.model.strategy", calls[1])
            self.assertEqual([entry["strategy"] for entry in result["experiments"]], ["default", "mcmc"])
            self.assertTrue(Path(result["manifest_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
