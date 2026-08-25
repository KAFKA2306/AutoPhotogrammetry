from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from processing.mesh_export import _artifact_files, _format_from_output


class MeshExportContractTests(unittest.TestCase):
    def test_accepts_supported_formats(self) -> None:
        for extension in ("glb", "obj", "stl"):
            self.assertEqual(_format_from_output(f"asset.{extension}"), extension)

    def test_rejects_unsupported_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported mesh export format"):
            _format_from_output("asset.ply")

    def test_obj_artifact_group_tracks_only_obj_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "asset.obj"
            primary.write_text("o mesh\n", encoding="utf-8")
            (root / "asset.mtl").write_text("newmtl material\n", encoding="utf-8")
            (root / "asset_albedo.png").write_bytes(b"png")
            (root / "asset_normal.jpg").write_bytes(b"jpg")
            (root / "asset.glb").write_bytes(b"glb")
            (root / "asset.stl").write_bytes(b"stl")
            (root / "unrelated.txt").write_text("ignore", encoding="utf-8")

            files = _artifact_files(primary)
            names = {Path(record["path"]).name for record in files}
            self.assertEqual(
                names,
                {"asset.obj", "asset.mtl", "asset_albedo.png", "asset_normal.jpg"},
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in files))
            self.assertTrue(all(record["size_bytes"] > 0 for record in files))


if __name__ == "__main__":
    unittest.main()
