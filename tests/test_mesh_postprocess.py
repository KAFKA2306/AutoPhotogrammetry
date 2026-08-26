from __future__ import annotations

import unittest

import numpy as np

from processing.mesh_postprocess import (
    _canonical_triangles,
    _duplicate_count,
    _duplicate_triangle_count,
    _triangle_edges,
)


class MeshPostprocessContractTests(unittest.TestCase):
    def test_triangle_edges_are_undirected(self) -> None:
        triangles = np.array([[2, 0, 1]], dtype=np.int64)
        edges = _triangle_edges(triangles)
        self.assertEqual(edges.tolist(), [[0, 2], [0, 1], [1, 2]])

    def test_duplicate_count_uses_exact_rows(self) -> None:
        rows = np.array([[0, 1, 2], [0, 1, 2], [2, 1, 0]], dtype=np.int64)
        self.assertEqual(_duplicate_count(rows), 1)

    def test_canonical_triangles_ignore_vertex_order(self) -> None:
        triangles = np.array([[0, 1, 2], [2, 0, 1], [2, 1, 0]], dtype=np.int64)
        self.assertEqual(
            _canonical_triangles(triangles).tolist(),
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        )

    def test_duplicate_triangle_count_is_winding_independent(self) -> None:
        triangles = np.array(
            [
                [0, 1, 2],
                [2, 0, 1],
                [2, 1, 0],
                [0, 2, 3],
            ],
            dtype=np.int64,
        )
        self.assertEqual(_duplicate_triangle_count(triangles), 2)

    def test_duplicate_count_handles_empty_rows(self) -> None:
        self.assertEqual(_duplicate_count(np.empty((0, 3), dtype=np.int64)), 0)
        self.assertEqual(_duplicate_triangle_count(np.empty((0, 3), dtype=np.int64)), 0)


if __name__ == "__main__":
    unittest.main()
