"""Many-to-many paths must retain movement costs during reconstruction."""

import numpy as np
import pytest

import tcod.path


@pytest.mark.parametrize("diagonal", [0, 1, 2, 3])
def test_path2d_diagonal_predecessors(diagonal: int) -> None:
    """An expensive diagonal must not replace two cheaper cardinal moves."""
    cost = np.ones((4, 4), dtype=np.int16)
    path = tcod.path.path2d(
        cost, start_points=[(0, 0), (0, 1)], end_points=[(2, 2), (3, 3)], cardinal=1, diagonal=diagonal
    )
    total = sum(diagonal if np.all(a != b) else 1 for a, b in zip(path, path[1:]))
    assert total == (2 if diagonal == 1 else 3)
    assert tuple(path[0]) in [(0, 0), (0, 1)]
    assert tuple(path[-1]) in [(2, 2), (3, 3)]
    np.testing.assert_array_equal(cost, 1)
