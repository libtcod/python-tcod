"""Automatic heuristics must account for cheaper cardinal alternatives."""

import numpy as np
import pytest

import tcod.path


@pytest.mark.parametrize("order", ["C", "F"])
def test_expensive_diagonal_heuristic(order: str) -> None:
    """Eight cardinal steps beat the tempting nine-cost mixed route."""
    cost = np.array(
        [[1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]],
        dtype=np.int16,
        order=order,
    )
    graph = tcod.path.SimpleGraph(cost=cost, cardinal=1, diagonal=3)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((0, 0))
    path = pathfinder.path_to((4, 4))
    total = sum(3 if np.all(a != b) else 1 for a, b in zip(path, path[1:]))
    assert total == 8
    assert pathfinder.distance[4, 4] == 8


def test_diagonal_only_graph() -> None:
    """Disabling cardinal movement retains the diagonal estimate and edges."""
    graph = tcod.path.SimpleGraph(cost=np.ones((3, 3), dtype=np.int16), cardinal=0, diagonal=3)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((0, 0))
    np.testing.assert_array_equal(pathfinder.path_to((2, 2)), [[0, 0], [1, 1], [2, 2]])
    assert pathfinder.distance[2, 2] == 6
