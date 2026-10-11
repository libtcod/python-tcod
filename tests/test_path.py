"""Tests for incremental pathfinding."""

import numpy as np
import pytest

import tcod.path


@pytest.mark.parametrize("shape", [(4, 7), (3, 4, 7)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("value", [-3, 0, 3])
def test_add_root_after_resolve(shape: tuple[int, ...], order: str, value: int) -> None:
    graph = tcod.path.CustomGraph(shape, order=order)
    cost = np.ones(shape, dtype=np.int8)
    for axis in range(len(shape)):
        for direction in (-1, 1):
            edge = [0] * len(shape)
            edge[axis] = direction
            graph.add_edge(tuple(edge), cost=cost)

    pathfinder = tcod.path.Pathfinder(graph)
    origin = (0,) * len(shape)
    new_root = tuple(size - 1 for size in shape)
    pathfinder.add_root(origin)
    pathfinder.resolve()
    assert tuple(pathfinder.path_from(new_root)[-1]) == origin

    pathfinder.add_root(new_root, value=value)
    pathfinder.resolve()
    np.testing.assert_array_equal(pathfinder.path_from(new_root), [new_root])
    coordinates = np.indices(shape)
    expected = np.minimum(
        coordinates.sum(axis=0),
        sum(size - 1 - coordinates[axis] for axis, size in enumerate(shape)) + value,
    )
    np.testing.assert_array_equal(pathfinder.distance, expected)
    for index in np.ndindex(shape):
        path = pathfinder.path_from(index)
        root = tuple(path[-1])
        assert root in (origin, new_root)
        root_value = value if root == new_root else 0
        assert len(path) - 1 + root_value == expected[index]
