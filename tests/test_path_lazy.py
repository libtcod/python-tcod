"""Regression tests for resolving an existing distance after adding roots."""

from typing import Literal

import numpy as np
import pytest

import tcod.path


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("heuristic", [False, True])
@pytest.mark.parametrize("full_resolve", [False, True])
def test_resolve_after_lowering_root(order: Literal["C", "F"], heuristic: bool, full_resolve: bool) -> None:
    cost = np.ones((9, 13), dtype=np.int8, order=order)
    graph = tcod.path.SimpleGraph(cost=cost, cardinal=1, diagonal=0)
    if not heuristic:
        graph.set_heuristic(cardinal=0, diagonal=0)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((0, 0), value=7)
    pathfinder.resolve(None if full_resolve else (8, 12))
    pathfinder.add_root((0, 0), value=-3)
    for goal in ((8, 12), (0, 1), (4, 6)):
        path = pathfinder.path_from(goal)
        assert pathfinder.distance[goal] == sum(goal) - 3
        assert len(path) == sum(goal) + 1
        np.testing.assert_array_equal(path[-1], (0, 0))


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("heuristic", [False, True])
def test_resolve_after_adding_root(order: Literal["C", "F"], heuristic: bool) -> None:
    cost = np.ones((9, 13), dtype=np.int8, order=order)
    graph = tcod.path.SimpleGraph(cost=cost, cardinal=1, diagonal=0)
    if not heuristic:
        graph.set_heuristic(cardinal=0, diagonal=0)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((0, 0))
    pathfinder.resolve()
    pathfinder.add_root((8, 12))
    for goal in ((8, 11), (4, 6), (7, 12)):
        pathfinder.resolve(goal)
        assert pathfinder.distance[goal] == min(sum(goal), 20 - sum(goal))


@pytest.mark.parametrize("heuristic", [False, True])
def test_cached_goal_does_not_resolve_entire_frontier(heuristic: bool) -> None:
    graph = tcod.path.SimpleGraph(cost=np.ones((101, 101), dtype=np.int8), cardinal=1, diagonal=0)
    if not heuristic:
        graph.set_heuristic(cardinal=0, diagonal=0)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((0, 0))
    pathfinder.resolve()
    pathfinder.add_root((100, 100))
    before = pathfinder.distance.copy()
    pathfinder.resolve((0, 1))
    assert pathfinder.distance[0, 1] == 1
    assert np.count_nonzero(pathfinder.distance != before) <= 4
    assert pathfinder.distance[100, 98] == before[100, 98]
