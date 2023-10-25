#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this example.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""A parallelization example for the field-of-view and path-finding tasks.

Because of the Python GIL you can have only one thread in Python at a time.
However, many C functions with long computations will release the GIL for their
duration, allowing another Python thread to call into another C function.

This script tests the viability of running python-tcod tasks in parallel.
Typically the field-of-view tasks run good but not great, and the path-finding
tasks run poorly.
"""
import concurrent.futures
import multiprocessing
import platform
import sys
import timeit
from typing import Callable, List, Tuple

import tcod.map

THREADS = multiprocessing.cpu_count()

MAP_WIDTH = 100
MAP_HEIGHT = 100
MAP_NUMBER = 500

REPEAT = 10  # Number to times to run a test. Only the fastest result is shown.


def test_fov(map_: tcod.map.Map) -> tcod.map.Map:  # noqa: D103
    map_.compute_fov(MAP_WIDTH // 2, MAP_HEIGHT // 2)
    return map_


def test_fov_single(maps: List[tcod.map.Map]) -> None:  # noqa: D103
    for map_ in maps:
        test_fov(map_)


def test_fov_threads(executor: concurrent.futures.Executor, maps: List[tcod.map.Map]) -> None:  # noqa: D103
    for _result in executor.map(test_fov, maps):
        pass


def test_astar(map_: tcod.map.Map) -> List[Tuple[int, int]]:  # noqa: D103
    astar = tcod.path.AStar(map_)
    return astar.get_path(0, 0, MAP_WIDTH - 1, MAP_HEIGHT - 1)


def test_astar_single(maps: List[tcod.map.Map]) -> None:  # noqa: D103
    for map_ in maps:
        test_astar(map_)


def test_astar_threads(executor: concurrent.futures.Executor, maps: List[tcod.map.Map]) -> None:  # noqa: D103
    for _result in executor.map(test_astar, maps):
        pass


def run_test(
    maps: List[tcod.map.Map],
    single_func: Callable[[List[tcod.map.Map]], None],
    multi_func: Callable[[concurrent.futures.Executor, List[tcod.map.Map]], None],
) -> None:
    """Run a function designed for a single thread and compare it to a threaded version.

    This prints the results of these tests.
    """
    single_time = min(timeit.repeat(lambda: single_func(maps), number=1, repeat=REPEAT))
    print(f"Single threaded: {single_time * 1000:.2f}ms")

    for i in range(1, THREADS + 1):
        executor = concurrent.futures.ThreadPoolExecutor(i)
        multi_time = min(timeit.repeat(lambda: multi_func(executor, maps), number=1, repeat=REPEAT))
        print(f"{i} threads: {multi_time * 1000:.2f}ms, " f"{single_time / (multi_time * i) * 100:.2f}% efficiency")


def main() -> None:
    """Setup and run tests."""
    maps = [tcod.map.Map(MAP_WIDTH, MAP_HEIGHT) for i in range(MAP_NUMBER)]
    for map_ in maps:
        map_.walkable[...] = True
        map_.transparent[...] = True

    print(f"Python {sys.version}\n{platform.platform()}\n{platform.processor()}")

    print(f"\nComputing field-of-view for " f"{len(maps)} empty {MAP_WIDTH}x{MAP_HEIGHT} maps.")
    run_test(maps, test_fov_single, test_fov_threads)

    print(
        f"\nComputing AStar from corner to corner {len(maps)} times "
        f"on separate empty {MAP_WIDTH}x{MAP_HEIGHT} maps."
    )
    run_test(maps, test_astar_single, test_astar_threads)


if __name__ == "__main__":
    main()
