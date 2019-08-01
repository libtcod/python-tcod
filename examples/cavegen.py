#!/usr/bin/env python3
"""A basic cellular automata cave generation example using SciPy.

http://www.roguebasin.com/index.php?title=Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels

This will print the result to the console, so be sure to run this from the
command line.
"""
import scipy.signal  # type: ignore
import numpy as np  # type: ignore


def convolve(tiles: np.array, wall_rule: int = 5) -> np.array:
    """Return the next step of the cave generation algorithm.

    `tiles` is the input array. (0: wall, 1: floor)

    If the 3x3 area around a tile (including itself) has `wall_rule` number of
    walls then the tile will become a wall.
    """
    # Use convolve2d, the 2nd input is a 3x3 ones array.
    neighbors = scipy.signal.convolve2d(
        ~tiles, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "same"
    )
    return neighbors < wall_rule  # Apply the wall rule.


def show(tiles: np.array) -> None:
    """Print out the tiles of an array."""
    for line in tiles:
        print("".join("# "[int(cell)] for cell in line))


if __name__ == "__main__":
    WIDTH, HEIGHT = 60, 20
    INITIAL_CHANCE = 0.45  # Initial wall chance.
    CONVOLVE_STEPS = 4
    # 0: wall, 1: floor
    tiles = np.random.random((HEIGHT, WIDTH)) > INITIAL_CHANCE
    for _ in range(CONVOLVE_STEPS):
        tiles = convolve(tiles)
        tiles[[0, -1], :] = 0  # Ensure surrounding wall.
        tiles[:, [0, -1]] = 0
    show(tiles)
