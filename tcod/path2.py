from typing import Iterator, Sequence

import numpy as np


def get_2d_edges(cardinal: float, diagonal: float) -> np.ndarray:
    return (
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) * cardinal
        + np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) * diagonal
    )


def get_hex_edges(cost: float) -> np.ndarray:
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * cost


class EdgeRule:
    def __init__(self, vector: Sequence[int], destination: np.ndarray):
        self.vector = vector
        self.destination = destination


def new_rule(edges: np.ndarray) -> Iterator[EdgeRule]:
    i_center = (edges.shape[0] - 1) // 2
    j_center = (edges.shape[1] - 1) // 2
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] == 0:
                continue
            yield EdgeRule((i - i_center, j - j_center), edges[i, j])
