"""This modules holds functions for NumPy-based line of sight algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from numpy.typing import NDArray


def bresenham(start: tuple[int, int], end: tuple[int, int]) -> NDArray[np.intc]:
    """Return a thin Bresenham line as a NumPy array of shape (length, 2).

    `start` and `end` are the endpoints of the line.
    The result always includes both endpoints, and will always contain at
    least one index.

    You might want to use the results as is, convert them into a list with
    :any:`numpy.ndarray.tolist` or transpose them and use that to index
    another 2D array.

    Example::

        >>> import tcod
        >>> tcod.los.bresenham((3, 5),(7, 7)).tolist()  # Convert into list.
        [[3, 5], [4, 5], [5, 6], [6, 6], [7, 7]]
        >>> tcod.los.bresenham((0, 0), (0, 0))
        array([[0, 0]]...)
        >>> tcod.los.bresenham((0, 0), (4, 4))[1:-1]  # Clip both endpoints.
        array([[1, 1],
               [2, 2],
               [3, 3]]...)

        >>> array = np.zeros((5, 5), dtype=np.int8)
        >>> array
        array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]], dtype=int8)
        >>> tcod.los.bresenham((0, 0), (3, 4)).T  # Transposed results.
        array([[0, 1, 1, 2, 3],
               [0, 1, 2, 3, 4]]...)
        >>> indexes_ij = tuple(tcod.los.bresenham((0, 0), (3, 4)).T)
        >>> array[indexes_ij] = np.arange(len(indexes_ij[0]))
        >>> array
        array([[0, 0, 0, 0, 0],
               [0, 1, 2, 0, 0],
               [0, 0, 0, 3, 0],
               [0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0]], dtype=int8)
        >>> array[indexes_ij]
        array([0, 1, 2, 3, 4], dtype=int8)

    .. versionadded:: 11.14
    """
    x1, y1 = start
    x2, y2 = end
    length = lib.bresenham(x1, y1, x2, y2, 0, ffi.NULL)
    array: np.ndarray[Any, np.dtype[np.intc]] = np.ndarray((length, 2), dtype=np.intc)
    lib.bresenham(x1, y1, x2, y2, length, ffi.from_buffer("int*", array))
    return array
