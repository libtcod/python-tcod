"""libtcod map attributes and field-of-view functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import tcod._internal
import tcod.constants
from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Map:
    """A map containing libtcod attributes.

    .. versionchanged:: 4.1
        `transparent`, `walkable`, and `fov` are now numpy boolean arrays.

    .. versionchanged:: 4.3
        Added `order` parameter.

    Args:
        width (int): Width of the new Map.
        height (int): Height of the new Map.
        order (str): Which numpy memory order to use.

    Attributes:
        width (int): Read only width of this Map.
        height (int): Read only height of this Map.
        transparent: A boolean array of transparent cells.
        walkable: A boolean array of walkable cells.
        fov: A boolean array of the cells lit by :any:'compute_fov'.

    Example::

        >>> import tcod
        >>> m = tcod.map.Map(width=3, height=4)
        >>> m.walkable
        array([[False, False, False],
               [False, False, False],
               [False, False, False],
               [False, False, False]]...)

        # Like the rest of the tcod modules, all arrays here are
        # in row-major order and are addressed with [y,x]
        >>> m.transparent[:] = True  # Sets all to True.
        >>> m.transparent[1:3,0] = False  # Sets (1, 0) and (2, 0) to False.
        >>> m.transparent
        array([[ True,  True,  True],
               [False,  True,  True],
               [False,  True,  True],
               [ True,  True,  True]]...)

        >>> m.compute_fov(0, 0)
        >>> m.fov
        array([[ True,  True,  True],
               [ True,  True,  True],
               [False,  True,  True],
               [False, False,  True]]...)
        >>> m.fov.item(3, 1)
        False

    .. deprecated:: 11.13
        You no longer need to use this class to hold data for field-of-view
        or pathfinding as those functions can now take NumPy arrays directly.
        See :any:`tcod.map.compute_fov` and :any:`tcod.path`.
    """

    def __init__(
        self,
        width: int,
        height: int,
        order: Literal["C", "F"] = "C",
    ) -> None:
        """Initialize the map."""
        warnings.warn(
            "This class may perform poorly and is no longer needed.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.width = width
        self.height = height
        self._order = tcod._internal.verify_order(order)

        self.__buffer: NDArray[np.bool_] = np.zeros((height, width, 3), dtype=np.bool_)
        self.map_c = self.__as_cdata()

    def __as_cdata(self) -> Any:  # noqa: ANN401
        return ffi.new(
            "struct TCOD_Map*",
            (
                self.width,
                self.height,
                self.width * self.height,
                ffi.from_buffer("struct TCOD_MapCell*", self.__buffer),
            ),
        )

    @property
    def transparent(self) -> NDArray[np.bool_]:
        buffer: np.ndarray[Any, np.dtype[np.bool_]] = self.__buffer[:, :, 0]
        return buffer.T if self._order == "F" else buffer

    @property
    def walkable(self) -> NDArray[np.bool_]:
        buffer: np.ndarray[Any, np.dtype[np.bool_]] = self.__buffer[:, :, 1]
        return buffer.T if self._order == "F" else buffer

    @property
    def fov(self) -> NDArray[np.bool_]:
        buffer: np.ndarray[Any, np.dtype[np.bool_]] = self.__buffer[:, :, 2]
        return buffer.T if self._order == "F" else buffer

    def compute_fov(
        self,
        x: int,
        y: int,
        radius: int = 0,
        light_walls: bool = True,
        algorithm: int = tcod.constants.FOV_RESTRICTIVE,
    ) -> None:
        """Compute a field-of-view on the current instance.

        Args:
            x (int): Point of view, x-coordinate.
            y (int): Point of view, y-coordinate.
            radius (int): Maximum view distance from the point of view.

                A value of `0` will give an infinite distance.
            light_walls (bool): Light up walls, or only the floor.
            algorithm (int): Defaults to tcod.FOV_RESTRICTIVE

        If you already have transparency in a NumPy array then you could use
        :any:`tcod.map.compute_fov` instead.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            warnings.warn(
                f"Index ({x}, {y}) is outside of this maps shape ({self.width}, {self.height})."
                "\nThis will raise an error in future versions.",
                RuntimeWarning,
                stacklevel=2,
            )

        lib.TCOD_map_compute_fov(self.map_c, x, y, radius, light_walls, algorithm)

    def __setstate__(self, state: dict[str, Any]) -> None:
        if "_Map__buffer" not in state:  # deprecated
            # remove this check on major version update
            self.__buffer = np.zeros((state["height"], state["width"], 3), dtype=np.bool_)
            self.__buffer[:, :, 0] = state["buffer"] & 0x01
            self.__buffer[:, :, 1] = state["buffer"] & 0x02
            self.__buffer[:, :, 2] = state["buffer"] & 0x04
            del state["buffer"]
            state["_order"] = "F"
        self.__dict__.update(state)
        self.map_c = self.__as_cdata()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["map_c"]
        return state


def compute_fov(
    transparency: ArrayLike,
    pov: tuple[int, int],
    radius: int = 0,
    light_walls: bool = True,
    algorithm: int = tcod.constants.FOV_RESTRICTIVE,
) -> NDArray[np.bool_]:
    """Return a boolean mask of the area covered by a field-of-view.

    `transparency` is a 2 dimensional array where all non-zero values are
    considered transparent.  The returned array will match the shape of this
    array.

    `pov` is the point-of-view origin point.  Areas are visible if they can
    be seen from this position.  `pov` should be a 2D index matching the axes
    of the `transparency` array, and must be within the bounds of the
    `transparency` array.

    `radius` is the maximum view distance from `pov`.  If this is zero then
    the maximum distance is used.

    If `light_walls` is True then visible obstacles will be returned, otherwise
    only transparent areas will be.

    `algorithm` is the field-of-view algorithm to run.  The default value is
    `tcod.FOV_RESTRICTIVE`.
    The options are:

    * `tcod.FOV_BASIC`:
      Simple ray-cast implementation.
    * `tcod.FOV_DIAMOND`
    * `tcod.FOV_SHADOW`:
      Recursive shadow caster.
    * `tcod.FOV_PERMISSIVE(n)`:
      `n` starts at 0 (most restrictive) and goes up to 8 (most permissive.)
    * `tcod.FOV_RESTRICTIVE`
    * `tcod.FOV_SYMMETRIC_SHADOWCAST`

    .. versionadded:: 9.3

    .. versionchanged:: 11.0
        The parameters `x` and `y` have been changed to `pov`.

    .. versionchanged:: 11.17
        Added `tcod.FOV_SYMMETRIC_SHADOWCAST` option.

    Example:
        >>> explored = np.zeros((3, 5), dtype=bool, order="F")
        >>> transparency = np.ones((3, 5), dtype=bool, order="F")
        >>> transparency[:2, 2] = False
        >>> transparency  # Transparent area.
        array([[ True,  True, False,  True,  True],
               [ True,  True, False,  True,  True],
               [ True,  True,  True,  True,  True]]...)
        >>> visible = tcod.map.compute_fov(transparency, (0, 0))
        >>> visible  # Visible area.
        array([[ True,  True,  True, False, False],
               [ True,  True,  True, False, False],
               [ True,  True,  True,  True, False]]...)
        >>> explored |= visible  # Keep track of an explored area.

    .. seealso::
        :any:`numpy.where`: For selecting between two arrays using a boolean
        array, like the one returned by this function.

        :any:`numpy.select`: Select between arrays based on multiple
        conditions.
    """
    transparency = np.asarray(transparency)
    if len(transparency.shape) != 2:  # noqa: PLR2004
        msg = f"transparency must be an array of 2 dimensions (shape is {transparency.shape!r})"
        raise TypeError(msg)
    if isinstance(pov, int):
        msg = "The tcod.map.compute_fov function has changed.  The `x` and `y` parameters should now be given as a single tuple."
        raise TypeError(msg)
    if not (0 <= pov[0] < transparency.shape[0] and 0 <= pov[1] < transparency.shape[1]):
        warnings.warn(
            f"Given pov index {pov!r} is outside the array of shape {transparency.shape!r}."
            "\nThis will raise an error in future versions.",
            RuntimeWarning,
            stacklevel=2,
        )
    map_buffer: NDArray[np.bool_] = np.empty(
        transparency.shape,
        dtype=[("transparent", bool), ("walkable", bool), ("fov", bool)],
    )
    map_cdata = ffi.new(
        "struct TCOD_Map*",
        (
            map_buffer.shape[1],
            map_buffer.shape[0],
            map_buffer.shape[1] * map_buffer.shape[0],
            ffi.from_buffer("struct TCOD_MapCell*", map_buffer),
        ),
    )
    map_buffer["transparent"] = transparency  # type: ignore[call-overload]
    lib.TCOD_map_compute_fov(map_cdata, pov[1], pov[0], radius, light_walls, algorithm)
    return map_buffer["fov"]  # type: ignore[no-any-return,call-overload]
