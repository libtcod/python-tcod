"""libtcod map attributes and field-of-view functions.


"""
from typing import Any, Tuple

import numpy as np

from tcod.libtcod import lib, ffi
import tcod._internal
import tcod.constants


class Map(object):
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

        >>> import tcod.map
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
        >>> m.fov[3,1]
        False
    """

    def __init__(self, width: int, height: int, order: str = "C"):
        self.width = width
        self.height = height
        self._order = tcod._internal.verify_order(order)

        self.__buffer = np.zeros((height, width, 3), dtype=np.bool_)
        self.map_c = self.__as_cdata()

    def __as_cdata(self) -> Any:
        return ffi.new(
            "struct TCOD_Map*",
            (
                self.width,
                self.height,
                self.width * self.height,
                ffi.cast("struct TCOD_MapCell*", self.__buffer.ctypes.data),
            ),
        )

    @property
    def transparent(self) -> np.array:
        buffer = self.__buffer[:, :, 0]
        return buffer.T if self._order == "F" else buffer

    @property
    def walkable(self) -> np.array:
        buffer = self.__buffer[:, :, 1]
        return buffer.T if self._order == "F" else buffer

    @property
    def fov(self) -> np.array:
        buffer = self.__buffer[:, :, 2]
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
        :any:`tcod.map_compute_fov` instead.
        """
        lib.TCOD_map_compute_fov(
            self.map_c, x, y, radius, light_walls, algorithm
        )

    def __setstate__(self, state: Any) -> None:
        if "_Map__buffer" not in state:  # deprecated
            # remove this check on major version update
            self.__buffer = np.zeros(
                (state["height"], state["width"], 3), dtype=np.bool_
            )
            self.__buffer[:, :, 0] = state["buffer"] & 0x01
            self.__buffer[:, :, 1] = state["buffer"] & 0x02
            self.__buffer[:, :, 2] = state["buffer"] & 0x04
            del state["buffer"]
            state["_order"] = "F"
        if "_order" not in state:  # remove this check on major version update
            raise RuntimeError("This Map was saved with a bad version of tdl.")
        self.__dict__.update(state)
        self.map_c = self.__as_cdata()

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        del state["map_c"]
        return state


def compute_fov(
    transparency: np.array,
    pov: Tuple[int, int],
    radius: int = 0,
    light_walls: bool = True,
    algorithm: int = tcod.constants.FOV_RESTRICTIVE,
) -> np.array:
    """Return a boolean mask of the area covered by a field-of-view.

    `transparency` is a 2 dimensional array where all non-zero values are
    considered transparent.  The returned array will match the shape of this
    array.

    `pov` is the point-of-view origin point.  Areas are visible if they can
    be seen from this position.  The axes of the `pov` should match the axes
    of the `transparency` array.

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

    .. versionadded:: 9.3

    .. versionchanged:: 11.0
        The parameters `x` and `y` have been changed to `pov`.

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
        :any:`numpy.nonzero`
        :any:`numpy.choose`
        :any:`numpy.select`
    """
    transparency = np.asarray(transparency)
    if len(transparency.shape) != 2:
        raise TypeError(
            "transparency must be an array of 2 dimensions"
            " (shape is %r)" % transparency.shape
        )
    if isinstance(pov, int):
        raise TypeError(
            "The tcod.map.compute_fov function has changed.  The `x` and `y`"
            " parameters should now be given as a single tuple."
        )
    map_ = Map(transparency.shape[1], transparency.shape[0])
    map_.transparent[...] = transparency
    map_.compute_fov(pov[1], pov[0], radius, light_walls, algorithm)
    return map_.fov
