"""libtcod map attributes and field-of-view functions.

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
    >>> m.transparent[:] = True # Sets all to True.
    >>> m.transparent[1:3,0] = False # Sets (1, 0) and (2, 0) to False.
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

from __future__ import absolute_import

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

    """

    def __init__(self, width, height, order='C'):
        self.width = width
        self.height = height
        self._order = tcod._internal.verify_order(order)

        self.__buffer = np.zeros((height, width, 3), dtype=np.bool_)
        self.map_c = self.__as_cdata()


    def __as_cdata(self):
        return ffi.new(
            'struct TCOD_Map*',
            (
                self.width,
                self.height,
                self.width * self.height,
                ffi.cast('struct TCOD_MapCell*', self.__buffer.ctypes.data),
            )
        )

    @property
    def transparent(self):
        buffer = self.__buffer[:,:,0]
        return buffer.T if self._order == 'F' else buffer

    @property
    def walkable(self):
        buffer = self.__buffer[:,:,1]
        return buffer.T if self._order == 'F' else buffer

    @property
    def fov(self):
        buffer = self.__buffer[:,:,2]
        return buffer.T if self._order == 'F' else buffer

    def compute_fov(self, x, y, radius=0, light_walls=True,
                    algorithm=tcod.constants.FOV_RESTRICTIVE):
        # type (int, int, int, bool, int) -> None
        """Compute a field-of-view on the current instance.

        Args:
            x (int): Point of view, x-coordinate.
            y (int): Point of view, y-coordinate.
            radius (int): Maximum view distance from the point of view.

                A value of `0` will give an infinite distance.
            light_walls (bool): Light up walls, or only the floor.
            algorithm (int): Defaults to tcod.FOV_RESTRICTIVE
        """
        lib.TCOD_map_compute_fov(
            self.map_c, x, y, radius, light_walls, algorithm)

    def __setstate__(self, state):
        if '_Map__buffer' not in state: # deprecated
            # remove this check on major version update
            self.__buffer = np.zeros((state['height'], state['width'], 3),
                              dtype=np.bool_)
            self.__buffer[:,:,0] = state['buffer'] & 0x01
            self.__buffer[:,:,1] = state['buffer'] & 0x02
            self.__buffer[:,:,2] = state['buffer'] & 0x04
            del state['buffer']
            state['_order'] = 'F'
        if '_order' not in state: # remove this check on major version update
            raise RuntimeError(
                'This Map was saved with a bad version of tdl.')
        self.__dict__.update(state)
        self.map_c = self.__as_cdata()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['map_c']
        return state
