
from __future__ import absolute_import

import numpy as np

from tcod.libtcod import lib, ffi


class BufferBitProxy(object):

    def __init__(self, buffer, bitmask):
        self.buffer = buffer
        self.bitmask = np.asarray(bitmask, dtype=np.uint8)

    def __getitem__(self, index):
        return (self.buffer[index] & self.bitmask) != 0

    def __setitem__(self, index, values):
        self.buffer[index] &= 0xff ^ self.bitmask
        self.buffer[index] |= self.bitmask * np.asarray(values, dtype=bool)


class Map(object):
    """

    Args:
        width (int): Width of the new Map.
        height (int): Height of the new Map.

    Attributes:
        width (int): Read only width of this Map.
        height (int): Read only height of this Map.
        map_c (CData): A cffi pointer to a TCOD_map_t object.
        transparent: A writable boolean array of transparent cells.
        walkable: A writable boolean array of walkable cells.
        fov: A writable boolean array of the cells lit by :any:'compute_fov'.
    """

    def __init__(self, width, height):
        assert ffi.sizeof('cell_t') == 1 # assert buffer alignment
        self.buffer = np.zeros((height, width), dtype=np.uint8)

        self.map_c = ffi.new('map_t*')
        self.map_c.width = self.width = width
        self.map_c.height = self.height = height
        self.map_c.nbcells = width * height
        self.map_c.cells = ffi.cast('cell_t*', self.buffer.ctypes.data)

        self._init_proxies()

    def _init_proxies(self):
        self.transparent = BufferBitProxy(self.buffer, 0x01)
        self.walkable = BufferBitProxy(self.buffer, 0x02)
        self.fov = BufferBitProxy(self.buffer, 0x04)

    def compute_fov(self, x, y, radius=0, light_walls=True,
                    algorithm=lib.FOV_RESTRICTIVE):
        """Compute a field-of-view on the current instance.

        Args:
            x (int): Point of view, x-coordinate.
            y (int): Point of view, y-coordinate.
            radius (int): Maximum view distance from the point of view.

                A value of `0` will give an infinite distance.
            light_walls (bool): Light up walls, or only the floor.
            algorithm (int): Defaults to FOV_RESTRICTIVE
        """
        lib.TCOD_map_compute_fov(self.map_c, x, y, radius, light_walls,
                                 algorithm)

    def __getstate__(self):
        """Return this objects state.

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        state = self.__dict__.copy()
        del state['map_c']
        return state

    def __setstate__(self, state):
        """Unpack this object from a saved state.  A new buffer is used."""
        if 'width' not in state or 'height' not in state:
            state['width'], state['height'] = state['size']
            del state['size']

        self.__dict__.update(state)

        self.map_c = ffi.new(
            'map_t *',
            (
                self.width,
                self.height,
                self.width * self.height,
                ffi.cast('cell_t*', self.buffer.ctypes.data),
            )
        )

        self._init_proxies()
