
from __future__ import absolute_import as _

import numpy as np

from tcod.libtcod import lib, ffi


class BufferBitProxy(object):

    def __init__(self, buffer, bitmask):
        self.buffer = buffer
        self.bitmask = bitmask

    def __getitem__(self, index):
        return (self.buffer[index] & self.bitmask) != 0

    def __setitem__(self, index, values):
        self.buffer[index] &= 0xff ^ self.bitmask
        self.buffer[index] |= self.bitmask * np.asarray(values, bool)


class Map(object):
    """
    .. versionadded:: 2.0

    Args:
        width (int): Width of the new Map.
        height (int): Height of the new Map.

    Attributes:
        width (int): Read only width of this Map.
        height (int): Read only height of this Map.
    """

    def __init__(self, width, height):
        assert ffi.sizeof('cell_t') == 1 # assert buffer alignment
        self.buffer = np.zeros((height, width), dtype=np.uint8)

        self.cdata = ffi.new('map_t*')
        self.cdata.width = self.width = width
        self.cdata.height = self.height = height
        self.cdata.nbcells = width * height
        self.cdata.cells = ffi.cast('cell_t*', self.buffer.ctypes.data)

        self.transparent = BufferBitProxy(self.buffer, 0x01)
        self.walkable = BufferBitProxy(self.buffer, 0x02)
        self.fov = BufferBitProxy(self.buffer, 0x04)

    def compute_fov(self, x, y, radius=0, light_walls=True,
                    algorithm=lib.FOV_RESTRICTIVE):
        """

        Args:
            x (int):
            y (int):
            radius (int):
            light_walls (bool):
            algorithm (int): Defaults to FOV_RESTRICTIVE
        """
        lib.TCOD_map_compute_fov(self.cdata, x, y, radius, light_walls,
                                 algorithm)

    def __getstate__(self):
        """Return this objects state.

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        return {'size': (self.width, self.height), 'buffer': self.buffer}

    def __setstate__(self, state):
        """Unpack this object from a saved state.  A new buffer is used."""
        self.__init__(*state['size'])
        self.buffer[...] = state['buffer']
