
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
        self.width = width
        self.height = height
        self.cdata = ffi.gc(ffi.cast('map_t*',
                                     lib.TCOD_map_new(width, height)),
                            lib.TCOD_map_delete)

        assert ffi.sizeof('cell_t') == 1 # assert buffer alignment
        buffer = ffi.buffer(self.cdata.cells[0:self.cdata.nbcells])
        self.buffer = np.frombuffer(buffer, np.uint8).reshape((height, width))
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
