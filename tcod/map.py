
from __future__ import absolute_import as _

from tcod.libtcod import lib, ffi
from tcod.tcod import _CDataWrapper

class Map(_CDataWrapper):
    """
    .. versionadded:: 2.0

    Args:
        width (int): Width of the new Map.
        height (int): Height of the new Map.

    Attributes:
        width (int): Read only width of this Map.
        height (int): Read only height of this Map.
    """

    def __init__(self, *args, **kargs):
        super(Map, self).__init__(*args, **kargs)
        if not self.cdata:
            self._init(*args, **kargs)

        self.width = lib.TCOD_map_get_width(self.cdata)
        self.height = lib.TCOD_map_get_width(self.cdata)

    def _init(self, width, height):
        self.cdata = ffi.gc(lib.TCOD_map_new(width, height),
                            lib.TCOD_map_delete)

    def set_properties(self, x, y, transparent, walkable):
        lib.TCOD_map_set_properties(self.cdata, x, y, transparent, walkable)

    def clear(self, transparent, walkable):
        lib.TCOD_map_clear(self.cdata, transparent, walkable)

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

    def is_fov(self, x, y):
        return lib.TCOD_map_is_in_fov(self.cdata, x, y)

    def is_transparent(self, x, y):
        return lib.TCOD_map_is_transparent(self.cdata, x, y)

    def is_walkable(self, x, y):
        return lib.TCOD_map_is_walkable(self.cdata, x, y)
