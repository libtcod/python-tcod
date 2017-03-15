"""

"""

from __future__ import absolute_import

from tcod.libtcod import ffi, lib


class Color(list):
    """

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r=0, g=0, b=0):
        self[:] = (r & 0xff, g & 0xff, b & 0xff)

    @property
    def r(self):
        """int: Red value, always normalised to 0-255."""
        return self[0]

    @r.setter
    def r(self, value):
        self[0] = value & 0xff

    @property
    def g(self):
        """int: Green value, always normalised to 0-255."""
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value & 0xff

    @property
    def b(self):
        """int: Blue value, always normalised to 0-255."""
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value & 0xff

    @classmethod
    def _new_from_cdata(cls, cdata):
        """new in libtcod-cffi"""
        return cls(cdata.r, cdata.g, cdata.b)


    @classmethod
    def _new_from_int(cls, integer):
        """a TDL int color: 0xRRGGBB

        new in libtcod-cffi"""
        return cls(lib.TDL_color_from_int(integer))

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except TypeError:
            return list.__getitem__(self, 'rgb'.index(index))

    def __setitem__(self, index, value):
        try:
            list.__setitem__(self, index, value)
        except TypeError:
            list.__setitem__(self, 'rgb'.index(index), value)

    def __eq__(self, other):
        """Compare equality between colors.

        Also compares with standard sequences such as 3-item tuples or lists.
        """
        try:
            return bool(lib.TCOD_color_equals(self, other))
        except TypeError:
            return False

    def __add__(self, other):
        """Add two colors together."""
        return Color._new_from_cdata(lib.TCOD_color_add(self, other))

    def __sub__(self, other):
        """Subtract one color from another."""
        return Color._new_from_cdata(lib.TCOD_color_subtract(self, other))

    def __mul__(self, other):
        """Multiply with a scaler or another color."""
        if isinstance(other, (Color, list, tuple)):
            return Color._new_from_cdata(lib.TCOD_color_multiply(self, other))
        else:
            return Color._new_from_cdata(
                lib.TCOD_color_multiply_scalar(self, other))

    def __repr__(self):
        """Return a printable representation of the current color."""
        return "%s(%i,%i,%i)" % (self.__class__.__name__,
                                 self.r, self.g, self.b)

    def __int__(self):
        """Return this color as an integer in 0xRRGGBB format."""
        return lib.TDL_color_RGB(*self)


def _import_colors(lib):
    """Import all Color constants from lib into this module."""
    g = globals()
    for name in dir(lib):
        if name[:5] != 'TCOD_':
            continue
        value = getattr(lib, name)
        if not isinstance(value, ffi.CData):
            continue
        if ffi.typeof(value) != ffi.typeof('TCOD_color_t'):
            continue
        g[name[5:]] = Color._new_from_cdata(value)


_import_colors(lib)
