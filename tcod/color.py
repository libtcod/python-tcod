"""

"""
from tcod.libtcod import lib


class Color(list):
    """

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r: int = 0, g: int = 0, b: int = 0):
        self[:] = (r & 0xFF, g & 0xFF, b & 0xFF)

    @property
    def r(self) -> int:
        """int: Red value, always normalised to 0-255."""
        return self[0]

    @r.setter
    def r(self, value: int):
        self[0] = value & 0xFF

    @property
    def g(self) -> int:
        """int: Green value, always normalised to 0-255."""
        return self[1]

    @g.setter
    def g(self, value: int):
        self[1] = value & 0xFF

    @property
    def b(self) -> int:
        """int: Blue value, always normalised to 0-255."""
        return self[2]

    @b.setter
    def b(self, value: int):
        self[2] = value & 0xFF

    @classmethod
    def _new_from_cdata(cls, cdata):
        """new in libtcod-cffi"""
        return cls(cdata.r, cdata.g, cdata.b)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except TypeError:
            return list.__getitem__(self, "rgb".index(index))

    def __setitem__(self, index, value):
        try:
            list.__setitem__(self, index, value)
        except TypeError:
            list.__setitem__(self, "rgb".index(index), value)

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
                lib.TCOD_color_multiply_scalar(self, other)
            )

    def __repr__(self):
        """Return a printable representation of the current color."""
        return "%s(%i, %i, %i)" % (
            self.__class__.__name__,
            self.r,
            self.g,
            self.b,
        )
