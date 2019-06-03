"""

"""
from typing import Any, List
import warnings

from tcod._internal import deprecate
from tcod.libtcod import lib


class Color(List[int]):
    """

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r: int = 0, g: int = 0, b: int = 0) -> None:
        list.__setitem__(self, slice(None), (r & 0xFF, g & 0xFF, b & 0xFF))

    @property
    def r(self) -> int:
        """int: Red value, always normalised to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[0])

    @r.setter  # type: ignore
    @deprecate("Setting color attributes has been deprecated.")
    def r(self, value: int) -> None:
        self[0] = value & 0xFF

    @property
    def g(self) -> int:
        """int: Green value, always normalised to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[1])

    @g.setter  # type: ignore
    @deprecate("Setting color attributes has been deprecated.")
    def g(self, value: int) -> None:
        self[1] = value & 0xFF

    @property
    def b(self) -> int:
        """int: Blue value, always normalised to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[2])

    @b.setter  # type: ignore
    @deprecate("Setting color attributes has been deprecated.")
    def b(self, value: int) -> None:
        self[2] = value & 0xFF

    @classmethod
    def _new_from_cdata(cls, cdata: Any) -> "Color":
        """new in libtcod-cffi"""
        return cls(cdata.r, cdata.g, cdata.b)

    def __getitem__(self, index: Any) -> Any:
        """
        .. deprecated:: 9.2
            Accessing colors via a letter index is deprecated.
        """
        try:
            return super().__getitem__(index)
        except TypeError:
            warnings.warn(
                "Accessing colors via a letter index is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getitem__("rgb".index(index))

    @deprecate("This class will not be mutable in the future.")
    def __setitem__(self, index: Any, value: Any) -> None:
        try:
            super().__setitem__(index, value)
        except TypeError:
            super().__setitem__("rgb".index(index), value)

    def __eq__(self, other: Any) -> bool:
        """Compare equality between colors.

        Also compares with standard sequences such as 3-item tuples or lists.
        """
        try:
            return bool(lib.TCOD_color_equals(self, other))
        except TypeError:
            return False

    @deprecate("Use NumPy instead for color math operations.")
    def __add__(self, other: Any) -> "Color":
        """Add two colors together.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        return Color._new_from_cdata(lib.TCOD_color_add(self, other))

    @deprecate("Use NumPy instead for color math operations.")
    def __sub__(self, other: Any) -> "Color":
        """Subtract one color from another.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        return Color._new_from_cdata(lib.TCOD_color_subtract(self, other))

    @deprecate("Use NumPy instead for color math operations.")
    def __mul__(self, other: Any) -> "Color":
        """Multiply with a scaler or another color.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        if isinstance(other, (Color, list, tuple)):
            return Color._new_from_cdata(lib.TCOD_color_multiply(self, other))
        else:
            return Color._new_from_cdata(
                lib.TCOD_color_multiply_scalar(self, other)
            )

    def __repr__(self) -> str:
        """Return a printable representation of the current color."""
        return "%s(%r, %r, %r)" % (
            self.__class__.__name__,
            self.r,
            self.g,
            self.b,
        )
