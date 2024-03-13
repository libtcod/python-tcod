"""Old libtcod color management."""

from __future__ import annotations

import warnings
from typing import Any, List

from tcod._internal import deprecate
from tcod.cffi import lib


class Color(List[int]):
    """Old-style libtcodpy color class.

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r: int = 0, g: int = 0, b: int = 0) -> None:  # noqa: D107
        list.__setitem__(self, slice(None), (r & 0xFF, g & 0xFF, b & 0xFF))

    @property
    def r(self) -> int:
        """int: Red value, always normalized to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[0])

    @r.setter
    @deprecate("Setting color attributes has been deprecated.", FutureWarning)
    def r(self, value: int) -> None:
        self[0] = value & 0xFF

    @property
    def g(self) -> int:
        """int: Green value, always normalized to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[1])

    @g.setter
    @deprecate("Setting color attributes has been deprecated.", FutureWarning)
    def g(self, value: int) -> None:
        self[1] = value & 0xFF

    @property
    def b(self) -> int:
        """int: Blue value, always normalized to 0-255.

        .. deprecated:: 9.2
            Color attributes will not be mutable in the future.
        """
        return int(self[2])

    @b.setter
    @deprecate("Setting color attributes has been deprecated.", FutureWarning)
    def b(self, value: int) -> None:
        self[2] = value & 0xFF

    @classmethod
    def _new_from_cdata(cls, cdata: Any) -> Color:  # noqa: ANN401
        return cls(cdata.r, cdata.g, cdata.b)

    def __getitem__(self, index: Any) -> Any:  # noqa: ANN401
        """Return a color channel.

        .. deprecated:: 9.2
            Accessing colors via a letter index is deprecated.
        """
        if isinstance(index, str):
            warnings.warn(
                "Accessing colors via a letter index is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getitem__("rgb".index(index))
        return super().__getitem__(index)

    @deprecate("This class will not be mutable in the future.", FutureWarning)
    def __setitem__(self, index: Any, value: Any) -> None:  # noqa: ANN401, D105
        if isinstance(index, str):
            super().__setitem__("rgb".index(index), value)
        else:
            super().__setitem__(index, value)

    def __eq__(self, other: object) -> bool:
        """Compare equality between colors.

        Also compares with standard sequences such as 3-item tuples or lists.
        """
        try:
            return bool(lib.TCOD_color_equals(self, other))
        except TypeError:
            return False

    @deprecate("Use NumPy instead for color math operations.", FutureWarning)
    def __add__(self, other: object) -> Color:  # type: ignore[override]
        """Add two colors together.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        return Color._new_from_cdata(lib.TCOD_color_add(self, other))

    @deprecate("Use NumPy instead for color math operations.", FutureWarning)
    def __sub__(self, other: object) -> Color:
        """Subtract one color from another.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        return Color._new_from_cdata(lib.TCOD_color_subtract(self, other))

    @deprecate("Use NumPy instead for color math operations.", FutureWarning)
    def __mul__(self, other: object) -> Color:
        """Multiply with a scaler or another color.

        .. deprecated:: 9.2
            Use NumPy instead for color math operations.
        """
        if isinstance(other, (Color, list, tuple)):
            return Color._new_from_cdata(lib.TCOD_color_multiply(self, other))
        return Color._new_from_cdata(lib.TCOD_color_multiply_scalar(self, other))

    def __repr__(self) -> str:
        """Return a printable representation of the current color."""
        return f"{self.__class__.__name__}({self.r!r}, {self.g!r}, {self.b!r})"
