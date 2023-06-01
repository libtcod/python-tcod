"""Test deprecated features."""
from __future__ import annotations

import numpy as np
import pytest

import tcod
import tcod.event
import tcod.libtcodpy

with pytest.warns():
    import libtcodpy

# ruff: noqa: D103


@pytest.mark.filterwarnings("error")
def test_deprecate_color() -> None:
    with pytest.raises(FutureWarning, match=r".*\(0, 0, 0\)"):
        _ = tcod.black
    with pytest.raises(FutureWarning, match=r".*\(0, 0, 0\)"):
        _ = tcod.libtcodpy.black
    with pytest.raises(FutureWarning, match=r".*\(0, 0, 0\)"):
        _ = libtcodpy.black


@pytest.mark.filterwarnings("error")
def test_deprecate_key_constants() -> None:
    with pytest.raises(FutureWarning, match=r".*KeySym.N1"):
        _ = tcod.event.K_1
    with pytest.raises(FutureWarning, match=r".*Scancode.N1"):
        _ = tcod.event.SCANCODE_1


def test_line_where() -> None:
    with pytest.warns():
        where = tcod.libtcodpy.line_where(1, 0, 3, 4)
    np.testing.assert_array_equal(where, [[1, 1, 2, 2, 3], [0, 1, 2, 3, 4]])
