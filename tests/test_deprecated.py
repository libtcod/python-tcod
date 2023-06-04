"""Test deprecated features."""
from __future__ import annotations

import numpy as np
import pytest

import tcod
import tcod.constants
import tcod.event
import tcod.libtcodpy

with pytest.warns():
    import libtcodpy

# ruff: noqa: D103


def test_deprecate_color() -> None:
    with pytest.warns(FutureWarning, match=r"\(0, 0, 0\)"):
        assert tcod.black is tcod.constants.black
    with pytest.warns(FutureWarning, match=r"\(0, 0, 0\)"):
        assert tcod.libtcodpy.black is tcod.constants.black
    with pytest.warns(FutureWarning, match=r"\(0, 0, 0\)"):
        assert libtcodpy.black is tcod.constants.black


def test_constants() -> None:
    with pytest.warns(match=r"libtcodpy.RENDERER_SDL2"):
        assert tcod.RENDERER_SDL2 is tcod.constants.RENDERER_SDL2
    assert tcod.libtcodpy.RENDERER_SDL2 is tcod.constants.RENDERER_SDL2


def test_implicit_libtcodpy() -> None:
    with pytest.warns(match=r"libtcodpy.console_init_root"):
        assert tcod.console_init_root is tcod.libtcodpy.console_init_root


def test_deprecate_key_constants() -> None:
    with pytest.warns(FutureWarning, match=r"KeySym.N1"):
        _ = tcod.event.K_1
    with pytest.warns(FutureWarning, match=r"Scancode.N1"):
        _ = tcod.event.SCANCODE_1


def test_line_where() -> None:
    with pytest.warns():
        where = tcod.libtcodpy.line_where(1, 0, 3, 4)
    np.testing.assert_array_equal(where, [[1, 1, 2, 2, 3], [0, 1, 2, 3, 4]])
