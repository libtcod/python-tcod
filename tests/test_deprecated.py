"""Test deprecated features."""
from __future__ import annotations

import pytest

import tcod
import tcod.event

# ruff: noqa: D103


@pytest.mark.filterwarnings("error")
def test_deprecate_color() -> None:
    with pytest.raises(FutureWarning, match=r".*\(0, 0, 0\)"):
        _ = tcod.black


@pytest.mark.filterwarnings("error")
def test_deprecate_key_constants() -> None:
    with pytest.raises(FutureWarning, match=r".*KeySym.N1"):
        _ = tcod.event.K_1
    with pytest.raises(FutureWarning, match=r".*Scancode.N1"):
        _ = tcod.event.SCANCODE_1
