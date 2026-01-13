"""Test directory configuration."""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import pytest

import tcod
from tcod import libtcodpy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--no-window", action="store_true", help="Skip tests which need a rendering context.")


@pytest.fixture
def uses_window(request: pytest.FixtureRequest) -> Iterator[None]:
    """Marks tests which require a rendering context."""
    if request.config.getoption("--no-window"):
        pytest.skip("This test needs a rendering context.")
    yield None
    return


@pytest.fixture(scope="session", params=["SDL", "SDL2"])
def session_console(request: pytest.FixtureRequest) -> Iterator[tcod.console.Console]:
    if request.config.getoption("--no-window"):
        pytest.skip("This test needs a rendering context.")
    font_file = "libtcod/terminal.png"
    width = 12
    height = 10
    title = "libtcod-cffi tests"
    fullscreen = False
    renderer = getattr(libtcodpy, "RENDERER_" + request.param)

    libtcodpy.console_set_custom_font(font_file)
    with libtcodpy.console_init_root(width, height, title, fullscreen, renderer, vsync=False) as con:
        yield con


@pytest.fixture
def console(session_console: tcod.console.Console) -> tcod.console.Console:
    console = session_console
    libtcodpy.console_flush()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        console.default_fg = (255, 255, 255)
        console.default_bg = (0, 0, 0)
        console.default_bg_blend = libtcodpy.BKGND_SET
        console.default_alignment = libtcodpy.LEFT
    console.clear()
    return console


@pytest.fixture
def offscreen(console: tcod.console.Console) -> tcod.console.Console:
    """Return an off-screen console with the same size as the root console."""
    return tcod.console.Console(console.width, console.height)


@pytest.fixture
def fg() -> libtcodpy.Color:
    return libtcodpy.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


@pytest.fixture
def bg() -> libtcodpy.Color:
    return libtcodpy.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def ch_ascii_int() -> int:
    return random.randint(0x21, 0x7F)


def ch_ascii_str() -> str:
    return chr(ch_ascii_int())


def ch_latin1_int() -> int:
    return random.randint(0x80, 0xFF)


def ch_latin1_str() -> str:
    return chr(ch_latin1_int())


@pytest.fixture(
    params=[
        "ascii_int",
        "ascii_str",
        "latin1_int",
        "latin1_str",
    ]
)
def ch(request: pytest.FixtureRequest) -> Callable[[], int | str]:
    """Test with multiple types of ascii/latin1 characters."""
    return globals()[f"ch_{request.param}"]()  # type: ignore[no-any-return]
