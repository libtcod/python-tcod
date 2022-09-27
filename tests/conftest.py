import random
import warnings
from typing import Any, Callable, Iterator, Union

import pytest

import tcod


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--no-window", action="store_true", help="Skip tests which need a rendering context.")


@pytest.fixture(scope="session", params=["SDL", "SDL2"])
def session_console(request: Any) -> Iterator[tcod.console.Console]:
    if request.config.getoption("--no-window"):
        pytest.skip("This test needs a rendering context.")
    FONT_FILE = "libtcod/terminal.png"
    WIDTH = 12
    HEIGHT = 10
    TITLE = "libtcod-cffi tests"
    FULLSCREEN = False
    RENDERER = getattr(tcod, "RENDERER_" + request.param)

    tcod.console_set_custom_font(FONT_FILE)
    with tcod.console_init_root(WIDTH, HEIGHT, TITLE, FULLSCREEN, RENDERER, vsync=False) as con:
        yield con


@pytest.fixture(scope="function")
def console(session_console: tcod.console.Console) -> tcod.console.Console:
    console = session_console
    tcod.console_flush()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        console.default_fg = (255, 255, 255)
        console.default_bg = (0, 0, 0)
        console.default_bg_blend = tcod.BKGND_SET
        console.default_alignment = tcod.LEFT
    console.clear()
    return console


@pytest.fixture()
def offscreen(console: tcod.console.Console) -> tcod.console.Console:
    """Return an off-screen console with the same size as the root console."""
    return tcod.console.Console(console.width, console.height)


@pytest.fixture()
def fg() -> tcod.Color:
    return tcod.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


@pytest.fixture()
def bg() -> tcod.Color:
    return tcod.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


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
def ch(request: Any) -> Callable[[], Union[int, str]]:
    """Test with multiple types of ascii/latin1 characters"""
    return globals()["ch_%s" % request.param]()  # type: ignore
