
import pytest

import tcod

def pytest_addoption(parser):
    parser.addoption("--no-window", action="store_true",
        help="Skip tests which need a rendering context.")

@pytest.fixture(scope="module")
def session_console():
    if(pytest.config.getoption("--no-window")):
        pytest.skip("This test needs a rendering context.")
    FONT_FILE = 'libtcod/terminal.png'
    WIDTH = 12
    HEIGHT = 10
    TITLE = 'libtcod-cffi tests'
    FULLSCREEN = False

    tcod.console_set_custom_font(FONT_FILE)
    with tcod.console_init_root(WIDTH, HEIGHT, TITLE, FULLSCREEN) as con:
        yield con

@pytest.fixture(scope="function")
def console(session_console):
    console = session_console
    tcod.console_flush()
    console.default_fg = (255, 255, 255)
    console.default_bg = (0, 0, 0)
    console.default_blend = tcod.BKGND_SET
    console.default_alignment = tcod.LEFT
    console.clear()
    return console
