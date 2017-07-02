
import sys

import random

import pytest

import tcod

def pytest_addoption(parser):
    parser.addoption("--no-window", action="store_true",
        help="Skip tests which need a rendering context.")

@pytest.fixture(scope="session", params=['SDL', 'OPENGL', 'GLSL'])
def session_console(request):
    if(pytest.config.getoption("--no-window")):
        pytest.skip("This test needs a rendering context.")
    FONT_FILE = 'libtcod/terminal.png'
    WIDTH = 12
    HEIGHT = 10
    TITLE = 'libtcod-cffi tests'
    FULLSCREEN = False
    RENDERER = getattr(tcod, 'RENDERER_' + request.param)

    tcod.console_set_custom_font(FONT_FILE)
    with tcod.console_init_root(WIDTH, HEIGHT,
                                TITLE, FULLSCREEN, RENDERER) as con:
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

@pytest.fixture()
def offscreen(console):
    """Return an off-screen console with the same size as the root console."""
    return tcod.console.Console(console.width, console.height)

@pytest.fixture()
def fg():
    return tcod.Color(random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))

@pytest.fixture()
def bg():
    return tcod.Color(random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))

try:
    unichr
except NameError:
    unichr = chr

def ch_ascii_int():
    return random.randint(0x21, 0x7f)

def ch_ascii_str():
    return chr(ch_ascii_int())

def ch_latin1_int():
    return random.randint(0x80, 0xff)

def ch_latin1_str():
    return chr(ch_latin1_int())

def ch_bmp_int():
    # Basic Multilingual Plane, before surrogates
    return random.randint(0x100, 0xd7ff)

def ch_bmp_str():
    return unichr(ch_bmp_int())

def ch_smp_int():
    return random.randint(0x10000, 0x1f9ff)

def ch_smp_str():
    return unichr(ch_bmp_int())

@pytest.fixture(params=['ascii_int', 'ascii_str',
                        'latin1_int', 'latin1_str',
                        #'bmp_int', 'bmp_str', # causes crashes
                        ])
def ch(request):
    """Test with multiple types of ascii/latin1 characters"""
    return globals()['ch_%s' % request.param]()
