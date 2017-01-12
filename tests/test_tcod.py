#!/usr/bin/env python

import pytest
import unittest

from common import tcod, raise_Exception


def test_line_error():
    """
    test exception propagation
    """
    with pytest.raises(Exception):
        tcod.line(*LINE_ARGS, py_callback=raise_Exception)


def test_clipboard():
    tcod.clipboard_set('')
    tcod.clipboard_get()


def test_tcod_bsp():
    """
    test tcod additions to BSP
    """
    bsp = tcod.BSP(0, 0, 32, 32)

    assert bsp.level == 0
    assert not bsp.horizontal
    assert not bsp.children

    with pytest.raises(Exception):
        tcod.bsp_traverse_pre_order(bsp, raise_Exception)

    bsp.split_recursive(3, 4, 4, 1, 1)
    for node in bsp.walk():
        assert isinstance(node, tcod.BSP)

    assert bsp != 'asd'

    # test that operations on deep BSP nodes preserve depth
    sub_bsp = bsp.children[0]
    sub_bsp.split_recursive(3, 2, 2, 1, 1)
    assert sub_bsp.children[0].level == 2


def test_array_read_write(console):
    FG = (255, 254, 253)
    BG = (1, 2, 3)
    CH = ord('&')
    tcod.console_put_char_ex(console, 0, 0, CH, FG, BG)
    assert console.ch[0, 0] == CH
    assert tuple(console.fg[0, 0]) == FG
    assert tuple(console.bg[0, 0]) == BG

    tcod.console_put_char_ex(console, 1, 2, CH, FG, BG)
    assert console.ch[2, 1] == CH
    assert tuple(console.fg[2, 1]) == FG
    assert tuple(console.bg[2, 1]) == BG

    console.clear()
    assert console.ch[1, 1] == ord(' ')
    assert tuple(console.fg[1, 1]) == (255, 255, 255)
    assert tuple(console.bg[1, 1]) == (0, 0, 0)

    ch_slice = console.ch[1, :]
    ch_slice[2] = CH
    console.fg[1, ::2] = FG
    console.bg[...] = BG

    assert tcod.console_get_char(console, 2, 1) == CH
    assert tuple(tcod.console_get_char_foreground(console, 2, 1)) == FG
    assert tuple(tcod.console_get_char_background(console, 2, 1)) == BG


def test_console_defaults(console):
    console.default_bg = [2, 3, 4]
    assert console.default_bg == (2, 3, 4)

    console.default_fg = (4, 5, 6)
    assert console.default_fg == (4, 5, 6)

    console.default_blend = tcod.BKGND_ADD
    assert console.default_blend == tcod.BKGND_ADD

    console.default_alignment = tcod.RIGHT
    assert console.default_alignment == tcod.RIGHT


def test_tcod_map_set_bits(benchmark):
    map_ = tcod.map.Map(2,2)

    assert map_.transparent[:].any() == False
    assert map_.walkable[:].any() == False
    assert map_.fov[:].any() == False

    map_.transparent[1, 0] = True
    assert tcod.map_is_transparent(map_, 0, 1) == True
    map_.walkable[1, 0] = True
    assert tcod.map_is_walkable(map_, 0, 1) == True
    map_.fov[1, 0] = True
    assert tcod.map_is_in_fov(map_, 0, 1) == True

    benchmark(map_.transparent.__setitem__, 0, 0)


def test_tcod_map_get_bits(benchmark):
    map_ = tcod.map.Map(2,2)
    benchmark(map_.transparent.__getitem__, 0)
