#!/usr/bin/env python

import os

import tcod as libtcod

curdir = os.path.dirname(__file__)

FONT_FILE = os.path.join(curdir, 'data/fonts/consolas10x10_gs_tc.png')

#def test_console():
#    libtcod.console_set_custom_font(FONT_FILE, libtcod.FONT_LAYOUT_TCOD)
#    libtcod.console_init_root(40, 30, 'test', False, libtcod.RENDERER_SDL)
#    libtcod.console_flush()
