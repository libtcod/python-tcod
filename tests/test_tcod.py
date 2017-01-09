#!/usr/bin/env python

import unittest

from common import tcod, raise_Exception

class TestTCOD(unittest.TestCase):

    def test_line_error(self):
        """
        test exception propagation
        """
        with self.assertRaises(Exception):
            tcod.line(*LINE_ARGS, py_callback=self.raise_Exception)

    def test_clipboard(self):
        tcod.clipboard_set('')
        tcod.clipboard_get()

    def test_tcod_bsp(self):
        """
        test tcod additions to BSP
        """
        bsp = tcod.BSP(0, 0, 32, 32)

        self.assertEqual(bsp.level, 0)
        self.assertFalse(bsp.horizontal)
        self.assertFalse(bsp.children)

        with self.assertRaises(Exception):
            tcod.bsp_traverse_pre_order(bsp, raise_Exception)

        bsp.split_recursive(3, 4, 4, 1, 1)
        for node in bsp.walk():
            self.assertIsInstance(node, tcod.BSP)

        self.assertFalse(bsp == 'asd')

        # test that operations on deep BSP nodes preserve depth
        sub_bsp = bsp.children[0]
        sub_bsp.split_recursive(3, 2, 2, 1, 1)
        self.assertEqual(sub_bsp.children[0].level, 2)

class TestTCODConsole(unittest.TestCase):

    FONT_FILE = 'libtcod/terminal.png'
    WIDTH = 12
    HEIGHT = 10
    TITLE = 'libtcod-cffi tests'
    FULLSCREEN = False
    RENDERER = tcod.RENDERER_SDL

    @classmethod
    def setUpClass(cls):
        tcod.console_set_custom_font(cls.FONT_FILE)
        cls.console = tcod.console_init_root(cls.WIDTH, cls.HEIGHT,
                                             cls.TITLE, cls.FULLSCREEN,
                                             cls.RENDERER)

    @classmethod
    def tearDownClass(cls):
        tcod.console_delete(cls.console)

    def test_array_read_write(self):
        FG = (255, 254, 253)
        BG = (1, 2, 3)
        CH = ord('&')
        tcod.console_put_char_ex(self.console, 0, 0, CH, FG, BG)
        self.assertEqual(self.console.ch[0, 0], CH)
        self.assertEqual(tuple(self.console.fg[0, 0]), FG)
        self.assertEqual(tuple(self.console.bg[0, 0]), BG)

        tcod.console_put_char_ex(self.console, 1, 2, CH, FG, BG)
        self.assertEqual(self.console.ch[2, 1], CH)
        self.assertEqual(tuple(self.console.fg[2, 1]), FG)
        self.assertEqual(tuple(self.console.bg[2, 1]), BG)

        self.console.clear()
        self.assertEqual(self.console.ch[0, 0], ord(' '))
        self.assertEqual(tuple(self.console.fg[0, 0]), (255, 255, 255))
        self.assertEqual(tuple(self.console.bg[0, 0]), (0, 0, 0))

        ch_slice = self.console.ch[1, :]
        ch_slice[2] = CH
        self.console.fg[1, ::2] = FG
        self.console.bg[...] = BG

        self.assertEqual(tcod.console_get_char(self.console, 2, 1), CH)
        self.assertEqual(
            tuple(tcod.console_get_char_foreground(self.console, 2, 1)), FG)
        self.assertEqual(
            tuple(tcod.console_get_char_background(self.console, 2, 1)), BG)

    def test_console_defaults(self):
        self.console.default_bg = [2, 3, 4]
        self.assertEqual(self.console.default_bg, (2, 3, 4))

        self.console.default_fg = (4, 5, 6)
        self.assertEqual(self.console.default_fg, (4, 5, 6))

        self.console.default_blend = tcod.BKGND_ADD
        self.assertEqual(self.console.default_blend, tcod.BKGND_ADD)

        self.console.default_alignment = tcod.RIGHT
        self.assertEqual(self.console.default_alignment, tcod.RIGHT)
