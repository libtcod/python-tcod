#!/usr/bin/env python

import unittest

from common import tcod, raise_Exception

class TestLibtcodpy(unittest.TestCase):

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
