#!/usr/bin/env python

import unittest

from common import tcod

class TestLibtcodpyConsole(unittest.TestCase):

    FONT_FILE = 'libtcod/terminal.png'
    WIDTH = 12
    HEIGHT = 12
    TITLE = 'libtcod-cffi tests'
    FULLSCREEN = False
    RENDERER = tcod.RENDERER_SDL

    def setUp(self):
        tcod.console_set_custom_font(self.FONT_FILE)
        self.console = tcod.console_init_root(self.WIDTH, self.HEIGHT,
                                              self.TITLE, self.FULLSCREEN,
                                              self.RENDERER)

    def tearDown(self):
        tcod.console_flush()
        tcod.console_delete(self.console)

    def test_console_info(self):
        self.assertEqual(tcod.console_get_width(self.console), self.WIDTH)
        self.assertEqual(tcod.console_get_height(self.console), self.HEIGHT)
        self.assertEqual(tcod.console_is_fullscreen(), self.FULLSCREEN)
        tcod.console_set_window_title(self.TITLE)
        tcod.console_is_window_closed()


class TestLibtcodpy(unittest.TestCase):
    # arguments to test with and the results expected from these arguments
    LINE_ARGS = (-5, 0, 5, 10)
    EXCLUSIVE_RESULTS = [(-4, 1), (-3, 2), (-2, 3), (-1, 4), (0, 5), (1, 6),
                         (2, 7), (3, 8), (4, 9), (5, 10)]
    INCLUSIVE_RESULTS = [(-5, 0)] + EXCLUSIVE_RESULTS

    def test_line_step(self):
        """
        tcod.line_init and tcod.line_step
        """
        tcod.line_init(*self.LINE_ARGS)
        for expected_xy in self.EXCLUSIVE_RESULTS:
            self.assertEqual(tcod.line_step(), expected_xy)
        self.assertEqual(tcod.line_step(), (None, None))

    def test_line(self):
        """
        tests normal use, lazy evaluation, and error propagation
        """
        # test normal results
        test_result = []
        def line_test(*test_xy):
            test_result.append(test_xy)
            return 1
        self.assertEqual(tcod.line(*self.LINE_ARGS,
                                   py_callback=line_test), 1)
        self.assertEqual(test_result, self.INCLUSIVE_RESULTS)

        # test lazy evaluation
        test_result = []
        def return_false(*test_xy):
            test_result.append(test_xy)
            return False
        self.assertEqual(tcod.line(*self.LINE_ARGS,
                                        py_callback=return_false), 0)
        self.assertEqual(test_result, self.INCLUSIVE_RESULTS[:1])

    def test_line_iter(self):
        """
        tcod.line_iter
        """
        self.assertEqual(list(tcod.line_iter(*self.LINE_ARGS)),
                         self.EXCLUSIVE_RESULTS)

    def test_bsp(self):
        """
        cover bsp deprecated functions
        """
        bsp = tcod.bsp_new_with_size(0, 0, 64, 64)
        print(bsp) # test __repr__ on leaf
        tcod.bsp_resize(bsp, 0, 0, 32, 32)
        self.assertNotEqual(bsp, None)

        # test getter/setters
        bsp.x = bsp.x
        bsp.y = bsp.y
        bsp.w = bsp.w
        bsp.h = bsp.h
        bsp.position = bsp.position
        bsp.horizontal = bsp.horizontal
        bsp.level = bsp.level

        # cover functions on leaf
        self.assertFalse(tcod.bsp_left(bsp))
        self.assertFalse(tcod.bsp_right(bsp))
        self.assertFalse(tcod.bsp_father(bsp))
        self.assertTrue(tcod.bsp_is_leaf(bsp))

        self.assertTrue(tcod.bsp_contains(bsp, 1, 1))
        self.assertFalse(tcod.bsp_contains(bsp, -1, -1))
        self.assertEqual(tcod.bsp_find_node(bsp, 1, 1), bsp)
        self.assertFalse(tcod.bsp_find_node(bsp, -1, -1))

        tcod.bsp_split_once(bsp, False, 4)
        print(bsp) # test __repr__ with parent
        tcod.bsp_split_once(bsp, True, 4)
        print(bsp)

        # cover functions on parent
        self.assertTrue(tcod.bsp_left(bsp))
        self.assertTrue(tcod.bsp_right(bsp))
        self.assertFalse(tcod.bsp_father(bsp))
        self.assertFalse(tcod.bsp_is_leaf(bsp))
        self.assertEqual(tcod.bsp_father(tcod.bsp_left(bsp)), bsp)
        self.assertEqual(tcod.bsp_father(tcod.bsp_right(bsp)), bsp)

        tcod.bsp_split_recursive(bsp, None, 4, 2, 2, 1.0, 1.0)

        # cover bsp_traverse
        def traverse(node, user_data):
            return True

        tcod.bsp_traverse_pre_order(bsp, traverse)
        tcod.bsp_traverse_in_order(bsp, traverse)
        tcod.bsp_traverse_post_order(bsp, traverse)
        tcod.bsp_traverse_level_order(bsp, traverse)
        tcod.bsp_traverse_inverted_level_order(bsp, traverse)

        # test __repr__ on deleted node
        son = tcod.bsp_left(bsp)
        tcod.bsp_remove_sons(bsp)
        print(son)

        tcod.bsp_delete(bsp)
