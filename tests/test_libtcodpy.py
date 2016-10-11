#!/usr/bin/env python

import unittest

import tcod

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


class TestLibtcodpyLine(unittest.TestCase):
    # arguments to test with and the results expected from these arguments
    LINE_ARGS = (-5, 0, 5, 10)
    EXCLUSIVE_RESULTS = [(-4, 1), (-3, 2), (-2, 3), (-1, 4), (0, 5), (1, 6),
                         (2, 7), (3, 8), (4, 9), (5, 10)]
    INCLUSIVE_RESULTS = [(-5, 0)] + EXCLUSIVE_RESULTS

    def test_step(self):
        ''' tcod.line_init and tcod.line_step
        '''
        tcod.line_init(*self.LINE_ARGS)
        for expected_xy in self.EXCLUSIVE_RESULTS:
            self.assertEqual(tcod.line_step(), expected_xy)
        self.assertEqual(tcod.line_step(), (None, None))

    def raise_error(self, *args):
        raise Exception()

    def test_line(self):
        ''' tcod.line: normal, lazy, error

        tests normal use, lazy evaluation, and error propagation
        '''
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

        # exception propagation
        with self.assertRaises(Exception):
            tcod.line(*LINE_ARGS, py_callback=self.raise_error)

    def test_line_iter(self):
        ''' tcod.line_iter
        '''
        self.assertEqual(list(tcod.line_iter(*self.LINE_ARGS)),
                         self.EXCLUSIVE_RESULTS)
