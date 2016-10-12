#!/usr/bin/env python

import shutil
import unittest
import tempfile

try:
    import numpy
except ImportError:
    numpy = None

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
        self.pad = tcod.console_new(self.WIDTH, self.HEIGHT)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        tcod.console_flush()
        tcod.console_delete(self.pad)
        tcod.console_delete(self.console)
        shutil.rmtree(self.temp_dir)

    def test_console_info(self):
        self.assertEqual(tcod.console_get_width(self.console), self.WIDTH)
        self.assertEqual(tcod.console_get_height(self.console), self.HEIGHT)
        self.assertEqual(tcod.console_is_fullscreen(), self.FULLSCREEN)
        tcod.console_set_window_title(self.TITLE)
        tcod.console_is_window_closed()

    @unittest.skip('takes too long')
    def test_credits_long(self):
        tcod.console_credits()

    def test_credits(self):
        tcod.console_credits_render(0, 0, True)
        tcod.console_credits_reset()

    FG = (0, 255, 255)
    BG = (64, 0, 0)

    def test_console_defaults(self):
        # defaults
        tcod.console_set_default_background(self.console, self.BG)
        tcod.console_set_default_foreground(self.console, self.FG)
        tcod.console_clear(self.console)

    def test_console_character_drawing(self):
        tcod.console_set_char_background(self.console, 0, 0,
                                         self.BG, tcod.BKGND_SET)
        tcod.console_set_char_foreground(self.console, 0, 0, self.FG)
        tcod.console_set_char(self.console, 0, 0, '@')
        tcod.console_put_char(self.console, 0, 0, '$', tcod.BKGND_SET)
        tcod.console_put_char_ex(self.console, 0, 0, '$',
                                 self.FG, self.BG)

    def test_console_printing(self):
        tcod.console_set_background_flag(self.console, tcod.BKGND_SET)
        self.assertEquals(tcod.console_get_background_flag(self.console),
                          tcod.BKGND_SET)

        tcod.console_set_alignment(self.console, tcod.LEFT)
        self.assertEquals(tcod.console_get_alignment(self.console), tcod.LEFT)

        tcod.console_print(self.console, 0, 0, 'print')
        tcod.console_print_ex(self.console, 0, 0, tcod.BKGND_SET, tcod.LEFT,
                              'print ex')

        self.assertIsInstance(tcod.console_print_rect(self.console, 0, 0, 8, 8,
                                                      'print rect'), int)
        self.assertIsInstance(
            tcod.console_print_rect_ex(self.console, 0, 0, 8, 8,
                tcod.BKGND_SET, tcod.LEFT, 'print rect ex'), int)

        self.assertIsInstance(tcod.console_get_height_rect(self.console,
                                                           0, 0, 8, 8,
                                                           'get height'), int)

        tcod.console_set_color_control(tcod.COLCTRL_1, self.FG, self.BG)

    def test_console_printing_advanced(self):
        tcod.console_rect(self.console, 0, 0, 4, 4, False, tcod.BKGND_SET)
        tcod.console_hline(self.console, 0, 0, 4)
        tcod.console_vline(self.console, 0, 0, 4)
        tcod.console_print_frame(self.console, 0, 0, 11, 11)

    def test_console_contents(self):
        self.assertIsInstance(tcod.console_get_default_background(self.console),
                              tcod.Color)
        self.assertIsInstance(tcod.console_get_default_foreground(self.console),
                              tcod.Color)

        tcod.console_get_char_background(self.console, 0, 0)
        tcod.console_get_char_foreground(self.console, 0, 0)
        tcod.console_get_char(self.console, 0, 0)

    def test_console_fade(self):
        tcod.console_set_fade(255, (0, 0, 0))
        self.assertIsInstance(tcod.console_get_fade(), int)
        tcod.console_get_fading_color()

    def assertConsolesEqual(self, a, b):
        for y in range(tcod.console_get_height(a)):
            for x in range(tcod.console_get_width(a)):
                self.assertEquals(tcod.console_get_char_background(a, x, y),
                                  tcod.console_get_char_background(b, x, y))
                self.assertEquals(tcod.console_get_char_foreground(a, x, y),
                                  tcod.console_get_char_foreground(b, x, y))
                self.assertEquals(tcod.console_get_char(a, x, y),
                                  tcod.console_get_char(b, x, y))


    def test_console_blit(self):
        tcod.console_print(self.pad, 0, 0, 'test')
        tcod.console_blit(self.pad, 0, 0, 0, 0, self.console, 0, 0, 1, 1)
        self.assertConsolesEqual(self.console, self.pad)
        tcod.console_set_key_color(self.pad, (0, 0, 0))

    def test_console_asc_read_write(self):
        tcod.console_print(self.console, 0, 0, 'test')

        asc_file = tempfile.mktemp(dir=self.temp_dir)
        print(asc_file)
        tcod.console_save_asc(self.console, asc_file)
        self.assertTrue(tcod.console_load_asc(self.pad, asc_file))
        self.assertConsolesEqual(self.console, self.pad)

    def test_console_apf_read_write(self):
        tcod.console_print(self.console, 0, 0, 'test')

        apf_file = tempfile.mktemp(dir=self.temp_dir)
        tcod.console_save_apf(self.console, apf_file)
        self.assertTrue(tcod.console_load_apf(self.pad, apf_file))
        self.assertConsolesEqual(self.console, self.pad)

    def test_console_fullscreen(self):
        tcod.console_set_fullscreen(False)

    def test_console_key_input(self):
        self.assertIsInstance(tcod.console_check_for_keypress(), tcod.Key)
        tcod.console_is_key_pressed(tcod.KEY_ENTER)

        tcod.console_set_keyboard_repeat(100, 10)
        tcod.console_disable_keyboard_repeat()

    def test_console_fill_errors(self):
        with self.assertRaises(TypeError):
            tcod.console_fill_background(self.console, [0], [], [])
        with self.assertRaises(TypeError):
            tcod.console_fill_foreground(self.console, [0], [], [])

    def test_console_fill(self):
        fill = [0] * self.HEIGHT * self.WIDTH
        tcod.console_fill_background(self.console, fill, fill, fill)
        tcod.console_fill_foreground(self.console, fill, fill, fill)
        tcod.console_fill_char(self.console, fill)

    @unittest.skipUnless(numpy, 'requires numpy module')
    def test_console_fill_numpy(self):
        fill = numpy.zeros((self.WIDTH, self.HEIGHT), dtype=numpy.intc)
        tcod.console_fill_background(self.console, fill, fill, fill)
        tcod.console_fill_foreground(self.console, fill, fill, fill)
        tcod.console_fill_char(self.console, fill)

    def test_console_buffer(self):
        buffer = tcod.ConsoleBuffer(self.WIDTH, self.HEIGHT)
        buffer = buffer.copy()
        buffer.set_fore(0, 0, 0, 0, 0, '@')
        buffer.set_back(0, 0, 0, 0, 0)
        buffer.set(0, 0, 0, 0, 0, 0, 0, 0, '@')
        buffer.blit(self.console)

    def test_console_buffer_error(self):
        buffer = tcod.ConsoleBuffer(0, 0)
        with self.assertRaises(ValueError):
            buffer.blit(self.console)

    def test_console_font_mapping(self):
        tcod.console_map_ascii_code_to_font('@', 0, 0)
        tcod.console_map_ascii_codes_to_font('@', 1, 0, 0)
        tcod.console_map_string_to_font('@', 0, 0)

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

    def test_map(self):
        map = tcod.map_new(16, 16)
        self.assertEqual(tcod.map_get_width(map), 16)
        self.assertEqual(tcod.map_get_height(map), 16)
        tcod.map_copy(map, map)
        tcod.map_clear(map)
        tcod.map_set_properties(map, 0, 0, True, True)
        self.assertEqual(tcod.map_is_transparent(map, 0, 0), True)
        self.assertEqual(tcod.map_is_walkable(map, 0, 0), True)
        tcod.map_is_in_fov(map, 0, 0)
        tcod.map_delete(map)
