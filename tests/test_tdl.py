#!/usr/bin/env python
import sys
import os

import unittest
import random
import itertools
import copy
import pickle
import gc

import tdl

#ERROR_RANGE = 100 # a number to test out of bound errors
WIDTH, HEIGHT = 30, 20
WINWIDTH, WINHEIGHT = 10, 10

DEFAULT_CHAR = (0x20, (0, 0, 0), (0, 0, 0))

IS_PYTHON2 = (sys.version_info[0] == 2)

class TDLTemplate(unittest.TestCase):
    "Nearly all tests need tdl.init to be called"

    @classmethod
    def setUpClass(cls):
        cls.console = tdl.init(WIDTH, HEIGHT, 'TDL UnitTest', False, renderer='GLSL')
        # make a small window in the corner
        cls.window = tdl.Window(cls.console, 0, 0, WINWIDTH, WINHEIGHT)

    def setUp(self):
        tdl.event.get()
        self.console.set_colors((0,0,0), (0,0,0))
        self.console.clear()

    @classmethod
    def tearDownClass(cls):
        del cls.console
        gc.collect() # make sure console.__del__ is called quickly

    def in_window(self, x, y):
        "returns True if this point is in the Window"
        return 0 <= x < WINWIDTH and 0 <= y < WINHEIGHT

    def randomize_console(self):
        "Randomize the console returning the random data"
        noise = [((x, y), self.get_random_character()) for x,y in self.get_drawables()]
        for (x, y), graphic in noise:
            self.console.draw_char(x, y, *graphic)
        return noise # [((x, y), (cg, fg, bg)), ...]

    def flush(self):
        'Pump events and refresh screen so show progress'
        #tdl.event.get() # no longer needed
        tdl.flush()

    def get_random_character(self):
        "returns a tuple with a random character and colors (ch, fg, bg)"
        return (random.getrandbits(8), self.get_random_color(), self.get_random_color())

    def get_random_color(self):
        "returns a single random color"
        return (random.getrandbits(8), random.getrandbits(8), random.getrandbits(8))

    def get_drawables(self, console=None):
        """return a list of all drawable (x,y) positions
        defaults to self.console
        """
        if console is None:
            console = self.console
        w, h = console.get_size()
        return itertools.product(range(w), range(h))

    def get_undrawables(self, console=None):
        """return a list of (x,y) positions that should raise errors when used
        positions are mostly random and will have at least one over the bounds of each side and each corner"""
        if console is None:
            console = self.console
        w, h = console.get_size()
        for y in range(-1, h+1):
            yield -w-1, y
            yield w, y
        for x in range(0, w):
            yield x, h
            yield x, -h-1

    def compare_consoles(self, consoleA, consoleB, errorMsg='colors should be the same'):
        "Compare two console assuming they match and failing if they don't"
        self.assertEqual(consoleA.get_size(), consoleB.get_size(), 'consoles should be the same size')
        for x, y in self.get_drawables(consoleA):
            self.assertEqual(consoleA.get_char(x, y),
                             consoleB.get_char(x, y), '%s, position: (%i, %i)' % (errorMsg, x, y))

class BasicTests(TDLTemplate):

    def test_clearConsole(self):
        self.randomize_console()
        _, fg, bg = self.get_random_character()
        ch = 0x20 # space
        self.console.clear(fg, bg)
        self.flush()
        for x,y in self.get_drawables():
            self.assertEqual((ch, fg, bg), self.console.get_char(x, y), 'color should be changed with clear')
        _, fg2, bg2 = self.get_random_character()
        self.window.clear(fg2, bg2)
        self.flush()
        for x,y in self.get_drawables():
            if self.in_window(x, y):
                self.assertEqual((ch, fg2, bg2), self.console.get_char(x, y), 'color in window should be changed')
            else:
                self.assertEqual((ch, fg, bg), self.console.get_char(x, y), 'color outside of window should persist')

    def test_cloneConsole(self):
        noiseData = self.randomize_console()
        clone = copy.copy(self.console)
        self.compare_consoles(self.console, clone, 'console clone should match root console')

    def test_pickleConsole(self):
        noiseData = self.randomize_console()
        pickled = pickle.dumps(self.console)
        clone = pickle.loads(pickled)
        self.compare_consoles(self.console, clone, 'pickled console should match root console')


class DrawingTests(TDLTemplate):

    def test_draw_charTuples(self):
        "Test passing tuple colors and int characters to draw_char"
        record = {}
        for x,y in self.get_drawables():
            ch, fg, bg = self.get_random_character()
            record[x,y] = (ch, fg, bg)
            self.console.draw_char(x, y, ch, fg, bg)
            self.assertEqual(record[x,y], self.console.get_char(x, y), 'console data should be overwritten')
            self.flush() # show progress

        for (x,y), data in record.items():
            self.assertEqual(data, self.console.get_char(x, y), 'draw_char should not overwrite any other tiles')

    def test_draw_charWebcolor(self):
        "Test passing web style colors and string characters to draw_char"
        record = {}
        for x,y in self.get_drawables():
            ch, fg, bg = self.get_random_character()
            record[x,y] = (ch, fg, bg)
            ch = chr(ch)
            fg = fg[0] << 16 | fg[1] << 8 | fg[2] # convert to a 0xRRGGBB style number
            bg = bg[0] << 16 | bg[1] << 8 | bg[2]
            self.console.draw_char(x, y, ch, fg, bg)
            self.assertEqual(record[x,y], self.console.get_char(x, y), 'console data should be overwritten')
            self.flush() # show progress
        for (x,y), data in record.items():
            self.assertEqual(data, self.console.get_char(x, y), 'draw_char should not overwrite any other tiles')

    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_draw_charErrors(self):
    #    "test out of bounds assertion errors"
    #    for x,y in self.get_undrawables():
    #        with self.assertRaisesRegexp(AssertionError, r"\(%i, %i\)" % (x, y)):
    #            self.console.draw_char(x, y, *(self.get_random_character()))

    def test_draw_str(self):
        """quick regression test for draw_str"""
        width, height = self.console.get_size()
        def str_check(array, string, desc):
            fg, bg = self.get_random_color(), self.get_random_color()
            self.console.clear()
            self.console.draw_str(0, 0, string, fg, bg)
            self.flush()
            i = 0
            for y in range(height):
                for x in range(width):
                    self.assertEqual(self.console.get_char(x, y), (array[i], fg, bg),
                                     '%s should be written out' % desc)
                    i += 1

        # array of numbers
        array = [random.getrandbits(8) for _ in range(width * height)]
        str_check(array, array, 'array of numbers')

        # array of strings
        #array = [random.getrandbits(8) for _ in range(width * height)]
        #array_str = [chr(c) for c in array]
        #str_check(array, array_str, 'array of characters')

        # standard string
        array = [random.getrandbits(8) for _ in range(width * height)]
        string = ''.join((chr(c) for c in array))
        str_check(array, string, 'standatd string')

        # Unicode string - Python 2
        if IS_PYTHON2:
            array = [random.getrandbits(7) for _ in range(width * height)]
            ucode = unicode().join((chr(c) for c in array))
            str_check(array, ucode, 'Unicode string')


    def test_draw_strArray(self):
        """strings will raise errors if they pass over the end of the console.
        The data will still be written however."""
        width, height = self.console.get_size()
        for x,y in self.get_drawables():
            string = [random.getrandbits(8) for _ in range(random.randint(2, 10))]
            fg, bg = self.get_random_color(), self.get_random_color()
            if len(string) > ((height - y) * width - x): # compare length of string to remaining space on the console
                with self.assertRaises(tdl.TDLError): # expect end of console error
                    self.console.draw_str(x, y, string, fg, bg)
            else:
                self.console.draw_str(x, y, string, fg, bg)
            for ch in string: # inspect console for changes
                self.assertEqual(self.console.get_char(x, y), (ch, fg, bg), 'console data should be overwritten, even after an error')
                x += 1
                if x == width:
                    x = 0
                    y += 1
                    if y == height:
                        break # end of console
            self.flush() # show progress

    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_draw_strErrors(self):
    #    "test out of bounds assertion errors"
    #    for x,y in self.get_undrawables():
    #        with self.assertRaisesRegexp(AssertionError, r"\(%i, %i\)" % (x, y)):
    #            self.console.draw_str(x, y, 'foo', self.get_random_color(), self.get_random_color())

    def test_draw_rect(self):
        consoleCopy = tdl.Console(*(self.console.get_size()))
        for x,y in random.sample(list(self.get_drawables()), 20):
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.get_random_character()
            width, height = self.console.get_size()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.draw_rect(x, y, width, height, ch, fg, bg)
            self.flush() # show progress
            for testX,testY in self.get_drawables():
                if x <= testX < x + width and y <= testY < y + height:
                    self.assertEqual(self.console.get_char(testX, testY), (ch, fg, bg), 'rectangle area should be overwritten')
                else:
                    self.assertEqual(self.console.get_char(testX, testY), consoleCopy.get_char(testX, testY), 'this area should remain untouched')

    def test_draw_frame(self):
        consoleCopy = tdl.Console(*(self.console.get_size()))
        for x,y in random.sample(list(self.get_drawables()), 20):
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.get_random_character()
            width, height = self.console.get_size()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.draw_frame(x, y, width, height, ch, fg, bg)
            self.flush() # show progress
            for testX,testY in self.get_drawables():
                if x + 1 <= testX < x + width - 1 and y + 1 <= testY < y + height - 1:
                    self.assertEqual(self.console.get_char(testX, testY), consoleCopy.get_char(testX, testY), 'inner frame should remain untouched')
                elif x <= testX < x + width and y <= testY < y + height:
                    self.assertEqual(self.console.get_char(testX, testY), (ch, fg, bg), 'frame area should be overwritten')
                else:
                    self.assertEqual(self.console.get_char(testX, testY), consoleCopy.get_char(testX, testY), 'outer frame should remain untouched')

    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_draw_rectFrameErrors(self):
    #    for x,y in self.get_drawables():
    #        ch, fg, bg = self.get_random_character()
    #        width, height = self.console.get_size()
    #        width, height = random.randint(x + width, x + width + ERROR_RANGE), random.randint(y + height, y + height + ERROR_RANGE)
    #        with self.assertRaises(AssertionError):
    #            self.console.draw_rect(x, y, width, height, ch, fg, bg)
    #        with self.assertRaises(AssertionError):
    #            self.console.draw_frame(x, y, width, height, ch, fg, bg)

    #@unittest.skip("Need this to be faster before unskipping")
    def test_scrolling(self):
        """marks a spot and then scrolls the console, checks to make sure no
        other spots are marked, test also knows if it's out of bounds.

        This test is a bit slow, it could be made more efficent by marking
        several areas and not clearing the console every loop.
        """
        scrollTests = set([(0, 0), (WIDTH, HEIGHT)]) # include zero and out of bounds
        while len(scrollTests) < 10: # add 3 more randoms
            scrollTests.add((random.randint(-WIDTH, WIDTH),
                             random.randint(-HEIGHT, HEIGHT)))
        for sx, sy in scrollTests:
            noiseData = dict(self.randomize_console())
            self.console.set_colors((0, 0, 0), (0, 0, 0))
            self.console.scroll(sx, sy)
            self.flush() # show progress
            for x, y in self.get_drawables():
                nX = x - sx
                nY = y - sy
                if (nX, nY) in noiseData:
                    self.assertEqual(self.console.get_char(x, y), noiseData[nX, nY], 'random noise should be scrolled')
                else:
                    self.assertEqual(self.console.get_char(x, y), DEFAULT_CHAR, 'scrolled away positions should be clear')


def test_fps():
    tdl.set_fps(0)
    tdl.get_fps()

def suite():
    loader = unittest.TestLoader()
    load = loader.loadTestsFromTestCase
    return unittest.TestSuite([load(BasicTests), load(DrawingTests)])

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner().run(suite)

