#!/usr/bin/env python
import sys
import os

import unittest
import time
import random
import itertools
import copy
import pickle
import gc

sys.path.insert(0, '..')
import tdl

#ERROR_RANGE = 100 # a number to test out of bound errors
WIDTH, HEIGHT = 30, 20
WINWIDTH, WINHEIGHT = 10, 10

DEFAULT_CHAR = (0x20, (0, 0, 0), (0, 0, 0))

BLACK = tdl.Color(0, 0, 0)

class TDLTemplate(unittest.TestCase):
    "Nearly all tests need tdl.init to be called"

    @classmethod
    def setUpClass(cls):
        tdl.setFont('../fonts/libtcod/terminal8x8_gs_ro.png')
        cls.console = tdl.init(WIDTH, HEIGHT, 'TDL UnitTest', False, renderer='SDL')
        # make a small window in the corner
        cls.window = tdl.Window(cls.console, 0, 0, WINWIDTH, WINHEIGHT)

    def setUp(self):
        tdl.setFont('../fonts/libtcod/terminal8x8_gs_ro.png')
        tdl.event.get()
        self.console.set_colors(BLACK, BLACK)
        self.console.clear()
        
    @classmethod
    def tearDownClass(cls):
        del cls.console
        gc.collect() # make sure console.__del__ is called quickly
        
    def inWindow(self, x, y):
        "returns True if this point is in the Window"
        return 0 <= x < WINWIDTH and 0 <= y < WINHEIGHT
        
    def randomizeConsole(self):
        "Randomize the console returning the random data"
        noise = [((x, y), self.getRandomCharacter()) for x,y in self.getDrawables()]
        for (x, y), graphic in noise:
            self.console.drawChar(x, y, *graphic)
        return noise # [((x, y), (cg, fg, bg)), ...]
        
    def flush(self):
        'Pump events and refresh screen so show progress'
        #tdl.event.get() # no longer needed
        tdl.flush()
        
    def getRandomCharacter(self):
        "returns a tuple with a random character and colors (ch, fg, bg)"
        return (random.getrandbits(8), self.getRandomColor(), self.getRandomColor())
        
    def getRandomColor(self):
        "returns a single random color"
        return tdl.Color((random.getrandbits(8),
                          random.getrandbits(8),
                          random.getrandbits(8)))
        
    def getDrawables(self, console=None):
        """return a list of all drawable (x,y) positions
        defaults to self.console
        """
        if console is None:
            console = self.console
        w, h = console.getSize()
        return itertools.product(range(w), range(h))
    
    def getUndrawables(self, console=None):
        """return a list of (x,y) positions that should raise errors when used
        positions are mostly random and will have at least one over the bounds of each side and each corner"""
        if console is None:
            console = self.console
        w, h = console.getSize()
        for y in range(-1, h+1):
            yield -w-1, y
            yield w, y
        for x in range(0, w):
            yield x, h
            yield x, -h-1
        
    def compareConsoles(self, consoleA, consoleB, errorMsg='colors should be the same'):
        "Compare two console assuming they match and failing if they don't"
        self.assertEqual(consoleA.getSize(), consoleB.getSize(), 'consoles should be the same size')
        for x, y in self.getDrawables(consoleA):
            self.assertEqual(consoleA.getChar(x, y),
                             consoleB.getChar(x, y), '%s, position: (%i, %i)' % (errorMsg, x, y))

class BasicTests(TDLTemplate):
    
    def test_clearConsole(self):
        self.randomizeConsole()
        _, fg, bg = self.getRandomCharacter()
        ch = 0x20 # space
        self.console.clear(fg, bg)
        self.flush()
        for x,y in self.getDrawables():
            self.assertEqual((ch, fg, bg), self.console.getChar(x, y), 'color should be changed with clear')
        _, fg2, bg2 = self.getRandomCharacter()
        self.window.clear(fg2, bg2)
        self.flush()
        for x,y in self.getDrawables():
            if self.inWindow(x, y):
                self.assertEqual((ch, fg2, bg2), self.console.getChar(x, y), 'color in window should be changed')
            else:
                self.assertEqual((ch, fg, bg), self.console.getChar(x, y), 'color outside of window should persist')
        
    def test_cloneConsole(self):
        noiseData = self.randomizeConsole()
        clone = copy.copy(self.console)
        self.compareConsoles(self.console, clone, 'console clone should match root console')
    
    def test_pickleConsole(self):
        noiseData = self.randomizeConsole()
        pickled = pickle.dumps(self.console)
        clone = pickle.loads(pickled)
        self.compareConsoles(self.console, clone, 'pickled console should match root console')
     
    # This isn't really supported.
    #def test_changeFonts(self):
    #    "Fonts are changable on the fly... kind of"
    #    FONT_DIR = '../fonts/X11'
    #    for font in os.listdir(FONT_DIR):
    #        if font[-4:] != '.png':
    #            continue # skip all those other files
    #        font = os.path.join(FONT_DIR, font)
    #        tdl.setFont(font)
    #        # only works at all in OPENGL
    #        self.console = tdl.init(WIDTH, HEIGHT, title=font, renderer='OPENGL')
    #        for x,y in self.getDrawables():
    #            self.console.draw_char(x, y, *self.getRandomCharacter())
    #        self.flush()
    #        time.sleep(.05)
        
        
class DrawingTests(TDLTemplate):

    def test_drawCharTuples(self):
        "Test passing tuple colors and int characters to drawChar"
        record = {}
        for x,y in self.getDrawables():
            ch, fg, bg = self.getRandomCharacter()
            record[x,y] = (ch, fg, bg)
            self.console.drawChar(x, y, ch, fg, bg)
            self.assertEqual(record[x,y], self.console.getChar(x, y), 'console data should be overwritten')
            self.flush() # show progress
            
        for (x,y), data in record.items():
            self.assertEqual(data, self.console.getChar(x, y), 'drawChar should not overwrite any other tiles')

    def test_drawCharWebcolor(self):
        "Test passing web style colors and string characters to drawChar"
        record = {}
        for x,y in self.getDrawables():
            ch, fg, bg = self.getRandomCharacter()
            record[x,y] = (ch, fg, bg)
            ch = chr(ch)
            fg = fg[0] << 16 | fg[1] << 8 | fg[2] # convert to a 0xRRGGBB style number
            bg = bg[0] << 16 | bg[1] << 8 | bg[2]
            self.console.drawChar(x, y, ch, fg, bg)
            self.assertEqual(record[x,y], self.console.getChar(x, y), 'console data should be overwritten')
            self.flush() # show progress
        for (x,y), data in record.items():
            self.assertEqual(data, self.console.getChar(x, y), 'drawChar should not overwrite any other tiles')
        
    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_drawCharErrors(self):
    #    "test out of bounds assertion errors"
    #    for x,y in self.getUndrawables():
    #        with self.assertRaisesRegexp(AssertionError, r"\(%i, %i\)" % (x, y)):
    #            self.console.drawChar(x, y, *(self.getRandomCharacter()))
        
    def test_drawStrArray(self):
        """strings will raise errors if they pass over the end of the console.
        The data will still be written however."""
        width, height = self.console.getSize()
        for x,y in self.getDrawables():
            string = [random.getrandbits(8) for _ in range(random.randint(2, 10))]
            fg, bg = self.getRandomColor(), self.getRandomColor()
            if len(string) > ((height - y) * width - x): # compare length of string to remaining space on the console
                with self.assertRaises(tdl.TDLError): # expect end of console error
                    self.console.drawStr(x, y, string, fg, bg)
            else:
                self.console.drawStr(x, y, string, fg, bg)
            for ch in string: # inspect console for changes
                self.assertEqual(self.console.getChar(x, y), (ch, fg, bg), 'console data should be overwritten, even after an error')
                x += 1
                if x == width:
                    x = 0
                    y += 1
                    if y == height:
                        break # end of console
            self.flush() # show progress
    
    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_drawStrErrors(self):
    #    "test out of bounds assertion errors"
    #    for x,y in self.getUndrawables():
    #        with self.assertRaisesRegexp(AssertionError, r"\(%i, %i\)" % (x, y)):
    #            self.console.drawStr(x, y, 'foo', self.getRandomColor(), self.getRandomColor())
    
    def test_drawRect(self):
        consoleCopy = tdl.Console(*(self.console.getSize()))
        for x,y in random.sample(list(self.getDrawables()), 20):
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.getRandomCharacter()
            width, height = self.console.getSize()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.drawRect(x, y, width, height, ch, fg, bg)
            self.flush() # show progress
            for testX,testY in self.getDrawables():
                if x <= testX < x + width and y <= testY < y + height:
                    self.assertEqual(self.console.getChar(testX, testY), (ch, fg, bg), 'rectangle area should be overwritten')
                else:
                    self.assertEqual(self.console.getChar(testX, testY), consoleCopy.getChar(testX, testY), 'this area should remain untouched')
                    
    def test_drawFrame(self):
        consoleCopy = tdl.Console(*(self.console.getSize()))
        for x,y in random.sample(list(self.getDrawables()), 20):
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.getRandomCharacter()
            width, height = self.console.getSize()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.drawFrame(x, y, width, height, ch, fg, bg)
            self.flush() # show progress
            for testX,testY in self.getDrawables():
                if x + 1 <= testX < x + width - 1 and y + 1 <= testY < y + height - 1:
                    self.assertEqual(self.console.getChar(testX, testY), consoleCopy.getChar(testX, testY), 'inner frame should remain untouched')
                elif x <= testX < x + width and y <= testY < y + height:
                    self.assertEqual(self.console.getChar(testX, testY), (ch, fg, bg), 'frame area should be overwritten')
                else:
                    self.assertEqual(self.console.getChar(testX, testY), consoleCopy.getChar(testX, testY), 'outer frame should remain untouched')
    
    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_drawRectFrameErrors(self):
    #    for x,y in self.getDrawables():
    #        ch, fg, bg = self.getRandomCharacter()
    #        width, height = self.console.getSize()
    #        width, height = random.randint(x + width, x + width + ERROR_RANGE), random.randint(y + height, y + height + ERROR_RANGE)
    #        with self.assertRaises(AssertionError):
    #            self.console.drawRect(x, y, width, height, ch, fg, bg)
    #        with self.assertRaises(AssertionError):
    #            self.console.drawFrame(x, y, width, height, ch, fg, bg)
    
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
            noiseData = dict(self.randomizeConsole())
            self.console.set_colors(BLACK, BLACK)
            self.console.scroll(sx, sy)
            self.flush() # show progress
            for x, y in self.getDrawables():
                nX = x - sx
                nY = y - sy
                if (nX, nY) in noiseData:
                    self.assertEqual(self.console.getChar(x, y), noiseData[nX, nY], 'random noise should be scrolled')
                else:
                    self.assertEqual(self.console.getChar(x, y), DEFAULT_CHAR, 'scrolled away positions should be clear')
        
        
def suite():
    loader = unittest.TestLoader()
    load = loader.loadTestsFromTestCase
    return unittest.TestSuite([load(BasicTests), load(DrawingTests)])

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner().run(suite)

