#!/usr/bin/env python
import sys
import unittest
import time
import random
import itertools
import gc

sys.path.insert(0, '..')
import tdl

class TDLTemplate(unittest.TestCase):
    "Nearly all tests need tdl.init to be called"

    @classmethod
    def setUpClass(cls):
        cls.console = tdl.init(30, 20, 'TDL UnitTest', False, renderer=tdl.RENDERER_SDL)

    @classmethod
    def tearDownClass(cls):
        del cls.console
        gc.collect() # make sure console.__del__ is called quickly
        
    def getRandomCharacter(self):
        "returns a tuple with a random character and colors (ch, fg, bg)"
        return (random.getrandbits(8), self.getRandomColor(), self.getRandomColor())
        
    def getRandomColor(self):
        "returns a single random color"
        return (random.getrandbits(8), random.getrandbits(8), random.getrandbits(8))
        
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
        RANGE = 1000 # distance from bounds to test, just needs to be some moderate number
        results = []
        for _ in range(8):
            for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1), # every side
                        (-1, -1), (-1, 1), (1, 1), (1, -1)]: # every corner
                if x == -1:
                    x = random.randint(-RANGE, -1) # over left bound
                elif x == 1:
                    x = random.randint(w + 1, w + RANGE) # over right bound
                else:
                    x = random.randint(0, w) # within bounds
                if y == -1:
                    y = random.randint(-RANGE, -1)
                elif y == 1:
                    y = random.randint(h + 1, h + RANGE)
                else:
                    y = random.randint(0, h)
                results.append((x, y))
        return results

class BasicTests(TDLTemplate):
    
    def test_clearConsole(self):
        _, fg, bg = self.getRandomCharacter()
        ch = 0x20 # space
        self.console.clear(fg, bg)
        for x,y in self.getDrawables():
            self.assertEqual((ch, fg, bg), self.console.getChar(x, y), 'color should be changeable with clear')
        fg = (255, 255, 255)
        bg = (0, 0, 0)
        self.console.clear()
        for x,y in self.getDrawables():
            self.assertEqual((ch, fg, bg), self.console.getChar(x, y), 'clear should default to white on black')
        
        
class DrawingTests(TDLTemplate):

    def test_drawCharTuples(self):
        "Test passing tuple colors and int characters to drawChar"
        record = {}
        for x,y in self.getDrawables():
            ch, fg, bg = self.getRandomCharacter()
            record[x,y] = (ch, fg, bg)
            self.console.drawChar(x, y, ch, fg, bg)
            self.assertEqual(record[x,y], self.console.getChar(x, y), 'console data should be overwritten')
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
        for (x,y), data in record.items():
            self.assertEqual(data, self.console.getChar(x, y), 'drawChar should not overwrite any other tiles')
        
    def test_drawCharErrors(self):
        "test out of bounds assertion errors"
        #if not __debug__:
        #    self.skipTest('python run with optimized flag, skipping an AssertionError test')
        for x,y in self.getUndrawables():
            with self.assertRaisesRegexp(tdl.TDLError, r"\(%i, %i\)" % (x, y)):
                self.console.drawChar(x, y, *(self.getRandomCharacter()))
        
    def test_drawStrArray(self):
        for x,y in self.getDrawables():
            string = (random.getrandbits(8) for _ in range(random.randint(1, 100)))
            self.console.drawStr(x, y, string, self.getRandomColor(), self.getRandomColor())
        
def suite():
    loader = unittest.TestLoader()
    load = loader.loadTestsFromTestCase
    return unittest.TestSuite([load(BasicTests), load(DrawingTests)])

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

