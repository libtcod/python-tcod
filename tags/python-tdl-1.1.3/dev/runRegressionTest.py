#!/usr/bin/env python
import sys
import os

import unittest
import time
import random
import itertools
import gc

sys.path.insert(0, '..')
import tdl

ERROR_RANGE = 100 # a number to test out of bound errors

class TDLTemplate(unittest.TestCase):
    "Nearly all tests need tdl.init to be called"

    @classmethod
    def setUpClass(cls):
        tdl.setFont('../fonts/libtcod/terminal8x8_gs_ro.png')
        cls.console = tdl.init(30, 20, 'TDL UnitTest', False, renderer='SDL')

    def setUp(self):
        self.console.clear()
        
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
        RANGE = ERROR_RANGE # distance from bounds to test, just needs to be some moderate number
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
        #fg = (255, 255, 255)
        #bg = (0, 0, 0)
        #self.console.clear()
        #for x,y in self.getDrawables():
        #    self.assertEqual((ch, fg, bg), self.console.getChar(x, y), 'clear should default to white on black')
        
    def test_changeFonts(self):
        "Fonts are changable on the fly... kind of"
        FONT_DIR = '../fonts/X11'
        for font in os.listdir(FONT_DIR):
            if font[-4:] != '.png':
                continue # skip all those other files
            font = os.path.join(FONT_DIR, font)
            tdl.setFont(font)
            for x,y in self.getDrawables():
                self.console.drawChar(x, y, *self.getRandomCharacter())
            tdl.setTitle(font)
            tdl.flush()
            time.sleep(.25)
        
        
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
    
    #@unittest.skipIf(not __debug__, 'python run with optimized flag, skipping an AssertionError test')
    #def test_drawStrErrors(self):
    #    "test out of bounds assertion errors"
    #    for x,y in self.getUndrawables():
    #        with self.assertRaisesRegexp(AssertionError, r"\(%i, %i\)" % (x, y)):
    #            self.console.drawStr(x, y, 'foo', self.getRandomColor(), self.getRandomColor())
    
    def test_drawRect(self):
        consoleCopy = tdl.Console(*(self.console.getSize()))
        for x,y in self.getDrawables():
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.getRandomCharacter()
            width, height = self.console.getSize()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.drawRect(x, y, width, height, ch, fg, bg)
            for testX,testY in self.getDrawables():
                if x <= testX < x + width and y <= testY < y + height:
                    self.assertEqual(self.console.getChar(testX, testY), (ch, fg, bg), 'rectangle are should be overwritten')
                else:
                    self.assertEqual(self.console.getChar(testX, testY), consoleCopy.getChar(testX, testY), 'this area should remain untouched')
                    
    def test_drawFrame(self):
        consoleCopy = tdl.Console(*(self.console.getSize()))
        for x,y in self.getDrawables():
            consoleCopy.blit(self.console) # copy the console to compare untouched areas
            ch, fg, bg = self.getRandomCharacter()
            width, height = self.console.getSize()
            width, height = random.randint(1, width - x), random.randint(1, height - y)
            self.console.drawFrame(x, y, width, height, ch, fg, bg)
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
    
    @unittest.skip("Need this to be faster before unskipping")
    def test_scrolling(self):
        """marks a spot and then scrolls the console, checks to make sure no
        other spots are marked, test also knows if it's out of bounds.
        
        This test is a bit slow, it could be made more efficent by marking
        several areas and not clearing the console every loop.
        """
        for sx, sy in itertools.product(range(-30, 30, 5), range(-20, 20, 5)):
            self.console.clear()
            char = self.getRandomCharacter()
            dx, dy = random.choice(list(self.getDrawables()))
            self.console.drawChar(dx, dy, *char)
            self.console.scroll(sx, sy)
            dx += sx # if these go out of bounds then the check will make sure everything is cleared
            dy += sy
            for x, y in self.getDrawables():
                if x == dx and y == dy:
                    self.assertEqual(self.console.getChar(x, y), char, 'marked position should have scrolled here')
                else:
                    self.assertEqual(self.console.getChar(x, y), (0x20, (255, 255, 255), (0, 0, 0)), 'every other place should be clear')
        
        
def suite():
    loader = unittest.TestLoader()
    load = loader.loadTestsFromTestCase
    return unittest.TestSuite([load(BasicTests), load(DrawingTests)])

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner().run(suite)

