#!/usr/bin/env python
import sys
import unittest
import time
import random
import atexit
atexit.register(lambda: time.sleep(3))

sys.path.insert(0, '..')
import tdl

class SimpleConsoleTest(unittest.TestCase):

    def setUp(self):
        self.console = tdl.init(30, 20, 'TDL UnitTest')
        self.console2 = tdl.Console(30, 20)

    def tearDown(self):
        del self.console

class DrawCharTest(SimpleConsoleTest):

    def test_dcTuples(self):
        self.console2.clear()
        ch = (1, (255, 255, 255), (0, 0, 0))
        self.console.drawChar(0, 0, *ch)
        tdl.flush()
        # a critcal error with getChar prevents testing, I'll need to update libtcod
        # getChar crashes with the root console and returns garbage on any other console
        self.assertEqual(ch, self.console.getChar(0, 0), 'err?')
        
def suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([loader.loadTestsFromTestCase(DrawCharTest)])

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

