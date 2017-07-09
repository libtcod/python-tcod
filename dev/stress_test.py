#!/usr/bin/env python
"""
    Stress test designed to test tdl under harsh conditions

    Note that one of the slower parts is converting colors over ctypes so you
    can get a bit of speed avoiding the fgcolor and bgcolor parameters.
    Another thing slowing things down are mass calls to ctypes functions.
    Eventually these will be optimized to work better.
"""
import sys

import itertools
import random
import time

sys.path.insert(0, '../')
import tdl

class StopWatch:
    "Simple tool used to count time within a block using the with statement"
    MAXSNAPSHOTS = 10

    def __init__(self):
        self.enterTime = None
        self.snapshots = []

    def __enter__(self):
        self.enterTime = time.clock()

    def __exit__(self, *exc):
        self.snapshots.append(time.clock() - self.enterTime)
        if len(self.snapshots) > self.MAXSNAPSHOTS:
            self.snapshots.pop(0)

    def getMeanTime(self):
        if not self.snapshots:
            return 0
        return sum(self.snapshots) / len(self.snapshots)


class TestApp(tdl.event.App):

    def __init__(self, console):
        self.console = console
        self.writer = self.console
        self.console.setMode('scroll')
        self.width, self.height = self.console.getSize()
        self.total = self.width * self.height
        self.cells = list(itertools.product(range(self.width), range(self.height)))
        self.tick = 0
        self.init()

    def init(self):
        pass

    def ev_MOUSEDOWN(self, event):
        self.suspend()

    def update(self, deltaTime):
        self.updateTest(deltaTime)
        self.tick += 1
        #if self.tick % 50 == 0:
        tdl.setTitle('%s: %i FPS' % (self.__class__.__name__, tdl.getFPS()))
        tdl.flush()

class FullDrawCharTest(TestApp):

    def updateTest(self, deltaTime):
        # getrandbits is around 5x faster than using randint
        bgcolors = [(random.getrandbits(7), random.getrandbits(7), random.getrandbits(7)) for _ in range(self.total)]
        char = [random.getrandbits(8) for _ in range(self.total)]
        fgcolor = (255, 255, 255)
        for (x,y), bgcolor, char in zip(self.cells, bgcolors, char):
            self.console.draw_char(x, y, char, fgcolor, bgcolor)

class PreCompiledColorTest(TestApp):
    """
        This test was to see if the cast to the ctypes Color object was a big
        impact on performance, turns out there's no hit with this class.
    """

    def init(self):
        self.bgcolors = [tdl._Color(random.getrandbits(7),
                                    random.getrandbits(7),
                                    random.getrandbits(7))
                         for _ in range(256)]
        self.fgcolor = tdl._Color(255, 255, 255)

    def updateTest(self, deltaTime):
        # getrandbits is around 5x faster than using randint
        char = [random.getrandbits(8) for _ in range(self.total)]
        for (x,y), char in zip(self.cells, char):
            self.console.draw_char(x, y, char,
                                   self.fgcolor, random.choice(self.bgcolors))


class CharOnlyTest(TestApp):

    def updateTest(self, deltaTime):
        self.console.clear((255, 255, 255), (0, 0, 0))
        char = [random.getrandbits(8) for _ in range(self.total)]
        for (x,y), char in zip(self.cells, char):
            self.console.draw_char(x, y, char)

class TypewriterCharOnlyTest(TestApp):

    def updateTest(self, deltaTime):
        self.writer.move(0, 0)
        char = [random.getrandbits(8) for _ in range(self.total)]
        for (x,y), char in zip(self.cells, char):
            self.writer.move(x, y)
            self.writer.print_str(chr(char))

class ColorOnlyTest(TestApp):

    def updateTest(self, deltaTime):
        # getrandbits is around 5x faster than using randint
        bgcolors = [(random.getrandbits(6), random.getrandbits(6), random.getrandbits(6)) for _ in range(self.total)]
        for (x,y), bgcolor in zip(self.cells, bgcolors):
            self.console.draw_char(x, y, None, None, bgcolor)

class GetCharTest(TestApp):

    def init(self):
        for (x,y) in self.cells:
            bgcolor = (random.getrandbits(6), random.getrandbits(6), random.getrandbits(6))
            ch = random.getrandbits(8)
            self.console.get_char(x, y, ch, bgcolor=bgcolor)

    def updateTest(self, deltaTime):
        for (x,y) in self.cells:
            self.console.draw_char(x, y, *self.console.get_char(x, y))

class SingleRectTest(TestApp):

    def updateTest(self, deltaTime):
        bgcolor = (random.getrandbits(6), random.getrandbits(6), random.getrandbits(6))
        self.console.draw_rect(0, 0, None, None, ' ', (255, 255, 255), bgcolor)

class DrawStrTest(TestApp):

    def updateTest(self, deltaTime):
        for y in range(self.height):
            bgcolor = (random.getrandbits(6), random.getrandbits(6), random.getrandbits(6))
            string = [random.getrandbits(8) for x in range(self.width)]
            self.console.draw_str(0, y, string, (255, 255, 255), bgcolor)

class BlitScrollTest(TestApp):
    def updateTest(self, deltaTime):
        self.console.scroll(0, 1)
        for x in range(self.width):
            bgcolor = (random.getrandbits(6), random.getrandbits(6), random.getrandbits(6))
            ch = random.getrandbits(8)
            self.console.draw_char(x, 0, ch, bg=bgcolor)

# match libtcod sample screen
WIDTH = 46
HEIGHT = 20
def main():
    console = tdl.init(46, 20, renderer='OPENGL')
    for Test in [FullDrawCharTest, CharOnlyTest, TypewriterCharOnlyTest, ColorOnlyTest, GetCharTest,
                 SingleRectTest, DrawStrTest, BlitScrollTest]:
        Test(console).run()
        console.clear()

if __name__ == '__main__':
    main()
