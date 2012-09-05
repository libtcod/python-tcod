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
        

def main():

    WIDTH = 80
    HEIGHT = 60
    TOTAL = WIDTH * HEIGHT
    # a list containing all x,y pairs
    CELLS = list(itertools.product(range(WIDTH), range(HEIGHT)))
    
    console = tdl.init(WIDTH, HEIGHT)
    fullTimer = StopWatch()
    randomTimer = StopWatch()
    drawTimer = StopWatch()
    flushTimer = StopWatch()
    while 1:
        for event in tdl.event.get():
            if event.type == tdl.QUIT:
                raise SystemExit()
        with randomTimer:
            # getrandbits is around 5x faster than using randint
            bgcolors = [(random.getrandbits(6), random.getrandbits(6), random.getrandbits(6)) for _ in range(TOTAL)]
            char = [random.getrandbits(8) for _ in range(TOTAL)]
        with drawTimer:
            for (x,y), bgcolor, char in zip(CELLS, bgcolors, char):
                console.drawChar(char=char,  x=x, y=y, fgcolor=(255, 255, 255), bgcolor=bgcolor)
        console.drawStr('Random%7.2fms ' % (randomTimer.getMeanTime() * 1000), 0, 0, tdl.C_WHITE, tdl.C_BLACK)
        console.drawStr('DrawCh%7.2fms ' % (drawTimer.getMeanTime() * 1000), 0, 1, tdl.C_WHITE, tdl.C_BLACK)
        console.drawStr('Flush %7.2fms ' % (flushTimer.getMeanTime() * 1000), 0, 2, tdl.C_WHITE, tdl.C_BLACK)
        with flushTimer:
            tdl.flush()
        tdl.setTitle('%i FPS' % tdl.getFPS())

if __name__ == '__main__':
    main()
    