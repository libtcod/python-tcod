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

    WIDTH = 60
    HEIGHT = 40
    TOTAL = WIDTH * HEIGHT
    # a list containing all x,y pairs
    CELLS = list(itertools.product(range(WIDTH), range(HEIGHT)))
    
    console = tdl.init(WIDTH, HEIGHT, renderer='GLSL')
    fullTimer = StopWatch()
    randomTimer = StopWatch()
    drawTimer = StopWatch()
    flushTimer = StopWatch()
    while 1:
        for event in tdl.event.get():
            if event.type == 'QUIT':
                raise SystemExit()
        with randomTimer:
            # getrandbits is around 5x faster than using randint
            bgcolors = [(random.getrandbits(6), random.getrandbits(6), random.getrandbits(6)) for _ in range(TOTAL)]
            #bgcolors = [(random.getrandbits(24)) for _ in range(TOTAL)]
            char = [random.getrandbits(8) for _ in range(TOTAL)]
        with drawTimer:
            for (x,y), bgcolor, char in zip(CELLS, bgcolors, char):
                console.drawChar(x=x, y=y, char=char, fgcolor=(255, 255, 255), bgcolor=bgcolor)
        console.drawStr(0, 0, 'Random%7.2fms ' % (randomTimer.getMeanTime() * 1000))
        console.drawStr(0, 1, 'DrawCh%7.2fms ' % (drawTimer.getMeanTime() * 1000))
        console.drawStr(0, 2, 'Flush %7.2fms ' % (flushTimer.getMeanTime() * 1000))
        with flushTimer:
            tdl.flush()
        tdl.setTitle('%i FPS' % tdl.getFPS())

if __name__ == '__main__':
    main()
    