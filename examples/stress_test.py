"""
    Stress test designed to test tdl under harsh conditions
    
    Note that one of the slower parts is converting colors over ctypes so you
    can get a bit of speed avoiding the fgcolor and bgcolor parameters.
    Another thing slowing things down are mass calls to ctypes functions.
    Eventually these will be optimized to work better.
"""
import sys
sys.path.insert(0, '../')

import random
import cProfile

import tdl

def main():

    WIDTH = 80
    HEIGHT = 60

    console = tdl.init(WIDTH, HEIGHT)

    while 1:
        for event in tdl.event.get():
            if event.type == tdl.QUIT:
                raise SystemExit()
        for y in range(HEIGHT):
            bgcolor = (random.randint(0, 64), random.randint(0, 64), random.randint(0, 64))
            for x in range(WIDTH):
                console.drawChar(random.randint(0, 255), fgcolor=(255, 255, 255), bgcolor=bgcolor)
        tdl.flush()
        tdl.setTitle('%i FPS' % tdl.getFPS())

if __name__ == '__main__':
    main()
    