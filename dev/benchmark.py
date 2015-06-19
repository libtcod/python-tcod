#!/usr/bin/env python

from __future__ import print_function

import time
import platform

import tdl

WIDTH = 80 # must be divisible by 16
HEIGHT = 48

RENDERER = 'OpenGL'

log = None

def print_result(string):
    print(string)
    print(string, file=log)

class Benchmark:
    default_frames = 100
        
    def run(self, console, frames=None, times=4):
        if times > 1:
            print_result('Running %s' % self.__class__.__name__)
            while times > 0:
                self.run(console, frames, times=1)
                times -= 1
            print_result('')
            return
        if frames is None:
            frames = self.default_frames
        self.total_frames = 0
        self.tiles = 0
        console.clear()
        self.start_time = time.time()
        while self.total_frames < frames:
            self.total_frames += 1
            self.test(console)
            for event in tdl.event.get():
                if event.type == 'QUIT':
                    raise SystemExit('Benchmark Canceled')
        self.total_time = time.time() - self.start_time
        self.tiles_per_second = self.tiles / self.total_time
        print_result(
            '%i tiles drawn in %.2f seconds, %.2f characters/ms, %.2f FPS' %
            (self.tiles, self.total_time,self.tiles_per_second / 1000,
             self.total_frames / self.total_time))
        
    def test(self, console):
        for x,y in console:
            console.draw_char(x, y, '.')
            tiles += 1
        tdl.flush()
        
        
class Benchmark_DrawChar_DefaultColor(Benchmark):
    
    def test(self, console):
        for x,y in console:
            console.draw_char(x, y, 'A')
            self.tiles += 1
        tdl.flush()

        
class Benchmark_DrawChar_NoColor(Benchmark):
    
    def test(self, console):
        for x,y in console:
            console.draw_char(x, y, 'B', None, None)
            self.tiles += 1
        tdl.flush()

        
class Benchmark_DrawStr16_DefaultColor(Benchmark):
    default_frames = 100
    
    def test(self, console):
        for y in range(HEIGHT):
            for x in range(0, WIDTH, 16):
                console.draw_str(x, y, '0123456789ABCDEF')
                self.tiles += 16
        tdl.flush()

        
class Benchmark_DrawStr16_NoColor(Benchmark):
    default_frames = 100
    
    def test(self, console):
        for y in range(HEIGHT):
            for x in range(0, WIDTH, 16):
                console.draw_str(x, y, '0123456789ABCDEF', None, None)
                self.tiles += 16
        tdl.flush()

def run_benchmark():
    global log
    log = open('results.log', 'a')
    print('', file=log)
    console = tdl.init(WIDTH, HEIGHT, renderer=RENDERER)
    
    print_result('Benchmark run on %s' % time.ctime())
    print_result('Running under %s %s' % (platform.python_implementation(),
                                          platform.python_version()))
    print_result('In %s mode' % (['release', 'debug'][__debug__]))
    print_result('%i characters/frame' % (WIDTH * HEIGHT))
    print_result('Opened console in %s mode' % RENDERER)
    Benchmark_DrawChar_DefaultColor().run(console)
    Benchmark_DrawChar_NoColor().run(console)
    Benchmark_DrawStr16_DefaultColor().run(console)
    Benchmark_DrawStr16_NoColor().run(console)
    log.close()
    print('results written to results.log')
    
if __name__ == '__main__':
    run_benchmark()
