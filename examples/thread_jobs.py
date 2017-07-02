#!/usr/bin/env python

import sys

import multiprocessing
import platform
import threading
try:
    import Queue as queue
except ImportError:
    import queue
import timeit

import tcod
#import libtcodpy as tcod

THREADS = multiprocessing.cpu_count()

MAP_WIDTH = 100
MAP_HEIGHT = 100
MAP_NUMBER = 500

PATH_NUMBER = 500

class JobConsumer(threading.Thread):

    def __init__(self, i):
        threading.Thread.__init__(self)
        self.daemon = True
        self.astar = tcod.path_new_using_map(maps[i])

    def run(self):
        while 1:
            job, obj = jobs.get()
            if job == 'fov':
                tcod.map_compute_fov(obj, MAP_WIDTH // 2, MAP_HEIGHT // 2)
            elif job == 'astar':
                tcod.path_compute(self.astar,
                                  0, 0, MAP_WIDTH - 1 , MAP_HEIGHT - 1)
                x, y = tcod.path_walk(self.astar, False)
                while x is not None:
                    x, y = tcod.path_walk(self.astar, False)
            jobs.task_done()

maps = [tcod.map_new(MAP_WIDTH, MAP_HEIGHT) for i in range(MAP_NUMBER)]
jobs = queue.Queue()
threads = [JobConsumer(i) for i in range(THREADS)]

def test_fov_single():
    for m in maps:
        tcod.map_compute_fov(m, MAP_WIDTH // 2, MAP_HEIGHT // 2)

def test_fov_threads():
    for m in maps:
        jobs.put(('fov', m))
    jobs.join()

def test_astar_single():
    astar = tcod.path_new_using_map(maps[0])
    for _ in range(PATH_NUMBER):
        tcod.path_compute(astar, 0, 0, MAP_WIDTH - 1 , MAP_HEIGHT - 1)
        x, y = tcod.path_walk(astar, False)
        while x is not None:
            x, y = tcod.path_walk(astar, False)

def test_astar_threads():
    for _ in range(PATH_NUMBER):
        jobs.put(('astar', None))
    jobs.join()

def main():
    for m in maps:
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                tcod.map_set_properties(m, x, y, True, True)

    for thread in threads:
        thread.start()

    print('Python %s\n%s\n%s' % (sys.version, platform.platform(),
                                   platform.processor()))

    print('\nComputing field-of-view for %i empty %ix%i maps.' %
          (len(maps), MAP_WIDTH, MAP_HEIGHT))
    single_time = min(timeit.repeat(test_fov_single, number=1))
    print('1 thread: %.2fms' % (single_time * 1000))

    multi_time = min(timeit.repeat(test_fov_threads, number=1))
    print('%i threads: %.2fms' % (THREADS, multi_time * 1000))
    print('%.2f%% efficiency' %
          (single_time / (multi_time * THREADS) * 100))

    print('\nComputing AStar from corner to corner %i times on seperate empty'
          ' %ix%i maps.' % (PATH_NUMBER, MAP_WIDTH, MAP_HEIGHT))
    single_time = min(timeit.repeat(test_astar_single, number=1))
    print('1 thread: %.2fms' % (single_time * 1000))

    multi_time = min(timeit.repeat(test_astar_threads, number=1))
    print('%i threads: %.2fms' % (THREADS, multi_time * 1000))
    print('%.2f%% efficiency' %
          (single_time / (multi_time * THREADS) * 100))

if __name__ == '__main__':
    main()
