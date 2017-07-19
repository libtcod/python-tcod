#!/usr/bin/env python

import unittest
import itertools

import tdl

class MapTests(unittest.TestCase):

    MAP = (
           '############',
           '#   ###    #',
           '#   ###    #',
           '#   ### ####',
           '## #### # ##',
           '##      ####',
           '############',
           )

    WIDTH = len(MAP[0])
    HEIGHT = len(MAP)

    POINT_A = (2, 2)
    POINT_B = (9, 2)
    POINT_C = (9, 4)

    POINTS_AB = POINT_A + POINT_B
    POINTS_AC = POINT_A + POINT_C

    @classmethod
    def map_is_transparant(cls, x, y):
        try:
            return cls.MAP[y][x] == ' '
        except IndexError:
            return False

    @classmethod
    def path_cost(cls, src_x, src_y, dest_x, dest_y):
        if cls.map_is_transparant(dest_x, dest_y):
            return 1
        return 0


    def setUp(self):
        self.map = tdl.map.Map(self.WIDTH, self.HEIGHT)
        for y, line in enumerate(self.MAP):
            for x, ch in enumerate(line):
                trans = ch == ' '
                self.map.transparent[x,y] = self.map.walkable[x,y] = trans
                self.assertEquals(self.map.transparent[x,y], trans)
                self.assertEquals(self.map.walkable[x,y], trans)

    def test_map_compute_fov(self):
        fov = self.map.compute_fov(*self.POINT_A)
        self.assertTrue(list(fov), 'should be non-empty')
        fov = self.map.compute_fov(*self.POINT_A, fov='PERMISSIVE8')
        self.assertTrue(list(fov), 'should be non-empty')
        with self.assertRaises(tdl.TDLError):
            self.map.compute_fov(*self.POINT_A, fov='invalid option')

    def test_map_compute_path(self):
        self.assertTrue(self.map.compute_path(*self.POINTS_AB),
                        'should be non-empty')
        self.assertFalse(self.map.compute_path(*self.POINTS_AC),
                        'invalid path should return an empty list')

    def test_map_specials(self):
        for x,y in self.map:
            self.assertTrue((x, y) in self.map)
        self.assertFalse((-1, -1) in self.map)

    def test_quick_fov(self):
        fov = tdl.map.quick_fov(self.POINT_B[0], self.POINT_B[1],
                                self.map_is_transparant, radius=2.5)
        self.assertTrue(fov, 'should be non-empty')

    def test_bresenham(self):
        for x1, x2, y1, y2 in itertools.permutations([-4, -2, 4, 4], 4):
            self.assertTrue(tdl.map.bresenham(x1, x2, y1, y2),
                            'should be non-empty')

    def test_astar(self):
        pathfinder = tdl.map.AStar(self.WIDTH, self.HEIGHT,
                                   self.map_is_transparant)
        self.assertTrue(pathfinder.get_path(*self.POINTS_AB))

        pathfinder = tdl.map.AStar(self.WIDTH, self.HEIGHT,
                                   self.path_cost, None, True)
        self.assertTrue(pathfinder.get_path(*self.POINTS_AB))
        self.assertFalse(pathfinder.get_path(*self.POINTS_AC),
                         'invalid path should return an empty list')


def test_map_fov_cumulative():
    map_ = tdl.map.Map(3, 3)
    map_.transparent[::2,:] = True # Add map with small divider.
    assert not map_.fov.any()
    map_.compute_fov(1, 0, cumulative=True) # Light left side.
    assert map_.fov.any()
    assert not map_.fov.all()
    map_.compute_fov(1, 2, cumulative=True) # Light both sides.
    assert map_.fov.all()
