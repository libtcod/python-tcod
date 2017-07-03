#!/usr/bin/env python

import copy
import pickle
import unittest

import tdl

class NoiseTests(unittest.TestCase):

    def test_noise(self):
        n = tdl.noise.Noise()
        self.assertIsInstance(n.get_point(0, 0), float)
        n = tdl.noise.Noise('Wavelet', 'FBM', seed=-1)
        self.assertIsInstance(n.get_point(0, 0), float)

    def test_noise_exceptions(self):
        with self.assertRaises(tdl.TDLError):
            tdl.noise.Noise(algorithm='')
        with self.assertRaises(tdl.TDLError):
            tdl.noise.Noise(mode='')

    def test_noise_copy(self):
        self.assertIsInstance(copy.copy(tdl.noise.Noise()), tdl.noise.Noise)

    def test_noise_pickle(self):
        self.assertIsInstance(pickle.loads(pickle.dumps(tdl.noise.Noise())),
                              tdl.noise.Noise)


