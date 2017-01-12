
import copy
import pickle

import pytest

import tcod

def test_tcod_random(benchmark):
    rand = tcod.random.Random(tcod.random.COMPLEMENTARY_MULTIPLY_WITH_CARRY)
    assert 0 <= rand.randint(0, 100) <= 100
    assert 0 <= rand.uniform(0, 100) <= 100
    rand.guass(0, 1)
    rand.inverse_guass(0, 1)
    benchmark(rand.uniform, 0, 1)

def test_tcod_random_copy():
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER)
    rand2 = copy.copy(rand)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)

def test_tcod_random_pickle():
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER)
    rand2 = pickle.loads(pickle.dumps(rand))
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
