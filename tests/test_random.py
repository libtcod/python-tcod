import copy
import pathlib
import pickle

import tcod


def test_tcod_random() -> None:
    rand = tcod.random.Random(tcod.random.COMPLEMENTARY_MULTIPLY_WITH_CARRY)
    assert 0 <= rand.randint(0, 100) <= 100
    assert 0 <= rand.uniform(0, 100) <= 100
    rand.guass(0, 1)
    rand.inverse_guass(0, 1)


def test_tcod_random_copy() -> None:
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER)
    rand2 = copy.copy(rand)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)


def test_tcod_random_pickle() -> None:
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER)
    rand2 = pickle.loads(pickle.dumps(rand))
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)
    assert rand.uniform(0, 1) == rand2.uniform(0, 1)


def test_load_rng_v13_1() -> None:
    with open(pathlib.Path(__file__).parent / "data/random_v13.pkl", "rb") as f:
        rand: tcod.random.Random = pickle.load(f)
    assert rand.randint(0, 0xFFFF) == 56422
    assert rand.randint(0, 0xFFFF) == 15795
