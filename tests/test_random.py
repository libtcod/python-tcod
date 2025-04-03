"""Test random number generators."""

import copy
import pickle
from pathlib import Path

import pytest

import tcod.random

# ruff: noqa: D103

SCRIPT_DIR = Path(__file__).parent


def test_tcod_random() -> None:
    rand = tcod.random.Random(tcod.random.COMPLEMENTARY_MULTIPLY_WITH_CARRY)
    assert 0 <= rand.randint(0, 100) <= 100  # noqa: PLR2004
    assert 0 <= rand.uniform(0, 100) <= 100  # noqa: PLR2004
    with pytest.warns(FutureWarning, match=r"typo"):
        rand.guass(0, 1)
    with pytest.warns(FutureWarning, match=r"typo"):
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
    rand: tcod.random.Random = pickle.loads((SCRIPT_DIR / "data/random_v13.pkl").read_bytes())
    assert rand.randint(0, 0xFFFF) == 56422  # noqa: PLR2004
    assert rand.randint(0, 0xFFFF) == 15795  # noqa: PLR2004
