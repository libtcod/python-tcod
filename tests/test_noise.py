import copy
import pickle

import numpy as np
import pytest

import tcod


@pytest.mark.parametrize("implementation", tcod.noise.Implementation)
@pytest.mark.parametrize("algorithm", tcod.noise.Algorithm)
@pytest.mark.parametrize("hurst", [0.5, 0.75])
@pytest.mark.parametrize("lacunarity", [2, 3])
@pytest.mark.parametrize("octaves", [4, 6])
def test_noise_class(
    implementation: tcod.noise.Implementation,
    algorithm: tcod.noise.Algorithm,
    hurst: float,
    lacunarity: float,
    octaves: float,
) -> None:
    noise = tcod.noise.Noise(
        2,
        algorithm=algorithm,
        implementation=implementation,
        hurst=hurst,
        lacunarity=lacunarity,
        octaves=octaves,
    )
    # cover attributes
    assert noise.dimensions == 2
    noise.algorithm = noise.algorithm
    noise.implementation = noise.implementation
    noise.octaves = noise.octaves
    assert noise.hurst
    assert noise.lacunarity

    assert noise.get_point(0, 0) == noise[0, 0]
    assert noise[0] == noise[0, 0]
    noise.sample_mgrid(np.mgrid[:2, :3])
    noise.sample_ogrid(np.ogrid[:2, :3])

    np.testing.assert_equal(
        noise.sample_mgrid(np.mgrid[:2, :3]),
        noise.sample_ogrid(np.ogrid[:2, :3]),
    )
    np.testing.assert_equal(noise.sample_mgrid(np.mgrid[:2, :3]), noise[tuple(np.mgrid[:2, :3])])
    repr(noise)


def test_noise_samples() -> None:
    noise = tcod.noise.Noise(2, tcod.noise.Algorithm.SIMPLEX, tcod.noise.Implementation.SIMPLE)
    np.testing.assert_equal(
        noise.sample_mgrid(np.mgrid[:32, :24]),
        noise.sample_ogrid(np.ogrid[:32, :24]),
    )


def test_noise_errors() -> None:
    with pytest.raises(ValueError):
        tcod.noise.Noise(0)
    with pytest.raises(ValueError):
        tcod.noise.Noise(1, implementation=-1)
    noise = tcod.noise.Noise(2)
    with pytest.raises(ValueError):
        noise.sample_mgrid(np.mgrid[:2, :2, :2])
    with pytest.raises(ValueError):
        noise.sample_ogrid(np.ogrid[:2, :2, :2])
    with pytest.raises(IndexError):
        noise[0, 0, 0, 0, 0]
    with pytest.raises(TypeError):
        noise[object]


@pytest.mark.parametrize("implementation", tcod.noise.Implementation)
def test_noise_pickle(implementation: tcod.noise.Implementation) -> None:
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER, 42)
    noise = tcod.noise.Noise(2, implementation, seed=rand)
    noise2 = copy.copy(noise)
    np.testing.assert_equal(
        noise.sample_ogrid(np.ogrid[:3, :1]),
        noise2.sample_ogrid(np.ogrid[:3, :1]),
    )


def test_noise_copy() -> None:
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER, 42)
    noise = tcod.noise.Noise(2, seed=rand)
    noise2 = pickle.loads(pickle.dumps(noise))
    np.testing.assert_equal(
        noise.sample_ogrid(np.ogrid[:3, :1]),
        noise2.sample_ogrid(np.ogrid[:3, :1]),
    )

    noise3 = tcod.noise.Noise(2, seed=None)
    assert repr(noise3) == repr(pickle.loads(pickle.dumps(noise3)))
    noise4 = tcod.noise.Noise(2, seed=42)
    assert repr(noise4) == repr(pickle.loads(pickle.dumps(noise4)))
