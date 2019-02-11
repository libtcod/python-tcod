#!/usr/bin/env python

import copy
import pickle

import numpy as np
import pytest

from common import tcod, raise_Exception
import tcod.noise
import tcod.path


def test_line_error():
    """
    test exception propagation
    """
    with pytest.raises(Exception):
        tcod.line(*LINE_ARGS, py_callback=raise_Exception)


@pytest.mark.filterwarnings("ignore:Iterate over nodes using")
@pytest.mark.filterwarnings("ignore:Use pre_order method instead of walk.")
def test_tcod_bsp():
    """
    test tcod additions to BSP
    """
    bsp = tcod.bsp.BSP(0, 0, 32, 32)

    assert bsp.level == 0
    assert not bsp.horizontal
    assert not bsp.children

    with pytest.raises(Exception):
        tcod.bsp_traverse_pre_order(bsp, raise_Exception)

    bsp.split_recursive(3, 4, 4, 1, 1)
    for node in bsp.walk():
        assert isinstance(node, tcod.bsp.BSP)

    assert bsp != 'asd'

    # test that operations on deep BSP nodes preserve depth
    sub_bsp = bsp.children[0]
    sub_bsp.split_recursive(3, 2, 2, 1, 1)
    assert sub_bsp.children[0].level == 2

    # cover find_node method
    assert bsp.find_node(0, 0)
    assert bsp.find_node(-1, -1) is None

    # cover __str__
    str(bsp)


@pytest.mark.filterwarnings("ignore:Use map.+ to check for this")
def test_tcod_map_set_bits():
    map_ = tcod.map.Map(2,2)

    assert map_.transparent[:].any() == False
    assert map_.walkable[:].any() == False
    assert map_.fov[:].any() == False

    map_.transparent[1, 0] = True
    assert tcod.map_is_transparent(map_, 0, 1) == True
    map_.walkable[1, 0] = True
    assert tcod.map_is_walkable(map_, 0, 1) == True
    map_.fov[1, 0] = True
    assert tcod.map_is_in_fov(map_, 0, 1) == True


def test_tcod_map_get_bits():
    map_ = tcod.map.Map(2,2)
    map_.transparent[0]


def test_tcod_map_copy():
    map_ = tcod.map.Map(3, 3)
    map_.transparent[:] = True
    assert (map_.transparent.tolist() == copy.copy(map_).transparent.tolist())


def test_tcod_map_pickle():
    map_ = tcod.map.Map(3, 3)
    map_.transparent[:] = True
    map2 = pickle.loads(pickle.dumps(copy.copy(map_)))
    assert (map_.transparent[:].tolist() == map2.transparent[:].tolist())


def test_tcod_map_pickle_fortran():
    map_ = tcod.map.Map(2, 3, order='F')
    map2 = pickle.loads(pickle.dumps(copy.copy(map_)))
    assert map_._Map__buffer.strides == map2._Map__buffer.strides
    assert map_.transparent.strides == map2.transparent.strides
    assert map_.walkable.strides == map2.walkable.strides
    assert map_.fov.strides == map2.fov.strides


@pytest.mark.parametrize('implementation', [tcod.noise.SIMPLE,
                                            tcod.noise.FBM,
                                            tcod.noise.TURBULENCE])
def test_noise_class(implementation):
    noise = tcod.noise.Noise(2, tcod.NOISE_SIMPLEX, implementation)
    # cover attributes
    assert noise.dimensions == 2
    noise.algorithm = noise.algorithm
    noise.implementation = noise.implementation
    noise.octaves = noise.octaves
    assert noise.hurst
    assert noise.lacunarity

    noise.get_point(0, 0)
    noise.sample_mgrid(np.mgrid[:2,:3])
    noise.sample_ogrid(np.ogrid[:2,:3])


def test_noise_samples():
    noise = tcod.noise.Noise(2, tcod.NOISE_SIMPLEX, tcod.noise.SIMPLE)
    np.testing.assert_equal(
        noise.sample_mgrid(np.mgrid[:32,:24]),
        noise.sample_ogrid(np.ogrid[:32,:24]),
        )


def test_noise_errors():
    with pytest.raises(ValueError):
        tcod.noise.Noise(0)
    with pytest.raises(ValueError):
        tcod.noise.Noise(1, implementation=-1)
    noise = tcod.noise.Noise(2)
    with pytest.raises(ValueError):
        noise.sample_mgrid(np.mgrid[:2,:2,:2])
    with pytest.raises(ValueError):
        noise.sample_ogrid(np.ogrid[:2,:2,:2])


@pytest.mark.parametrize('implementation',
    [tcod.noise.SIMPLE, tcod.noise.FBM, tcod.noise.TURBULENCE])
def test_noise_pickle(implementation):
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER, 42)
    noise = tcod.noise.Noise(2, implementation, seed=rand)
    noise2 = copy.copy(noise)
    assert (noise.sample_ogrid(np.ogrid[:3,:1]) ==
            noise2.sample_ogrid(np.ogrid[:3,:1])).all()


def test_noise_copy():
    rand = tcod.random.Random(tcod.random.MERSENNE_TWISTER, 42)
    noise = tcod.noise.Noise(2, seed=rand)
    noise2 = pickle.loads(pickle.dumps(noise))
    assert (noise.sample_ogrid(np.ogrid[:3,:1]) ==
            noise2.sample_ogrid(np.ogrid[:3,:1])).all()


def test_color_class():
    assert tcod.black == tcod.black
    assert tcod.black == (0, 0, 0)
    assert tcod.black == [0, 0, 0]
    assert tcod.black != tcod.white
    assert tcod.white * 1 == tcod.white
    assert tcod.white * tcod.black == tcod.black
    assert tcod.white - tcod.white == tcod.black
    assert tcod.black + (2, 2, 2) - (1, 1, 1) == (1, 1, 1)
    assert not tcod.black == None

    color = tcod.Color()
    color.r = 1
    color.g = 2
    color.b = 3
    assert color == (1, 2, 3)


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32,
                                   np.uint8, np.uint16, np.uint32, np.float32])
def test_path_numpy(dtype):
    map_np = np.ones((6, 6), dtype=dtype)
    map_np[1:4, 1:4] = 0

    astar = tcod.path.AStar(map_np, 0)
    astar = pickle.loads(pickle.dumps(astar)) # test pickle
    astar = tcod.path.AStar(astar.cost, 0) # use existing cost attribute
    assert len(astar.get_path(0, 0, 5, 5)) == 10

    dijkstra = tcod.path.Dijkstra(map_np, 0)
    dijkstra.set_goal(0, 0)
    assert len(dijkstra.get_path(5, 5)) == 10
    repr(dijkstra) # cover __repr__ methods

    # cover errors
    with pytest.raises(ValueError):
        tcod.path.AStar(np.ones((3, 3, 3), dtype=dtype))
    with pytest.raises(ValueError):
        tcod.path.AStar(np.ones((2, 2), dtype=np.float64))


def path_cost(this_x, this_y, dest_x, dest_y):
        return 1

def test_path_callback():
    astar = tcod.path.AStar(
        tcod.path.EdgeCostCallback(path_cost, (10, 10))
        )
    astar = pickle.loads(pickle.dumps(astar))
    assert astar.get_path(0, 0, 5, 0) == \
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    repr(astar) # cover __repr__ methods


def test_key_repr():
    Key = tcod.Key
    key = Key(vk=1, c=2, shift=True)
    assert key.vk == 1
    assert key.c == 2
    assert key.shift
    key_copy = eval(repr(key))
    assert key.vk == key_copy.vk
    assert key.c == key_copy.c
    assert key.shift == key_copy.shift


def test_mouse_repr():
    Mouse = tcod.Mouse
    mouse = Mouse(x=1, lbutton=True)
    mouse_copy = eval(repr(mouse))
    assert mouse.x == mouse_copy.x
    assert mouse.lbutton == mouse_copy.lbutton
