"""Tests for newer tcod API."""
import copy
import pickle
from typing import Any, NoReturn

import numpy as np
import pytest
from numpy.typing import DTypeLike, NDArray

import tcod

# ruff: noqa: D103


def raise_Exception(*args: object) -> NoReturn:
    raise RuntimeError("testing exception")  # noqa: TRY003, EM101


def test_line_error() -> None:
    """Test exception propagation."""
    with pytest.raises(RuntimeError):
        tcod.line(0, 0, 10, 10, py_callback=raise_Exception)


@pytest.mark.filterwarnings("ignore:Iterate over nodes using")
@pytest.mark.filterwarnings("ignore:Use pre_order method instead of walk.")
def test_tcod_bsp() -> None:
    """Test tcod additions to BSP."""
    bsp = tcod.bsp.BSP(0, 0, 32, 32)

    assert bsp.level == 0
    assert not bsp.horizontal
    assert not bsp.children

    with pytest.raises(RuntimeError):
        tcod.bsp_traverse_pre_order(bsp, raise_Exception)

    bsp.split_recursive(3, 4, 4, 1, 1)
    for node in bsp.walk():
        assert isinstance(node, tcod.bsp.BSP)

    assert bsp.children

    # test that operations on deep BSP nodes preserve depth
    sub_bsp = bsp.children[0]
    sub_bsp.split_recursive(3, 2, 2, 1, 1)
    assert sub_bsp.children[0].level == 2  # noqa: PLR2004

    # cover find_node method
    assert bsp.find_node(0, 0)
    assert bsp.find_node(-1, -1) is None

    # cover __str__
    str(bsp)


@pytest.mark.filterwarnings("ignore:Use map.+ to check for this")
@pytest.mark.filterwarnings("ignore:This class may perform poorly")
def test_tcod_map_set_bits() -> None:
    map_ = tcod.map.Map(2, 2)

    assert not map_.transparent[:].any()
    assert not map_.walkable[:].any()
    assert not map_.fov[:].any()

    map_.transparent[1, 0] = True
    assert tcod.map_is_transparent(map_, 0, 1)
    map_.walkable[1, 0] = True
    assert tcod.map_is_walkable(map_, 0, 1)
    map_.fov[1, 0] = True
    assert tcod.map_is_in_fov(map_, 0, 1)


@pytest.mark.filterwarnings("ignore:This class may perform poorly")
def test_tcod_map_get_bits() -> None:
    map_ = tcod.map.Map(2, 2)
    map_.transparent[0]


@pytest.mark.filterwarnings("ignore:This class may perform poorly")
def test_tcod_map_copy() -> None:
    map_ = tcod.map.Map(3, 3)
    map_.transparent[:] = True
    assert map_.transparent.tolist() == copy.copy(map_).transparent.tolist()


@pytest.mark.filterwarnings("ignore:This class may perform poorly")
def test_tcod_map_pickle() -> None:
    map_ = tcod.map.Map(3, 3)
    map_.transparent[:] = True
    map2 = pickle.loads(pickle.dumps(copy.copy(map_)))
    assert map_.transparent[:].tolist() == map2.transparent[:].tolist()


@pytest.mark.filterwarnings("ignore:This class may perform poorly")
def test_tcod_map_pickle_fortran() -> None:
    map_ = tcod.map.Map(2, 3, order="F")
    map2: tcod.map.Map = pickle.loads(pickle.dumps(copy.copy(map_)))
    assert map_._Map__buffer.strides == map2._Map__buffer.strides  # type: ignore
    assert map_.transparent.strides == map2.transparent.strides
    assert map_.walkable.strides == map2.walkable.strides
    assert map_.fov.strides == map2.fov.strides


@pytest.mark.filterwarnings("ignore")
def test_color_class() -> None:
    assert tcod.black == tcod.black
    assert tcod.black == (0, 0, 0)
    assert tcod.black == [0, 0, 0]
    assert tcod.black != tcod.white
    assert tcod.white * 1 == tcod.white
    assert tcod.white * tcod.black == tcod.black
    assert tcod.white - tcod.white == tcod.black
    assert tcod.black + (2, 2, 2) - (1, 1, 1) == (1, 1, 1)  # noqa: RUF005

    color = tcod.Color()
    color.r = 1
    color.g = 2
    color.b = 3
    assert color == (1, 2, 3)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32, np.float32])
def test_path_numpy(dtype: DTypeLike) -> None:
    map_np: NDArray[Any] = np.ones((6, 6), dtype=dtype)
    map_np[1:4, 1:4] = 0

    astar = tcod.path.AStar(map_np, 0)
    astar = pickle.loads(pickle.dumps(astar))  # test pickle
    astar = tcod.path.AStar(astar.cost, 0)  # use existing cost attribute
    assert len(astar.get_path(0, 0, 5, 5)) == 10  # noqa: PLR2004

    dijkstra = tcod.path.Dijkstra(map_np, 0)
    dijkstra.set_goal(0, 0)
    assert len(dijkstra.get_path(5, 5)) == 10  # noqa: PLR2004
    repr(dijkstra)  # cover __repr__ methods

    # cover errors
    with pytest.raises(ValueError, match=r"Array must have a 2d shape, shape is \(3, 3, 3\)"):
        tcod.path.AStar(np.ones((3, 3, 3), dtype=dtype))
    with pytest.raises(ValueError, match=r"dtype must be one of dict_keys"):
        tcod.path.AStar(np.ones((2, 2), dtype=np.float64))


def path_cost(this_x: int, this_y: int, dest_x: int, dest_y: int) -> bool:
    return True


def test_path_callback() -> None:
    astar = tcod.path.AStar(tcod.path.EdgeCostCallback(path_cost, (10, 10)))
    astar = pickle.loads(pickle.dumps(astar))
    assert astar.get_path(0, 0, 5, 0) == [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    repr(astar)  # cover __repr__ methods


def test_key_repr() -> None:
    Key = tcod.Key
    key = Key(vk=1, c=2, shift=True)
    assert key.vk == 1
    assert key.c == 2  # noqa: PLR2004
    assert key.shift
    key_copy = eval(repr(key))
    assert key.vk == key_copy.vk
    assert key.c == key_copy.c
    assert key.shift == key_copy.shift


def test_mouse_repr() -> None:
    Mouse = tcod.Mouse
    mouse = Mouse(x=1, lbutton=True)
    mouse_copy = eval(repr(mouse))
    assert mouse.x == mouse_copy.x
    assert mouse.lbutton == mouse_copy.lbutton


def test_cffi_structs() -> None:
    # Make sure cffi structures are the correct size.
    tcod.ffi.new("SDL_Event*")
    tcod.ffi.new("SDL_AudioCVT*")


@pytest.mark.filterwarnings("ignore")
def test_recommended_size(console: tcod.console.Console) -> None:
    tcod.console.recommended_size()


@pytest.mark.filterwarnings("ignore")
def test_context() -> None:
    with tcod.context.new_window(32, 32, renderer=tcod.RENDERER_SDL2):
        pass
    WIDTH, HEIGHT = 16, 4
    with tcod.context.new_terminal(columns=WIDTH, rows=HEIGHT, renderer=tcod.RENDERER_SDL2) as context:
        console = tcod.Console(*context.recommended_console_size())
        context.present(console)
        assert context.sdl_window_p is not None
        assert context.renderer_type >= 0
        context.change_tileset(tcod.tileset.Tileset(16, 16))
        context.pixel_to_tile(0, 0)
        context.pixel_to_subtile(0, 0)
