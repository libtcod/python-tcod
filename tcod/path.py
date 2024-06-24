"""This module provides a fast configurable pathfinding implementation.

To get started create a 2D NumPy array of integers where a value of zero is a
blocked node and any higher value is the cost to move to that node.
You then pass this array to :any:`SimpleGraph`, and then pass that graph to
:any:`Pathfinder`.

Once you have a :any:`Pathfinder` you call :any:`Pathfinder.add_root` to set
the root node.  You can then get a path towards or away from the root with
:any:`Pathfinder.path_from` and :any:`Pathfinder.path_to` respectively.

:any:`SimpleGraph` includes a code example of the above process.

.. versionchanged:: 5.0
    All path-finding functions now respect the NumPy array shape (if a NumPy
    array is used.)
"""

from __future__ import annotations

import functools
import itertools
import warnings
from typing import Any, Callable, Final

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Literal

from tcod._internal import _check
from tcod.cffi import ffi, lib


@ffi.def_extern()  # type: ignore
def _pycall_path_old(x1: int, y1: int, x2: int, y2: int, handle: Any) -> float:  # noqa: ANN401
    """Libtcodpy style callback, needs to preserve the old userData issue."""
    func, userData = ffi.from_handle(handle)
    return func(x1, y1, x2, y2, userData)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_simple(x1: int, y1: int, x2: int, y2: int, handle: Any) -> float:  # noqa: ANN401
    """Does less and should run faster, just calls the handle function."""
    return ffi.from_handle(handle)(x1, y1, x2, y2)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_swap_src_dest(x1: int, y1: int, x2: int, y2: int, handle: Any) -> float:  # noqa: ANN401
    """A TDL function dest comes first to match up with a dest only call."""
    return ffi.from_handle(handle)(x2, y2, x1, y1)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_dest_only(x1: int, y1: int, x2: int, y2: int, handle: Any) -> float:  # noqa: ANN401
    """A TDL function which samples the dest coordinate only."""
    return ffi.from_handle(handle)(x2, y2)  # type: ignore


def _get_path_cost_func(
    name: str,
) -> Callable[[int, int, int, int, Any], float]:
    """Return a properly cast PathCostArray callback."""
    if not ffi:
        return lambda x1, y1, x2, y2, _: 0
    return ffi.cast("TCOD_path_func_t", ffi.addressof(lib, name))  # type: ignore


class _EdgeCostFunc:
    """Generic edge-cost function factory.

    `userdata` is the custom userdata to send to the C call.

    `shape` is the maximum boundary for the algorithm.
    """

    _CALLBACK_P = lib._pycall_path_old

    def __init__(self, userdata: object, shape: tuple[int, int]) -> None:
        self._userdata = userdata
        self.shape = shape

    def get_tcod_path_ffi(self) -> tuple[Any, Any, tuple[int, int]]:
        """Return (C callback, userdata handle, shape)."""
        return self._CALLBACK_P, ffi.new_handle(self._userdata), self.shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._userdata!r}, shape={self.shape!r})"


class EdgeCostCallback(_EdgeCostFunc):
    """Calculate cost from an edge-cost callback.

    `callback` is the custom userdata to send to the C call.

    `shape` is a 2-item tuple representing the maximum boundary for the
    algorithm.  The callback will not be called with parameters outside of
    these bounds.

    .. versionchanged:: 5.0
        Now only accepts a `shape` argument instead of `width` and `height`.
    """

    _CALLBACK_P = lib._pycall_path_simple

    def __init__(
        self,
        callback: Callable[[int, int, int, int], float],
        shape: tuple[int, int],
    ) -> None:
        self.callback = callback
        super().__init__(callback, shape)


class NodeCostArray(np.ndarray):  # type: ignore
    """Calculate cost from a numpy array of nodes.

    `array` is a NumPy array holding the path-cost of each node.
    A cost of 0 means the node is blocking.
    """

    _C_ARRAY_CALLBACKS: Final = {
        np.float32: ("float*", _get_path_cost_func("PathCostArrayFloat32")),
        np.bool_: ("int8_t*", _get_path_cost_func("PathCostArrayInt8")),
        np.int8: ("int8_t*", _get_path_cost_func("PathCostArrayInt8")),
        np.uint8: ("uint8_t*", _get_path_cost_func("PathCostArrayUInt8")),
        np.int16: ("int16_t*", _get_path_cost_func("PathCostArrayInt16")),
        np.uint16: ("uint16_t*", _get_path_cost_func("PathCostArrayUInt16")),
        np.int32: ("int32_t*", _get_path_cost_func("PathCostArrayInt32")),
        np.uint32: ("uint32_t*", _get_path_cost_func("PathCostArrayUInt32")),
    }

    def __new__(cls, array: ArrayLike) -> NodeCostArray:
        """Validate a numpy array and setup a C callback."""
        return np.asarray(array).view(cls)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.view(np.ndarray))!r})"

    def get_tcod_path_ffi(self) -> tuple[Any, Any, tuple[int, int]]:
        if len(self.shape) != 2:  # noqa: PLR2004
            msg = f"Array must have a 2d shape, shape is {self.shape!r}"
            raise ValueError(msg)
        if self.dtype.type not in self._C_ARRAY_CALLBACKS:
            msg = f"dtype must be one of {self._C_ARRAY_CALLBACKS.keys()!r}, dtype is {self.dtype.type!r}"
            raise ValueError(msg)

        array_type, callback = self._C_ARRAY_CALLBACKS[self.dtype.type]
        userdata = ffi.new(
            "struct PathCostArray*",
            (ffi.cast("char*", self.ctypes.data), self.strides),
        )
        return callback, userdata, (self.shape[0], self.shape[1])


class _PathFinder:
    """A class sharing methods used by AStar and Dijkstra."""

    def __init__(self, cost: Any, diagonal: float = 1.41) -> None:
        self.cost = cost
        self.diagonal = diagonal
        self._path_c: Any = None
        self._callback = self._userdata = None

        if hasattr(self.cost, "map_c"):
            self.shape = self.cost.width, self.cost.height
            self._path_c = ffi.gc(
                self._path_new_using_map(self.cost.map_c, diagonal),
                self._path_delete,
            )
            return

        if not hasattr(self.cost, "get_tcod_path_ffi"):
            assert not callable(self.cost), (
                "Any callback alone is missing shape information. " "Wrap your callback in tcod.path.EdgeCostCallback"
            )
            self.cost = NodeCostArray(self.cost)

        (
            self._callback,
            self._userdata,
            self.shape,
        ) = self.cost.get_tcod_path_ffi()
        self._path_c = ffi.gc(
            self._path_new_using_function(
                self.cost.shape[0],
                self.cost.shape[1],
                self._callback,
                self._userdata,
                diagonal,
            ),
            self._path_delete,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cost={self.cost!r}, diagonal={self.diagonal!r})"

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_path_c"]
        del state["shape"]
        del state["_callback"]
        del state["_userdata"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__init__(self.cost, self.diagonal)  # type: ignore

    _path_new_using_map = lib.TCOD_path_new_using_map
    _path_new_using_function = lib.TCOD_path_new_using_function
    _path_delete = lib.TCOD_path_delete


class AStar(_PathFinder):
    """The older libtcod A* pathfinder.

    Args:
        cost (Union[tcod.map.Map, numpy.ndarray, Any]):
        diagonal (float): Multiplier for diagonal movement.
            A value of 0 will disable diagonal movement entirely.
    """

    def get_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int) -> list[tuple[int, int]]:
        """Return a list of (x, y) steps to reach the goal point, if possible.

        Args:
            start_x (int): Starting X position.
            start_y (int): Starting Y position.
            goal_x (int): Destination X position.
            goal_y (int): Destination Y position.

        Returns:
            List[Tuple[int, int]]:
                A list of points, or an empty list if there is no valid path.
        """
        lib.TCOD_path_compute(self._path_c, start_x, start_y, goal_x, goal_y)
        path = []
        x = ffi.new("int[2]")
        y = x + 1
        while lib.TCOD_path_walk(self._path_c, x, y, False):
            path.append((x[0], y[0]))
        return path


class Dijkstra(_PathFinder):
    """The older libtcod Dijkstra pathfinder.

    Args:
        cost (Union[tcod.map.Map, numpy.ndarray, Any]):
        diagonal (float): Multiplier for diagonal movement.
            A value of 0 will disable diagonal movement entirely.
    """

    _path_new_using_map = lib.TCOD_dijkstra_new
    _path_new_using_function = lib.TCOD_dijkstra_new_using_function
    _path_delete = lib.TCOD_dijkstra_delete

    def set_goal(self, x: int, y: int) -> None:
        """Set the goal point and recompute the Dijkstra path-finder."""
        lib.TCOD_dijkstra_compute(self._path_c, x, y)

    def get_path(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return a list of (x, y) steps to reach the goal point, if possible."""
        lib.TCOD_dijkstra_path_set(self._path_c, x, y)
        path = []
        pointer_x = ffi.new("int[2]")
        pointer_y = pointer_x + 1
        while lib.TCOD_dijkstra_path_walk(self._path_c, pointer_x, pointer_y):
            path.append((pointer_x[0], pointer_y[0]))
        return path


_INT_TYPES = {
    np.bool_: lib.np_uint8,
    np.int8: lib.np_int8,
    np.int16: lib.np_int16,
    np.int32: lib.np_int32,
    np.intc: lib.np_int32,
    np.int64: lib.np_int64,
    np.uint8: lib.np_uint8,
    np.uint16: lib.np_uint16,
    np.uint32: lib.np_uint32,
    np.uint64: lib.np_uint64,
}


def maxarray(
    shape: tuple[int, ...],
    dtype: DTypeLike = np.int32,
    order: Literal["C", "F"] = "C",
) -> NDArray[Any]:
    """Return a new array filled with the maximum finite value for `dtype`.

    `shape` is of the new array.  Same as other NumPy array initializers.

    `dtype` should be a single NumPy integer type.

    `order` can be "C" or "F".

    This works the same as
    ``np.full(shape, np.iinfo(dtype).max, dtype, order)``.

    This kind of array is an ideal starting point for distance maps.  Just set
    any point to a lower value such as 0 and then pass this array to a
    function such as :any:`dijkstra2d`.
    """
    return np.full(shape, np.iinfo(dtype).max, dtype, order)


def _export_dict(array: NDArray[Any]) -> dict[str, Any]:
    """Convert a NumPy array into a format compatible with CFFI."""
    if array.dtype.type not in _INT_TYPES:
        msg = f"dtype was {array.dtype.type}, but must be one of {tuple(_INT_TYPES.keys())}."
        raise TypeError(msg)
    return {
        "type": _INT_TYPES[array.dtype.type],
        "ndim": array.ndim,
        "data": ffi.cast("void*", array.ctypes.data),
        "shape": array.shape,
        "strides": array.strides,
    }


def _export(array: NDArray[Any]) -> Any:  # noqa: ANN401
    """Convert a NumPy array into a cffi object."""
    return ffi.new("struct NArray*", _export_dict(array))


def _compile_cost_edges(edge_map: ArrayLike) -> tuple[NDArray[np.intc], int]:
    """Return an edge_cost array using an integer map."""
    edge_map = np.array(edge_map, copy=True)
    if edge_map.ndim != 2:  # noqa: PLR2004
        raise ValueError("edge_map must be 2 dimensional. (Got %i)" % edge_map.ndim)
    edge_center = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    edge_map[edge_center] = 0
    edge_map[edge_map < 0] = 0
    edge_nz = edge_map.nonzero()
    edge_array = np.transpose(edge_nz)
    edge_array -= edge_center
    c_edges = ffi.new("int[]", len(edge_array) * 3)
    edges = np.frombuffer(ffi.buffer(c_edges), dtype=np.intc).reshape(len(edge_array), 3)
    edges[:, :2] = edge_array
    edges[:, 2] = edge_map[edge_nz]
    return c_edges, len(edge_array)


def dijkstra2d(  # noqa: PLR0913
    distance: ArrayLike,
    cost: ArrayLike,
    cardinal: int | None = None,
    diagonal: int | None = None,
    *,
    edge_map: ArrayLike | None = None,
    out: np.ndarray | None = ...,  # type: ignore
) -> NDArray[Any]:
    """Return the computed distance of all nodes on a 2D Dijkstra grid.

    `distance` is an input array of node distances.  Is this often an
    array filled with maximum finite values and 1 or more points with a low
    value such as 0.  Distance will flow from these low values to adjacent
    nodes based the cost to reach those nodes.

    `cost` is an array of node costs.  Any node with a cost less than or equal
    to 0 is considered blocked off.  Positive values are the distance needed to
    reach that node.

    `cardinal` and `diagonal` are the cost multipliers for edges in those
    directions.  A value of None or 0 will disable those directions.  Typical
    values could be: ``1, None``, ``1, 1``, ``2, 3``, etc.

    `edge_map` is a 2D array of edge costs with the origin point centered on
    the array.  This can be used to define the edges used from one node to
    another.  This parameter can be hard to understand so you should see how
    it's used in the examples.

    `out` is the array to fill with the computed Dijkstra distance map.
    Having `out` be the same as `distance` will modify the array in-place,
    which is normally the fastest option.
    If `out` is `None` then the result is returned as a new array.

    Example::

        >>> import numpy as np
        >>> import tcod
        >>> cost = np.ones((3, 3), dtype=np.uint8)
        >>> cost[:2, 1] = 0
        >>> cost
        array([[1, 0, 1],
               [1, 0, 1],
               [1, 1, 1]], dtype=uint8)
        >>> dist = tcod.path.maxarray((3, 3), dtype=np.int32)
        >>> dist[0, 0] = 0
        >>> dist
        array([[         0, 2147483647, 2147483647],
               [2147483647, 2147483647, 2147483647],
               [2147483647, 2147483647, 2147483647]]...)
        >>> tcod.path.dijkstra2d(dist, cost, 2, 3, out=dist)
        array([[         0, 2147483647,         10],
               [         2, 2147483647,          8],
               [         4,          5,          7]]...)
        >>> path = tcod.path.hillclimb2d(dist, (2, 2), True, True)
        >>> path
        array([[2, 2],
               [2, 1],
               [1, 0],
               [0, 0]], dtype=int32)
        >>> path = path[::-1].tolist()
        >>> while path:
        ...     print(path.pop(0))
        [0, 0]
        [1, 0]
        [2, 1]
        [2, 2]

    `edge_map` is used for more complicated graphs.  The following example
    uses a 'knight move' edge map.

    Example::

        >>> import numpy as np
        >>> import tcod
        >>> knight_moves = [
        ...     [0, 1, 0, 1, 0],
        ...     [1, 0, 0, 0, 1],
        ...     [0, 0, 0, 0, 0],
        ...     [1, 0, 0, 0, 1],
        ...     [0, 1, 0, 1, 0],
        ... ]
        >>> dist = tcod.path.maxarray((8, 8))
        >>> dist[0,0] = 0
        >>> cost = np.ones((8, 8), int)
        >>> tcod.path.dijkstra2d(dist, cost, edge_map=knight_moves, out=dist)
        array([[0, 3, 2, 3, 2, 3, 4, 5],
               [3, 4, 1, 2, 3, 4, 3, 4],
               [2, 1, 4, 3, 2, 3, 4, 5],
               [3, 2, 3, 2, 3, 4, 3, 4],
               [2, 3, 2, 3, 4, 3, 4, 5],
               [3, 4, 3, 4, 3, 4, 5, 4],
               [4, 3, 4, 3, 4, 5, 4, 5],
               [5, 4, 5, 4, 5, 4, 5, 6]]...)
        >>> tcod.path.hillclimb2d(dist, (7, 7), edge_map=knight_moves)
        array([[7, 7],
               [5, 6],
               [3, 5],
               [1, 4],
               [0, 2],
               [2, 1],
               [0, 0]], dtype=int32)

    `edge_map` can also be used to define a hex-grid.
    See https://www.redblobgames.com/grids/hexagons/ for more info.
    The following example is using axial coordinates.

    Example::

        hex_edges = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]

    .. versionadded:: 11.2

    .. versionchanged:: 11.13
        Added the `edge_map` parameter.

    .. versionchanged:: 12.1
        Added `out` parameter.  Now returns the output array.
    """
    dist: NDArray[Any] = np.asarray(distance)
    if out is ...:
        out = dist
        warnings.warn(
            "No `out` parameter was given. "
            "Currently this modifies the distance array in-place, but this "
            "will change in the future to return a copy instead. "
            "To ensure the existing behavior is kept you must add an `out` "
            "parameter with the same array as the `distance` parameter.",
            DeprecationWarning,
            stacklevel=2,
        )
    elif out is None:
        out = np.array(distance, copy=True)
    else:
        out[...] = dist

    if dist.shape != out.shape:
        msg = f"distance and output must have the same shape {dist.shape!r} != {out.shape!r}"
        raise TypeError(msg)
    cost = np.asarray(cost)
    if dist.shape != cost.shape:
        msg = f"output and cost must have the same shape {out.shape!r} != {cost.shape!r}"
        raise TypeError(msg)
    c_dist = _export(out)
    if edge_map is not None:
        if cardinal is not None or diagonal is not None:
            msg = "`edge_map` can not be set at the same time as `cardinal` or `diagonal`."
            raise TypeError(msg)
        c_edges, n_edges = _compile_cost_edges(edge_map)
        _check(lib.dijkstra2d(c_dist, _export(cost), n_edges, c_edges))
    else:
        if cardinal is None:
            cardinal = 0
        if diagonal is None:
            diagonal = 0
        _check(lib.dijkstra2d_basic(c_dist, _export(cost), cardinal, diagonal))
    return out


def _compile_bool_edges(edge_map: ArrayLike) -> tuple[Any, int]:
    """Return an edge array using a boolean map."""
    edge_map = np.array(edge_map, copy=True)
    edge_center = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    edge_map[edge_center] = 0
    edge_array = np.transpose(edge_map.nonzero())
    edge_array -= edge_center
    return ffi.new("int[]", list(edge_array.flat)), len(edge_array)


def hillclimb2d(
    distance: ArrayLike,
    start: tuple[int, int],
    cardinal: bool | None = None,
    diagonal: bool | None = None,
    *,
    edge_map: ArrayLike | None = None,
) -> NDArray[Any]:
    """Return a path on a grid from `start` to the lowest point.

    `distance` should be a fully computed distance array.  This kind of array
    is returned by :any:`dijkstra2d`.

    `start` is a 2-item tuple with starting coordinates.  The axes if these
    coordinates should match the axis of the `distance` array.
    An out-of-bounds `start` index will raise an IndexError.

    At each step nodes adjacent toe current will be checked for a value lower
    than the current one.  Which directions are checked is decided by the
    boolean values `cardinal` and `diagonal`.  This process is repeated until
    all adjacent nodes are equal to or larger than the last point on the path.

    If `edge_map` was used with :any:`tcod.path.dijkstra2d` then it should be
    reused for this function.  Keep in mind that `edge_map` must be
    bidirectional since hill-climbing will traverse the map backwards.

    The returned array is a 2D NumPy array with the shape: (length, axis).
    This array always includes both the starting and ending point and will
    always have at least one item.

    Typical uses of the returned array will be to either convert it into a list
    which can be popped from, or transpose it and convert it into a tuple which
    can be used to index other arrays using NumPy's advanced indexing rules.

    .. versionadded:: 11.2

    .. versionchanged:: 11.13
        Added `edge_map` parameter.
    """
    x, y = start
    dist: NDArray[Any] = np.asarray(distance)
    if not (0 <= x < dist.shape[0] and 0 <= y < dist.shape[1]):
        msg = f"Starting point {start!r} not in shape {dist.shape!r}"
        raise IndexError(msg)
    c_dist = _export(dist)
    if edge_map is not None:
        if cardinal is not None or diagonal is not None:
            msg = "`edge_map` can not be set at the same time as `cardinal` or `diagonal`."
            raise TypeError(msg)
        c_edges, n_edges = _compile_bool_edges(edge_map)
        func = functools.partial(lib.hillclimb2d, c_dist, x, y, n_edges, c_edges)
    else:
        func = functools.partial(lib.hillclimb2d_basic, c_dist, x, y, cardinal, diagonal)
    length = _check(func(ffi.NULL))
    path: np.ndarray[Any, np.dtype[np.intc]] = np.ndarray((length, 2), dtype=np.intc)
    c_path = ffi.from_buffer("int*", path)
    _check(func(c_path))
    return path


def _world_array(shape: tuple[int, ...], dtype: DTypeLike = np.int32) -> NDArray[Any]:
    """Return an array where ``ij == arr[ij]``."""
    return np.ascontiguousarray(
        np.transpose(
            np.meshgrid(
                *(np.arange(i, dtype=dtype) for i in shape),
                indexing="ij",
                copy=False,
            ),
            axes=(*range(1, len(shape) + 1), 0),
        )
    )


def _as_hashable(obj: np.ndarray[Any, Any] | None) -> Any | None:
    """Return NumPy arrays as a more hashable form."""
    if obj is None:
        return obj
    return obj.ctypes.data, tuple(obj.shape), tuple(obj.strides)


class CustomGraph:
    """A customizable graph defining how a pathfinder traverses the world.

    If you only need to path over a 2D array with typical edge rules then you
    should use :any:`SimpleGraph`.
    This is an advanced interface for defining custom edge rules which would
    allow things such as 3D movement.

    The graph is created with a `shape` defining the size and number of
    dimensions of the graph.  The `shape` can only be 4 dimensions or lower.

    `order` determines what style of indexing the interface expects.
    This is inherited by the pathfinder and will affect the `ij/xy` indexing
    order of all methods in the graph and pathfinder objects.
    The default order of `"C"` is for `ij` indexing.
    The `order` can be set to `"F"` for `xy` indexing.

    After this graph is created you'll need to add edges which define the
    rules of the pathfinder.  These rules usually define movement in the
    cardinal and diagonal directions, but can also include stairway type edges.
    :any:`set_heuristic` should also be called so that the pathfinder will use
    A*.

    After all edge rules are added the graph can be used to make one or more
    :any:`Pathfinder` instances.

    Example::

        >>> import numpy as np
        >>> import tcod
        >>> graph = tcod.path.CustomGraph((5, 5))
        >>> cost = np.ones((5, 5), dtype=np.int8)
        >>> CARDINAL = [
        ...     [0, 1, 0],
        ...     [1, 0, 1],
        ...     [0, 1, 0],
        ... ]
        >>> graph.add_edges(edge_map=CARDINAL, cost=cost)
        >>> pf = tcod.path.Pathfinder(graph)
        >>> pf.add_root((0, 0))
        >>> pf.resolve()
        >>> pf.distance
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]]...)
        >>> pf.path_to((3, 3))
        array([[0, 0],
               [0, 1],
               [1, 1],
               [2, 1],
               [2, 2],
               [2, 3],
               [3, 3]]...)

    .. versionadded:: 11.13

    .. versionchanged:: 11.15
        Added the `order` parameter.
    """

    def __init__(self, shape: tuple[int, ...], *, order: str = "C") -> None:
        self._shape = self._shape_c = tuple(shape)
        self._ndim = len(self._shape)
        self._order = order
        if self._order == "F":
            self._shape_c = self._shape_c[::-1]
        if not 0 < self._ndim <= 4:  # noqa: PLR2004
            msg = "Graph dimensions must be 1 <= n <= 4."
            raise TypeError(msg)
        self._graph: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._edge_rules_keep_alive: list[Any] = []
        self._edge_rules_p: Any = None
        self._heuristic: tuple[int, int, int, int] | None = None

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self._ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of this graph."""
        return self._shape

    def add_edge(
        self,
        edge_dir: tuple[int, ...],
        edge_cost: int = 1,
        *,
        cost: NDArray[Any],
        condition: ArrayLike | None = None,
    ) -> None:
        """Add a single edge rule.

        `edge_dir` is a tuple with the same length as the graphs dimensions.
        The edge is relative to any node.

        `edge_cost` is the cost multiplier of the edge. Its multiplied with the
        `cost` array to the edges actual cost.

        `cost` is a NumPy array where each node has the cost for movement into
        that node.  Zero or negative values are used to mark blocked areas.

        `condition` is an optional array to mark which nodes have this edge.
        If the node in `condition` is zero then the edge will be skipped.
        This is useful to mark portals or stairs for some edges.

        The expected indexing for `edge_dir`, `cost`, and `condition` depend
        on the graphs `order`.

        Example::

            >>> import numpy as np
            >>> import tcod
            >>> graph3d = tcod.path.CustomGraph((2, 5, 5))
            >>> cost = np.ones((2, 5, 5), dtype=np.int8)
            >>> up_stairs = np.zeros((2, 5, 5), dtype=np.int8)
            >>> down_stairs = np.zeros((2, 5, 5), dtype=np.int8)
            >>> up_stairs[0, 0, 4] = 1
            >>> down_stairs[1, 0, 4] = 1
            >>> CARDINAL = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            >>> graph3d.add_edges(edge_map=CARDINAL, cost=cost)
            >>> graph3d.add_edge((1, 0, 0), 1, cost=cost, condition=up_stairs)
            >>> graph3d.add_edge((-1, 0, 0), 1, cost=cost, condition=down_stairs)
            >>> pf3d = tcod.path.Pathfinder(graph3d)
            >>> pf3d.add_root((0, 1, 1))
            >>> pf3d.path_to((1, 2, 2))
            array([[0, 1, 1],
                   [0, 1, 2],
                   [0, 1, 3],
                   [0, 0, 3],
                   [0, 0, 4],
                   [1, 0, 4],
                   [1, 1, 4],
                   [1, 1, 3],
                   [1, 2, 3],
                   [1, 2, 2]]...)

        Note in the above example that both sets of up/down stairs were added,
        but bidirectional edges are not a requirement for the graph.
        One directional edges such as pits can be added which will
        only allow movement outwards from the root nodes of the pathfinder.
        """
        self._edge_rules_p = None
        edge_dir = tuple(edge_dir)
        cost = np.asarray(cost)
        if len(edge_dir) != self._ndim:
            raise TypeError("edge_dir must have exactly %i items, got %r" % (self._ndim, edge_dir))
        if edge_cost <= 0:
            msg = f"edge_cost must be greater than zero, got {edge_cost!r}"
            raise ValueError(msg)
        if cost.shape != self._shape:
            msg = f"cost array must be shape {self._shape!r}, got {cost.shape!r}"
            raise TypeError(msg)
        if condition is not None:
            condition = np.asarray(condition)
            if condition.shape != self._shape:
                msg = f"condition array must be shape {self._shape!r}, got {condition.shape!r}"
                raise TypeError(msg)
        if self._order == "F":
            # Inputs need to be converted to C.
            edge_dir = edge_dir[::-1]
            cost = cost.T
            if condition is not None:
                condition = condition.T
        key = (_as_hashable(cost), _as_hashable(condition))
        try:
            rule = self._graph[key]
        except KeyError:
            rule = self._graph[key] = {
                "cost": cost,
                "edge_list": [],
            }
            if condition is not None:
                rule["condition"] = condition
        edge = (*edge_dir, edge_cost)
        if edge not in rule["edge_list"]:
            rule["edge_list"].append(edge)

    def add_edges(
        self,
        *,
        edge_map: ArrayLike,
        cost: NDArray[Any],
        condition: ArrayLike | None = None,
    ) -> None:
        """Add a rule with multiple edges.

        `edge_map` is a NumPy array mapping the edges and their costs.
        This is easier to understand by looking at the examples below.
        Edges are relative to center of the array.  The center most value is
        always ignored.  If `edge_map` has fewer dimensions than the graph then
        it will apply to the right-most axes of the graph.

        `cost` is a NumPy array where each node has the cost for movement into
        that node.  Zero or negative values are used to mark blocked areas.

        `condition` is an optional array to mark which nodes have this edge.
        See :any:`add_edge`.
        If `condition` is the same array as `cost` then the pathfinder will
        not move into open area from a non-open ones.

        The expected indexing for `edge_map`, `cost`, and `condition` depend
        on the graphs `order`.  You may need to transpose the examples below
        if you're using `xy` indexing.

        Example::

            # 2D edge maps:
            CARDINAL = [  # Simple arrow-key moves.  Manhattan distance.
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
            CHEBYSHEV = [  # Chess king moves.  Chebyshev distance.
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
            EUCLIDEAN = [  # Approximate euclidean distance.
                [99, 70, 99],
                [70, 0, 70],
                [99, 70, 99],
            ]
            EUCLIDEAN_SIMPLE = [  # Very approximate euclidean distance.
                [3, 2, 3],
                [2, 0, 2],
                [3, 2, 3],
            ]
            KNIGHT_MOVE = [  # Chess knight L-moves.
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
            ]
            AXIAL = [  # https://www.redblobgames.com/grids/hexagons/
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
            # 3D edge maps:
            CARDINAL_PLUS_Z = [  # Cardinal movement with Z up/down edges.
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ]
            CHEBYSHEV_3D = [  # Chebyshev distance, but in 3D.
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
            ]
        """
        edge_map = np.array(edge_map, copy=True)
        if edge_map.ndim < self._ndim:
            edge_map = np.asarray(edge_map[(np.newaxis,) * (self._ndim - edge_map.ndim)])
        if edge_map.ndim != self._ndim:
            raise TypeError("edge_map must must match graph dimensions (%i). (Got %i)" % (self.ndim, edge_map.ndim))
        if self._order == "F":
            # edge_map needs to be converted into C.
            # The other parameters are converted by the add_edge method.
            edge_map = edge_map.T
        edge_center = tuple(i // 2 for i in edge_map.shape)
        edge_map[edge_center] = 0
        edge_map[edge_map < 0] = 0
        edge_nz = edge_map.nonzero()
        edge_costs = edge_map[edge_nz]
        edge_array = np.transpose(edge_nz)
        edge_array -= edge_center
        for edge, edge_cost in zip(edge_array, edge_costs):
            edge = tuple(edge)
            self.add_edge(edge, edge_cost, cost=cost, condition=condition)

    def set_heuristic(self, *, cardinal: int = 0, diagonal: int = 0, z: int = 0, w: int = 0) -> None:
        """Set a pathfinder heuristic so that pathfinding can done with A*.

        `cardinal`, `diagonal`, `z, and `w` are the lower-bound cost of
        movement in those directions.  Values above the lower-bound can be
        used to create a greedy heuristic, which will be faster at the cost of
        accuracy.

        Example::

            >>> import numpy as np
            >>> import tcod
            >>> graph = tcod.path.CustomGraph((5, 5))
            >>> cost = np.ones((5, 5), dtype=np.int8)
            >>> EUCLIDEAN = [[99, 70, 99], [70, 0, 70], [99, 70, 99]]
            >>> graph.add_edges(edge_map=EUCLIDEAN, cost=cost)
            >>> graph.set_heuristic(cardinal=70, diagonal=99)
            >>> pf = tcod.path.Pathfinder(graph)
            >>> pf.add_root((0, 0))
            >>> pf.path_to((4, 4))
            array([[0, 0],
                   [1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]]...)
            >>> pf.distance
            array([[         0,         70,        198, 2147483647, 2147483647],
                   [        70,         99,        169,        297, 2147483647],
                   [       198,        169,        198,        268,        396],
                   [2147483647,        297,        268,        297,        367],
                   [2147483647, 2147483647,        396,        367,        396]]...)
            >>> pf.path_to((2, 0))
            array([[0, 0],
                   [1, 0],
                   [2, 0]]...)
            >>> pf.distance
            array([[         0,         70,        198, 2147483647, 2147483647],
                   [        70,         99,        169,        297, 2147483647],
                   [       140,        169,        198,        268,        396],
                   [       210,        239,        268,        297,        367],
                   [2147483647, 2147483647,        396,        367,        396]]...)

        Without a heuristic the above example would need to evaluate the entire
        array to reach the opposite side of it.
        With a heuristic several nodes can be skipped, which will process
        faster.  Some of the distances in the above example look incorrect,
        that's because those nodes are only partially evaluated, but
        pathfinding to those nodes will work correctly as long as the heuristic
        isn't greedy.
        """
        if 0 == cardinal == diagonal == z == w:
            self._heuristic = None
        if diagonal and cardinal > diagonal:
            msg = "Diagonal parameter can not be lower than cardinal."
            raise ValueError(msg)
        if cardinal < 0 or diagonal < 0 or z < 0 or w < 0:
            msg = "Parameters can not be set to negative values."
            raise ValueError(msg)
        self._heuristic = (cardinal, diagonal, z, w)

    def _compile_rules(self) -> Any:  # noqa: ANN401
        """Compile this graph into a C struct array."""
        if not self._edge_rules_p:
            self._edge_rules_keep_alive = []
            rules = []
            for rule_ in self._graph.values():
                rule = rule_.copy()
                rule["edge_count"] = len(rule["edge_list"])
                # Edge rule format: [i, j, cost, ...] etc.
                edge_obj = ffi.new("int[]", len(rule["edge_list"]) * (self._ndim + 1))
                edge_obj[0 : len(edge_obj)] = itertools.chain(*rule["edge_list"])
                self._edge_rules_keep_alive.append(edge_obj)
                rule["edge_array"] = edge_obj
                self._edge_rules_keep_alive.append(rule["cost"])
                rule["cost"] = _export_dict(rule["cost"])
                if "condition" in rule:
                    self._edge_rules_keep_alive.append(rule["condition"])
                    rule["condition"] = _export_dict(rule["condition"])
                del rule["edge_list"]
                rules.append(rule)
            self._edge_rules_p = ffi.new("struct PathfinderRule[]", rules)
        return self._edge_rules_p, self._edge_rules_keep_alive

    def _resolve(self, pathfinder: Pathfinder) -> None:
        """Run the pathfinding algorithm for this graph."""
        rules, keep_alive = self._compile_rules()
        _check(
            lib.path_compute(
                pathfinder._frontier_p,
                pathfinder._distance_p,
                pathfinder._travel_p,
                len(rules),
                rules,
                pathfinder._heuristic_p,
            )
        )


class SimpleGraph:
    """A simple 2D graph implementation.

    `cost` is a NumPy array where each node has the cost for movement into
    that node.  Zero or negative values are used to mark blocked areas.
    A reference of this array is used.  Any changes to the array will be
    reflected in the graph.

    `cardinal` and `diagonal` are the cost to move along the edges for those
    directions.  The total cost to move from one node to another is the `cost`
    array value multiplied by the edge cost.
    A value of zero will block that direction.

    `greed` is used to define the heuristic.
    To get the fastest accurate heuristic `greed` should be the lowest
    non-zero value on the `cost` array.
    Higher values may be used for an inaccurate but faster heuristic.

    Example::

        >>> import numpy as np
        >>> import tcod
        >>> cost = np.ones((5, 10), dtype=np.int8, order="F")
        >>> graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
        >>> pf = tcod.path.Pathfinder(graph)
        >>> pf.add_root((2, 4))
        >>> pf.path_to((3, 7)).tolist()
        [[2, 4], [2, 5], [2, 6], [3, 7]]

    .. versionadded:: 11.15
    """

    def __init__(self, *, cost: ArrayLike, cardinal: int, diagonal: int, greed: int = 1) -> None:
        cost = np.asarray(cost)
        if cost.ndim != 2:  # noqa: PLR2004
            msg = f"The cost array must e 2 dimensional, array of shape {cost.shape!r} given."
            raise TypeError(msg)
        if greed <= 0:
            msg = f"Greed must be greater than zero, got {greed}"
            raise ValueError(msg)
        edge_map = (
            (diagonal, cardinal, diagonal),
            (cardinal, 0, cardinal),
            (diagonal, cardinal, diagonal),
        )
        self._order = "C" if cost.strides[0] > cost.strides[1] else "F"
        self._subgraph = CustomGraph(cost.shape, order=self._order)
        self._ndim = 2
        self._shape = self._subgraph._shape[0], self._subgraph._shape[1]
        self._shape_c = self._subgraph._shape_c
        self._subgraph.add_edges(edge_map=edge_map, cost=cost)
        self.set_heuristic(cardinal=cardinal * greed, diagonal=diagonal * greed)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def _heuristic(self) -> tuple[int, int, int, int] | None:
        return self._subgraph._heuristic

    def set_heuristic(self, *, cardinal: int, diagonal: int) -> None:
        """Change the heuristic for this graph.

        When created a :any:`SimpleGraph` will automatically have a heuristic.
        So calling this method is often unnecessary.

        `cardinal` and `diagonal` are weights for the heuristic.
        Higher values are more greedy.
        The default values are set to ``cardinal * greed`` and
        ``diagonal * greed`` when the :any:`SimpleGraph` is created.
        """
        self._subgraph.set_heuristic(cardinal=cardinal, diagonal=diagonal)

    def _resolve(self, pathfinder: Pathfinder) -> None:
        self._subgraph._resolve(pathfinder)


class Pathfinder:
    """A generic modular pathfinder.

    How the pathfinder functions depends on the graph provided. see
    :any:`SimpleGraph` for how to set one up.

    .. versionadded:: 11.13
    """

    def __init__(self, graph: CustomGraph | SimpleGraph) -> None:
        self._graph = graph
        self._order = graph._order
        self._frontier_p = ffi.gc(lib.TCOD_frontier_new(self._graph._ndim), lib.TCOD_frontier_delete)
        self._distance = maxarray(self._graph._shape_c)
        self._travel = _world_array(self._graph._shape_c)
        self._distance_p = _export(self._distance)
        self._travel_p = _export(self._travel)
        self._heuristic: tuple[int, int, int, int, tuple[int, ...]] | None = None
        self._heuristic_p: Any = ffi.NULL

    @property
    def distance(self) -> NDArray[Any]:
        """Distance values of the pathfinder.

        The array returned from this property maintains the graphs `order`.

        Unreachable or unresolved points will be at their maximum values.
        You can use :any:`numpy.iinfo` if you need to check for these.

        Example::

            pf  # Resolved Pathfinder instance.
            reachable = pf.distance != numpy.iinfo(pf.distance.dtype).max
            reachable  # A boolean array of reachable area.

        You may edit this array manually, but the pathfinder won't know of
        your changes until :any:`rebuild_frontier` is called.
        """
        return self._distance.T if self._order == "F" else self._distance

    @property
    def traversal(self) -> NDArray[Any]:
        """Array used to generate paths from any point to the nearest root.

        The array returned from this property maintains the graphs `order`.
        It has an extra dimension which includes the index of the next path.

        Example::

            # This example demonstrates the purpose of the traversal array.
            >>> import tcod.path
            >>> graph = tcod.path.SimpleGraph(
            ...     cost=np.ones((5, 5), np.int8), cardinal=2, diagonal=3,
            ... )
            >>> pf = tcod.path.Pathfinder(graph)
            >>> pf.add_root((0, 0))
            >>> pf.resolve()
            >>> pf.traversal[3, 3].tolist()  # Faster.
            [2, 2]
            >>> pf.path_from((3, 3))[1].tolist()  # Slower.
            [2, 2]
            >>> i, j = (3, 3)  # Starting index.
            >>> path = [(i, j)]  # List of nodes from the start to the root.
            >>> while not (pf.traversal[i, j] == (i, j)).all():
            ...     i, j = pf.traversal[i, j].tolist()
            ...     path.append((i, j))
            >>> path  # Slower.
            [(3, 3), (2, 2), (1, 1), (0, 0)]
            >>> pf.path_from((3, 3)).tolist()  # Faster.
            [[3, 3], [2, 2], [1, 1], [0, 0]]


        The above example is slow and will not detect infinite loops.  Use
        :any:`path_from` or :any:`path_to` when you need to get a path.

        As the pathfinder is resolved this array is filled
        """
        if self._order == "F":
            axes = range(self._travel.ndim)
            return self._travel.transpose((*axes[-2::-1], axes[-1]))[..., ::-1]
        return self._travel

    def clear(self) -> None:
        """Reset the pathfinder to its initial state.

        This sets all values on the :any:`distance` array to their maximum
        value.
        """
        self._distance[...] = np.iinfo(self._distance.dtype).max
        self._travel = _world_array(self._graph._shape_c)
        lib.TCOD_frontier_clear(self._frontier_p)

    def add_root(self, index: tuple[int, ...], value: int = 0) -> None:
        """Add a root node and insert it into the pathfinder frontier.

        `index` is the root point to insert.  The length of `index` must match
        the dimensions of the graph.

        `value` is the distance to use for this root.  Zero is typical, but
        if multiple roots are added they can be given different weights.
        """
        index = tuple(index)  # Check for bad input.
        if self._order == "F":  # Convert to ij indexing order.
            index = index[::-1]
        if len(index) != self._distance.ndim:
            raise TypeError("Index must be %i items, got %r" % (self._distance.ndim, index))
        self._distance[index] = value
        self._update_heuristic(None)
        lib.TCOD_frontier_push(self._frontier_p, index, value, value)

    def _update_heuristic(self, goal_ij: tuple[int, ...] | None) -> bool:
        """Update the active heuristic.  Return True if the heuristic changed."""
        if goal_ij is None:
            heuristic = None
        elif self._graph._heuristic is None:
            heuristic = (0, 0, 0, 0, goal_ij)
        else:
            heuristic = (*self._graph._heuristic, goal_ij)
        if self._heuristic == heuristic:
            return False  # Frontier does not need updating.
        self._heuristic = heuristic
        if heuristic is None:
            self._heuristic_p = ffi.NULL
        else:
            self._heuristic_p = ffi.new("struct PathfinderHeuristic*", heuristic)
        lib.update_frontier_heuristic(self._frontier_p, self._heuristic_p)
        return True  # Frontier was updated.

    def rebuild_frontier(self) -> None:
        """Reconstruct the frontier using the current distance array.

        If you are using :any:`add_root` then you will not need to call this
        function.  This is only needed if the :any:`distance` array has been
        modified manually.

        After you are finished editing :any:`distance` you must call this
        function before calling :any:`resolve` or any function which calls
        :any:`resolve` implicitly such as :any:`path_from` or :any:`path_to`.
        """
        lib.TCOD_frontier_clear(self._frontier_p)
        self._update_heuristic(None)
        _check(lib.rebuild_frontier_from_distance(self._frontier_p, self._distance_p))

    def resolve(self, goal: tuple[int, ...] | None = None) -> None:
        """Manually run the pathfinder algorithm.

        The :any:`path_from` and :any:`path_to` methods will automatically
        call this method on demand.

        If `goal` is `None` then this will attempt to complete the entire
        :any:`distance` and :any:`traversal` arrays without a heuristic.
        This is similar to Dijkstra.

        If `goal` is given an index then it will attempt to resolve the
        :any:`distance` and :any:`traversal` arrays only up to the `goal`.
        If the graph has set a heuristic then it will be used with a process
        similar to `A*`.

        Example::

            >>> import tcod.path
            >>> graph = tcod.path.SimpleGraph(
            ...     cost=np.ones((4, 4), np.int8), cardinal=2, diagonal=3,
            ... )
            >>> pf = tcod.path.Pathfinder(graph)
            >>> pf.distance
            array([[2147483647, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647]]...)
            >>> pf.add_root((0, 0))
            >>> pf.distance
            array([[         0, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647]]...)
            >>> pf.resolve((1, 1))  # Resolve up to (1, 1) as A*.
            >>> pf.distance  # Partially resolved distance.
            array([[         0,          2,          6, 2147483647],
                   [         2,          3,          5, 2147483647],
                   [         6,          5,          6, 2147483647],
                   [2147483647, 2147483647, 2147483647, 2147483647]]...)
            >>> pf.resolve()  # Resolve the full graph as Dijkstra.
            >>> pf.distance  # Fully resolved distance.
            array([[0, 2, 4, 6],
                   [2, 3, 5, 7],
                   [4, 5, 6, 8],
                   [6, 7, 8, 9]]...)
        """
        if goal is not None:
            goal = tuple(goal)  # Check for bad input.
            if len(goal) != self._distance.ndim:
                raise TypeError("Goal must be %i items, got %r" % (self._distance.ndim, goal))
            if self._order == "F":
                # Goal is now ij indexed for the rest of this function.
                goal = goal[::-1]
            if self._distance[goal] != np.iinfo(self._distance.dtype).max:
                if not lib.frontier_has_index(self._frontier_p, goal):
                    return
        self._update_heuristic(goal)
        self._graph._resolve(self)

    def path_from(self, index: tuple[int, ...]) -> NDArray[Any]:
        """Return the shortest path from `index` to the nearest root.

        The returned array is of shape `(length, ndim)` where `length` is the
        total inclusive length of the path and `ndim` is the dimensions of the
        pathfinder defined by the graph.

        The return value is inclusive, including both the starting and ending
        points on the path.  If the root point is unreachable or `index` is
        already at a root then `index` will be the only point returned.

        This automatically calls :any:`resolve` if the pathfinder has not
        yet reached `index`.

        A common usage is to slice off the starting point and convert the array
        into a list.

        Example::

            >>> import tcod.path
            >>> cost = np.ones((5, 5), dtype=np.int8)
            >>> cost[:, 3:] = 0
            >>> graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
            >>> pf = tcod.path.Pathfinder(graph)
            >>> pf.add_root((0, 0))
            >>> pf.path_from((2, 2)).tolist()
            [[2, 2], [1, 1], [0, 0]]
            >>> pf.path_from((2, 2))[1:].tolist()  # Exclude the starting point by slicing the array.
            [[1, 1], [0, 0]]
            >>> pf.path_from((4, 4)).tolist()  # Blocked paths will only have the index point.
            [[4, 4]]
            >>> pf.path_from((4, 4))[1:].tolist()  # Exclude the starting point so that a blocked path is an empty list.
            []

        """
        index = tuple(index)  # Check for bad input.
        if len(index) != self._graph._ndim:
            raise TypeError("Index must be %i items, got %r" % (self._distance.ndim, index))
        self.resolve(index)
        if self._order == "F":  # Convert to ij indexing order.
            index = index[::-1]
        length = _check(lib.get_travel_path(self._graph._ndim, self._travel_p, index, ffi.NULL))
        path: np.ndarray[Any, np.dtype[np.intc]] = np.ndarray((length, self._graph._ndim), dtype=np.intc)
        _check(
            lib.get_travel_path(
                self._graph._ndim,
                self._travel_p,
                index,
                ffi.from_buffer("int*", path),
            )
        )
        return path[:, ::-1] if self._order == "F" else path

    def path_to(self, index: tuple[int, ...]) -> NDArray[Any]:
        """Return the shortest path from the nearest root to `index`.

        See :any:`path_from`.
        This is an alias for ``path_from(...)[::-1]``.

        This is the method to call when the root is an entity to move to a
        position rather than a destination itself.

        Example::

            >>> import tcod.path
            >>> graph = tcod.path.SimpleGraph(
            ...     cost=np.ones((5, 5), np.int8), cardinal=2, diagonal=3,
            ... )
            >>> pf = tcod.path.Pathfinder(graph)
            >>> pf.add_root((0, 0))
            >>> pf.path_to((0, 0)).tolist()  # This method always returns at least one point.
            [[0, 0]]
            >>> pf.path_to((3, 3)).tolist()  # Always includes both ends on a valid path.
            [[0, 0], [1, 1], [2, 2], [3, 3]]
            >>> pf.path_to((3, 3))[1:].tolist()  # Exclude the starting point by slicing the array.
            [[1, 1], [2, 2], [3, 3]]
            >>> pf.path_to((0, 0))[1:].tolist()  # Exclude the starting point so that a blocked path is an empty list.
            []
        """
        return self.path_from(index)[::-1]
