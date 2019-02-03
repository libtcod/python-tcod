"""

Example::

    >>> import numpy as np
    >>> import tcod.path
    >>> dungeon = np.array(
    ...     [
    ...         [1, 0, 1, 1, 1],
    ...         [1, 0, 1, 0, 1],
    ...         [1, 1, 1, 0, 1],
    ...     ],
    ...     dtype=np.int8,
    ...     )
    ...

    # Create a pathfinder from a numpy array.
    # This is the recommended way to use the tcod.path module.
    >>> astar = tcod.path.AStar(dungeon)
    >>> print(astar.get_path(0, 0, 2, 4))
    [(1, 0), (2, 1), (1, 2), (0, 3), (1, 4), (2, 4)]
    >>> astar.cost[0, 1] = 1 # You can access the map array via this attribute.
    >>> print(astar.get_path(0, 0, 2, 4))
    [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)]

    # Create a pathfinder from an edge_cost function.
    # Calling Python functions from C is known to be very slow.
    >>> def edge_cost(my_x, my_y, dest_x, dest_y):
    ...     return dungeon[dest_x, dest_y]
    ...
    >>> dijkstra = tcod.path.Dijkstra(
    ...     tcod.path.EdgeCostCallback(edge_cost, dungeon.shape),
    ...     )
    ...
    >>> dijkstra.set_goal(0, 0)
    >>> print(dijkstra.get_path(2, 4))
    [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)]

.. versionchanged:: 5.0
    All path-finding functions now respect the NumPy array shape (if a NumPy
    array is used.)
"""
from typing import Any, Callable, List, Tuple, Union  # noqa: F401

import numpy as np

from tcod.libtcod import lib, ffi
import tcod.map  # noqa: F401


@ffi.def_extern()  # type: ignore
def _pycall_path_old(x1: int, y1: int, x2: int, y2: int, handle: Any) -> float:
    """libtcodpy style callback, needs to preserve the old userData issue."""
    func, userData = ffi.from_handle(handle)
    return func(x1, y1, x2, y2, userData)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_simple(
    x1: int, y1: int, x2: int, y2: int, handle: Any
) -> float:
    """Does less and should run faster, just calls the handle function."""
    return ffi.from_handle(handle)(x1, y1, x2, y2)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_swap_src_dest(
    x1: int, y1: int, x2: int, y2: int, handle: Any
) -> float:
    """A TDL function dest comes first to match up with a dest only call."""
    return ffi.from_handle(handle)(x2, y2, x1, y1)  # type: ignore


@ffi.def_extern()  # type: ignore
def _pycall_path_dest_only(
    x1: int, y1: int, x2: int, y2: int, handle: Any
) -> float:
    """A TDL function which samples the dest coordinate only."""
    return ffi.from_handle(handle)(x2, y2)  # type: ignore


def _get_pathcost_func(
    name: str
) -> Callable[[int, int, int, int, Any], float]:
    """Return a properly cast PathCostArray callback."""
    return ffi.cast(  # type: ignore
        "TCOD_path_func_t", ffi.addressof(lib, name)
    )


class _EdgeCostFunc(object):
    """Generic edge-cost function factory.

    `userdata` is the custom userdata to send to the C call.

    `shape` is the maximum boundary for the algorithm.
    """

    _CALLBACK_P = lib._pycall_path_old

    def __init__(self, userdata: Any, shape: Tuple[int, int]) -> None:
        self._userdata = userdata
        self.shape = shape

    def get_tcod_path_ffi(self) -> Tuple[Any, Any, Tuple[int, int]]:
        """Return (C callback, userdata handle, shape)"""
        return self._CALLBACK_P, ffi.new_handle(self._userdata), self.shape

    def __repr__(self) -> str:
        return "%s(%r, shape=%r)" % (
            self.__class__.__name__,
            self._userdata,
            self.shape,
        )


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
        shape: Tuple[int, int],
    ):
        self.callback = callback
        super(EdgeCostCallback, self).__init__(callback, shape)


class NodeCostArray(np.ndarray):  # type: ignore
    """Calculate cost from a numpy array of nodes.

    `array` is a NumPy array holding the path-cost of each node.
    A cost of 0 means the node is blocking.
    """

    _C_ARRAY_CALLBACKS = {
        np.float32: ("float*", _get_pathcost_func("PathCostArrayFloat32")),
        np.bool_: ("int8_t*", _get_pathcost_func("PathCostArrayInt8")),
        np.int8: ("int8_t*", _get_pathcost_func("PathCostArrayInt8")),
        np.uint8: ("uint8_t*", _get_pathcost_func("PathCostArrayUInt8")),
        np.int16: ("int16_t*", _get_pathcost_func("PathCostArrayInt16")),
        np.uint16: ("uint16_t*", _get_pathcost_func("PathCostArrayUInt16")),
        np.int32: ("int32_t*", _get_pathcost_func("PathCostArrayInt32")),
        np.uint32: ("uint32_t*", _get_pathcost_func("PathCostArrayUInt32")),
    }

    def __new__(cls, array: np.array) -> "NodeCostArray":
        """Validate a numpy array and setup a C callback."""
        self = np.asarray(array).view(cls)
        return self  # type: ignore

    def __repr__(self) -> str:
        return "%s(%r)" % (
            self.__class__.__name__,
            repr(self.view(np.ndarray)),
        )

    def get_tcod_path_ffi(self) -> Tuple[Any, Any, Tuple[int, int]]:
        if len(self.shape) != 2:
            raise ValueError(
                "Array must have a 2d shape, shape is %r" % (self.shape,)
            )
        if self.dtype.type not in self._C_ARRAY_CALLBACKS:
            raise ValueError(
                "dtype must be one of %r, dtype is %r"
                % (self._C_ARRAY_CALLBACKS.keys(), self.dtype.type)
            )

        array_type, callback = self._C_ARRAY_CALLBACKS[self.dtype.type]
        userdata = ffi.new(
            "struct PathCostArray*",
            (ffi.cast("char*", self.ctypes.data), self.strides),
        )
        return callback, userdata, self.shape


class _PathFinder(object):
    """A class sharing methods used by AStar and Dijkstra."""

    def __init__(self, cost: Any, diagonal: float = 1.41):
        self.cost = cost
        self.diagonal = diagonal
        self._path_c = None
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
                "Any callback alone is missing shape information. "
                "Wrap your callback in tcod.path.EdgeCostCallback"
            )
            self.cost = NodeCostArray(self.cost)

        self._callback, self._userdata, self.shape = (
            self.cost.get_tcod_path_ffi()
        )
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
        return "%s(cost=%r, diagonal=%r)" % (
            self.__class__.__name__,
            self.cost,
            self.diagonal,
        )

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        del state["_path_c"]
        del state["shape"]
        del state["_callback"]
        del state["_userdata"]
        return state

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)
        self.__init__(self.cost, self.diagonal)  # type: ignore

    _path_new_using_map = lib.TCOD_path_new_using_map
    _path_new_using_function = lib.TCOD_path_new_using_function
    _path_delete = lib.TCOD_path_delete


class AStar(_PathFinder):
    """
    Args:
        cost (Union[tcod.map.Map, numpy.ndarray, Any]):
        diagonal (float): Multiplier for diagonal movement.
            A value of 0 will disable diagonal movement entirely.
    """

    def get_path(
        self, start_x: int, start_y: int, goal_x: int, goal_y: int
    ) -> List[Tuple[int, int]]:
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
    """
    Args:
        cost (Union[tcod.map.Map, numpy.ndarray, Any]):
        diagonal (float): Multiplier for diagonal movement.
            A value of 0 will disable diagonal movement entirely.
    """

    _path_new_using_map = lib.TCOD_dijkstra_new
    _path_new_using_function = lib.TCOD_dijkstra_new_using_function
    _path_delete = lib.TCOD_dijkstra_delete

    def set_goal(self, x: int, y: int) -> None:
        """Set the goal point and recompute the Dijkstra path-finder.
        """
        lib.TCOD_dijkstra_compute(self._path_c, x, y)

    def get_path(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Return a list of (x, y) steps to reach the goal point, if possible.
        """
        lib.TCOD_dijkstra_path_set(self._path_c, x, y)
        path = []
        pointer_x = ffi.new("int[2]")
        pointer_y = pointer_x + 1
        while lib.TCOD_dijkstra_path_walk(self._path_c, pointer_x, pointer_y):
            path.append((pointer_x[0], pointer_y[0]))
        return path
