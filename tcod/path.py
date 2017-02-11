
from __future__ import absolute_import as _

import numpy as np

import tcod.map

from tcod.libtcod import lib, ffi

@ffi.def_extern()
def _pycall_path_old(x1, y1, x2, y2, handle):
    """libtcod style callback, needs to preserve the old userData issue."""
    func, userData = ffi.from_handle(handle)
    return func(x1, y1, x2, y2, userData)


@ffi.def_extern()
def _pycall_path_simple(x1, y1, x2, y2, handle):
    """Does less and should run faster, just calls the handle function."""
    return ffi.from_handle(handle)(x1, y1, x2, y2)


@ffi.def_extern()
def _pycall_path_swap_src_dest(x1, y1, x2, y2, handle):
    """A TDL function dest comes first to match up with a dest only call."""
    return ffi.from_handle(handle)(x2, y2, x1, y1)


@ffi.def_extern()
def _pycall_path_dest_only(x1, y1, x2, y2, handle):
    """A TDL function which samples the dest coordinate only."""
    return ffi.from_handle(handle)(x2, y2)

def _get_pathcost_callback(name):
    """Return a properly cast PathCostArray callback."""
    return ffi.cast('TCOD_path_func_t', ffi.addressof(lib, name))


class _PathFinder(object):
    """
    .. versionadded:: 2.0
    """

    _C_ARRAY_CALLBACKS = {
        np.float32: ('float*', _get_pathcost_callback('PathCostArrayFloat32')),
        np.bool_: ('int8_t*', _get_pathcost_callback('PathCostArrayInt8')),
        np.int8: ('int8_t*', _get_pathcost_callback('PathCostArrayInt8')),
        np.uint8: ('uint8_t*', _get_pathcost_callback('PathCostArrayUInt8')),
        np.int16: ('int16_t*', _get_pathcost_callback('PathCostArrayInt16')),
        np.uint16: ('uint16_t*', _get_pathcost_callback('PathCostArrayUInt16')),
        np.int32: ('int32_t*', _get_pathcost_callback('PathCostArrayInt32')),
        np.uint32: ('uint32_t*', _get_pathcost_callback('PathCostArrayUInt32')),
        }

    def __init__(self, cost, diagonal=1.41,
                 width=None, height=None):
        """

        Args:
            cost (Union[tcod.map.Map,
                        Callable[float, int, int, int, int],
                        numpy.ndarray]):
            diagonal (float): Multiplier for diagonal movement.
                            A value of 0 disables diagonal movement entirely.
            width (int): The clipping width of this pathfinder.
                         Only needed if ``cost`` is a function call.
            height (int): The clipping height of this pathfinder.
                          Only needed if ``cost`` is a function call.
        """
        self.cost = cost
        self.width = width
        self.height = height
        self.diagonal = diagonal
        self.cdata = None
        self.handle = None

        if isinstance(cost, tcod.map.Map):
            self._setup_map(cost)
        elif callable(cost):
            self._setup_callback(lib._pycall_path_simple, cost)
        elif isinstance(cost, tuple) and len(cost) == 2 and callable(cost[0]):
            # hacked in support for old libtcodpy functions, don't abuse this!
            self._setup_callback(lib._pycall_path_old, cost)
        elif isinstance(cost, np.ndarray):
            self._setup_ndarray(cost)
        else:
            raise TypeError('cost must be a Map, function, or numpy array, '
                            'got %r' % (cost,))

    def _setup_map(self, cost):
        """Setup this pathfinder using libtcod's path_new_using_map."""
        self.width = cost.width
        self.height = cost.height
        self.cdata = ffi.gc(
            self._path_new_using_map(cost.cdata, self.diagonal),
            self._path_delete)

    def _setup_callback(self, c_call, args):
        """Setup this pathfinder using libtcod's path_new_using_function.

        The c_call will be called with (x1,y1,x2,y2,args).
        """
        self.handle = ffi.new_handle(args)
        self.cdata = ffi.gc(
                self._path_new_using_function(
                    self.width, self.height, c_call,
                    self.handle, self.diagonal),
                self._path_delete)

    def _setup_ndarray(self, cost):
        """Validate a numpy array and setup a C callback."""
        self.height, self.width = cost.shape # must be a 2d array
        if cost.dtype.type not in self._C_ARRAY_CALLBACKS:
            raise ValueError('dtype must be one of %r, dtype is %r' %
                             (self._C_ARRAY_CALLBACKS.keys(), cost.dtype.type))
        self.cost = cost = np.ascontiguousarray(cost)
        array_type, c_callback = self._C_ARRAY_CALLBACKS[cost.dtype.type]
        cost = ffi.cast(array_type, cost.ctypes.data)
        self.handle = ffi.new('PathCostArray*', (self.width, cost))
        self.cdata = ffi.gc(
                self._path_new_using_function(
                    self.width, self.height,
                    c_callback,
                    self.handle,
                    self.diagonal),
                self._path_delete)

    _path_new_using_map = lib.TCOD_path_new_using_map
    _path_new_using_function = lib.TCOD_path_new_using_function
    _path_delete = lib.TCOD_path_delete


class AStar(_PathFinder):
    """
    .. versionadded:: 2.0
    """

    def get_path(self, start_x, start_y, goal_x, goal_y):
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
        lib.TCOD_path_compute(self.cdata, start_x, start_y, goal_x, goal_y)
        path = []
        x = ffi.new('int[2]')
        y = x + 1
        while lib.TCOD_path_walk(self.cdata, x, y, False):
            path.append((x[0], y[0]))
        return path


class Dijkstra(_PathFinder):
    """
    .. versionadded:: 2.0
    """

    _path_new_using_map = lib.TCOD_dijkstra_new
    _path_new_using_function = lib.TCOD_dijkstra_new_using_function
    _path_delete = lib.TCOD_dijkstra_delete

    def set_goal(self, x, y):
        """Set the goal point and recompute the Dijkstra path-finder.
        """
        lib.TCOD_dijkstra_compute(self.cdata, x, y)

    def get_path(self, x, y):
        """Return a list of (x, y) steps to reach the goal point, if possible.
        """
        lib.TCOD_dijkstra_path_set(self.cdata, x, y)
        path = []
        pointer_x = ffi.new('int[2]')
        pointer_y = pointer_x + 1
        while lib.TCOD_dijkstra_path_walk(self.cdata, pointer_x, pointer_y):
            path.append((pointer_x[0], pointer_y[0]))
        return path
