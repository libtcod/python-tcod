
from __future__ import absolute_import as _

import numpy as np

import tcod.map

from tcod.libtcod import lib, ffi

@ffi.def_extern()
def _pycall_path_old(x1, y1, x2, y2, handle):
    """libtcodpy style callback, needs to preserve the old userData issue."""
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

def _get_pathcost_func(name):
    """Return a properly cast PathCostArray callback."""
    return ffi.cast('TCOD_path_func_t', ffi.addressof(lib, name))


class _EdgeCostFunc(object):
    _CALLBACK_P = lib._pycall_path_old

    def __init__(self, userdata, width, height):
        """
        Args:
            userdata (Any): Custom userdata to send to the C call.
            width (int): The maximum width of this callback.
            height (int): The maximum height of this callback.
        """
        self._userdata = userdata
        self.width = width
        self.height = height
        self._userdata_p = None

    def get_cffi_callback(self):
        """

        Returns:
            Tuple[CData, CData]: A cffi (callback, userdata) pair.


        """
        if self._userdata_p is None:
            self._userdata_p = ffi.new_handle(self._userdata)
        return self._CALLBACK_P, self._userdata_p

    def __repr__(self):
        return '%s(%r, width=%r, height=%r)' % (
            self.__class__.__name__,
            self._userdata, self.width, self.height,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_userdata_p'] = None
        return state


class EdgeCostCallback(_EdgeCostFunc):
    _CALLBACK_P = lib._pycall_path_simple

    def __init__(self, callback, width, height):
        """
        Args:
            callback (Callable[[int, int, int, int], float]):
                A callback which can handle the following parameters:
                `(source_x:int, source_y:int, dest_x:int, dest_y:int) -> float`
            width (int): The maximum width of this callback.
            height (int): The maximum height of this callback.
        """
        self.callback = callback
        super(EdgeCostCallback, self).__init__(callback, width, height)

class NodeCostArray(np.ndarray):

    _C_ARRAY_CALLBACKS = {
        np.float32: ('float*', _get_pathcost_func('PathCostArrayFloat32')),
        np.bool_: ('int8_t*', _get_pathcost_func('PathCostArrayInt8')),
        np.int8: ('int8_t*', _get_pathcost_func('PathCostArrayInt8')),
        np.uint8: ('uint8_t*', _get_pathcost_func('PathCostArrayUInt8')),
        np.int16: ('int16_t*', _get_pathcost_func('PathCostArrayInt16')),
        np.uint16: ('uint16_t*', _get_pathcost_func('PathCostArrayUInt16')),
        np.int32: ('int32_t*', _get_pathcost_func('PathCostArrayInt32')),
        np.uint32: ('uint32_t*', _get_pathcost_func('PathCostArrayUInt32')),
        }

    def __new__(cls, array):
        """Validate a numpy array and setup a C callback."""
        self = np.ascontiguousarray(array).view(cls)
        return self

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           repr(self.view(np.ndarray)))

    def get_cffi_callback(self):
        if hasattr(self, '_callback_p'):
            return self._callback_p, self._userdata_p

        if len(self.shape) != 2:
            raise ValueError('Array must have a 2d shape, shape is %r' %
                             (self.shape,))
        if self.dtype.type not in self._C_ARRAY_CALLBACKS:
            raise ValueError('dtype must be one of %r, dtype is %r' %
                             (self._C_ARRAY_CALLBACKS.keys(), self.dtype.type))

        array_type, self._callback_p = \
            self._C_ARRAY_CALLBACKS[self.dtype.type]
        self._userdata_p = ffi.new(
            'PathCostArray*',
            (self.width, ffi.cast(array_type, self.ctypes.data)),
        )
        return self._callback_p, self._userdata_p


class _PathFinder(object):
    """
    .. versionadded:: 2.0
    """

    def __init__(self, cost, diagonal=1.41):
        """
        .. versionchanged:: 3.0
            Removed width and height parameters.

            Callbacks must be wrapped in an :any:`tcod.path.EdgeCostCallback`
            instance or similar.

        Args:
            cost (Union[tcod.map.Map, numpy.ndarray, Any]):
            diagonal (float): Multiplier for diagonal movement.
                A value of 0 will disable diagonal movement entirely.
        """
        self.cost = cost
        self.diagonal = diagonal
        self._path_c = None

        if hasattr(self.cost, 'map_c'):
            self._path_c = ffi.gc(
                self._path_new_using_map(self.cost.map_c, diagonal),
                self._path_delete,
                )
            return

        if not hasattr(self.cost, 'get_cffi_callback'):
            assert not callable(self.cost), \
                "A callback alone is missing width&height information."
            self.cost = NodeCostArray(self.cost)

        callback, userdata = self.cost.get_cffi_callback()
        self._path_c = ffi.gc(
            self._path_new_using_function(
                self.cost.width,
                self.cost.height,
                callback,
                userdata,
                diagonal
                ),
            self._path_delete,
            )

    def __repr__(self):
        return '%s(cost=%r, diagonal=%r)' % (self.__class__.__name__,
                                             self.cost, self.diagonal)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_path_c']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(self.cost, self.diagonal)

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
        lib.TCOD_path_compute(self._path_c, start_x, start_y, goal_x, goal_y)
        path = []
        x = ffi.new('int[2]')
        y = x + 1
        while lib.TCOD_path_walk(self._path_c, x, y, False):
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
        lib.TCOD_dijkstra_compute(self._path_c, x, y)

    def get_path(self, x, y):
        """Return a list of (x, y) steps to reach the goal point, if possible.
        """
        lib.TCOD_dijkstra_path_set(self._path_c, x, y)
        path = []
        pointer_x = ffi.new('int[2]')
        pointer_y = pointer_x + 1
        while lib.TCOD_dijkstra_path_walk(self._path_c, pointer_x, pointer_y):
            path.append((pointer_x[0], pointer_y[0]))
        return path
