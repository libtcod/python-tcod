
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


class _PathFinder(object):
    """
    .. versionadded:: 2.0
    """

    def __init__(self):
        self.width = None
        self.height = None
        self.diagonal_cost = None
        self.cdata = None
        self.handle = None
        self.map_obj = None

    @classmethod
    def new_with_map(cls, tcod_map, diagonal_cost=1.41):
        self = cls()
        self.width = tcod_map.width
        self.height = tcod_map.height
        self.diagonal_cost = diagonal_cost

        self.map_obj = tcod_map
        self.cdata = ffi.gc(
            self._path_new_using_map(tcod_map.cdata, diagonal_cost),
            self._path_delete)
        return self

    @classmethod
    def new_with_callback(cls, width, height, callback, diagonal_cost=1.41):
        self = cls()
        self.width = width
        self.height = height
        self.diagonal_cost = diagonal_cost

        self.handle = ffi.new_handle(callback)
        self.cdata = ffi.gc(
            self._path_new_using_function(
                width, height, lib._pycall_path_simple,
                self.handle, diagonal_cost),
            self._path_delete)
        return self

    @classmethod
    def _new_with_callback_old(cls, width, height, callback, diagonal_cost,
                               userData):
        self = cls()
        self.width = width
        self.height = height
        self.diagonal_cost = diagonal_cost

        self.handle = ffi.new_handle((callback, userData))
        self.cdata = ffi.gc(
            self._path_new_using_function(
                width, height, lib._pycall_path_old,
                self.handle, diagonal_cost),
            self._path_delete)
        return self


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
            path.append(pointer_x[0], pointer_y[0])
        return path

__all__ = [
           AStar,
           Dijkstra,
           ]