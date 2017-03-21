"""
    Rogue-like map utilitys such as line-of-sight, field-of-view, and path-finding.

"""

import itertools as _itertools
import math as _math

from tcod import ffi as _ffi
from tcod import lib as _lib

import tdl as _tdl
from . import style as _style

_FOVTYPES = {'BASIC' : 0, 'DIAMOND': 1, 'SHADOW': 2, 'RESTRICTIVE': 12, 'PERMISSIVE': 11}

def _get_fov_type(fov):
    "Return a FOV from a string"
    oldFOV = fov
    fov = str(fov).upper()
    if fov in _FOVTYPES:
        return _FOVTYPES[fov]
    if fov[:10] == 'PERMISSIVE' and fov[10].isdigit() and fov[10] != '9':
        return 4 + int(fov[10])
    raise _tdl.TDLError('No such fov option as %s' % oldFOV)

class Map(object):
    """Fast field-of-view and path-finding on stored data.

    Set map conditions with the walkable and transparency attributes, this
    object can be iterated and checked for containment similar to consoles.

    For example, you can set all tiles and transparent and walkable with the
    following code::

        map = tdl.map.Map(80, 60)
        for x,y in map:
            map.transparent[x,y] = true
            map.walkable[x,y] = true

    @ivar transparent: Map transparency, access this attribute with
                       map.transparent[x,y]

                       Set to True to allow field-of-view rays, False will
                       block field-of-view.

                       Transparent tiles only affect field-of-view.

    @ivar walkable: Map accessibility, access this attribute with
                    map.walkable[x,y]

                    Set to True to allow path-finding through that tile,
                    False will block passage to that tile.

                    Walkable tiles only affect path-finding.

    @ivar fov: Map tiles touched by a field-of-view computation,
               access this attribute with map.fov[x,y]

               Is True if a the tile is if view, otherwise False.

               You can set to this attribute if you want, but you'll typically
               be using it to read the field-of-view of a L{compute_fov} call.

    @since: 1.5.0
    """

    class _MapAttribute(object):
        def __init__(self, map, bit_index):
            self.map = map
            self.bit_index = bit_index
            self.bit = 1 << bit_index
            self.bit_inverse = 0xFF ^ self.bit

        def __getitem__(self, key):
            return bool(self.map._array_cdata[key[1]][key[0]] & self.bit)

        def __setitem__(self, key, value):
            self.map._array_cdata[key[1]][key[0]] = (
                (self.map._array_cdata[key[1]][key[0]] & self.bit_inverse) |
                (self.bit * bool(value))
                )

    def __init__(self, width, height):
        """Create a new Map with width and height.

        @type width: int
        @type height: int
        @param width: Width of the new Map instance, in tiles.
        @param width: Height of the new Map instance, in tiles.
        """
        self.width = width
        self.height = height
        self._map_cdata = _lib.TCOD_map_new(width, height)
        # cast array into cdata format: uint8[y][x]
        # for quick Python access
        self._array_cdata = _ffi.new('uint8_t[%i][%i]' % (height, width))
        # flat array to pass to TDL's C helpers
        self._array_cdata_flat = _ffi.cast('uint8_t *', self._array_cdata)
        self.transparent = self._MapAttribute(self, 0)
        self.walkable = self._MapAttribute(self, 1)
        self.fov = self._MapAttribute(self, 2)

    def compute_fov(self, x, y, fov='PERMISSIVE', radius=None, light_walls=True,
                    sphere=True, cumulative=False):
        """Compute the field-of-view of this Map and return an iterator of the
        points touched.

        @type x: int
        @type y: int

        @param x: x center of the field-of-view
        @param y: y center of the field-of-view
        @type fov: string
        @param fov: The type of field-of-view to be used.  Available types are:

                    'BASIC', 'DIAMOND', 'SHADOW', 'RESTRICTIVE', 'PERMISSIVE',
                    'PERMISSIVE0', 'PERMISSIVE1', ..., 'PERMISSIVE8'
        @type radius: int
        @param radius: Raduis of the field-of-view.
        @type light_walls: boolean
        @param light_walls: Include or exclude wall tiles in the field-of-view.
        @type sphere: boolean
        @param sphere: True for a spherical field-of-view.
                       False for a square one.
        @type cumulative: boolean
        @param cumulative:

        @rtype: iter((x, y), ...)
        @return: An iterator of (x, y) points of tiles touched by the
                 field-of-view.

                 Unexpected behaviour can happen if you modify the Map while
                 using the iterator.

                 You can use the Map's fov attribute as an alternative to this
                 iterator.
        """
        # refresh cdata
        _lib.TDL_map_data_from_buffer(self._map_cdata,
                                      self._array_cdata_flat)
        if radius is None: # infinite radius
            radius = max(self.width, self.height)
        _lib.TCOD_map_compute_fov(self._map_cdata, x, y, radius, light_walls,
                                  _get_fov_type(fov))
        _lib.TDL_map_fov_to_buffer(self._map_cdata,
                                   self._array_cdata_flat, cumulative)
        def iterate_fov():
            _array_cdata = self._array_cdata
            for y in range(self.height):
                for x in range(self.width):
                    if(_array_cdata[y][x] & 4):
                        yield (x, y)
        return iterate_fov()



    def compute_path(self, start_x, start_y, dest_x, dest_y,
                     diagonal_cost=_math.sqrt(2)):
        """Get the shortest path between two points.

        The start position is not included in the list.

        @type diagnalCost: float
        @param diagnalCost: Multiplier for diagonal movement.

                            Can be set to zero to disable diagonal movement
                            entirely.
        @rtype: [(x, y), ...]
        @return: Returns a the shortest list of points to get to the destination
                 position from the starting position
        """
        # refresh cdata
        _lib.TDL_map_data_from_buffer(self._map_cdata,
                                      self._array_cdata_flat)
        path_cdata = _lib.TCOD_path_new_using_map(self._map_cdata, diagonal_cost)
        try:
            _lib.TCOD_path_compute(path_cdata, start_x, start_y, dest_x, dest_y)
            x = _ffi.new('int *')
            y = _ffi.new('int *')
            length = _lib.TCOD_path_size(path_cdata)
            path = [None] * length
            for i in range(length):
                _lib.TCOD_path_get(path_cdata, i, x, y)
                path[i] = ((x[0], y[0]))
        finally:
            _lib.TCOD_path_delete(path_cdata)
        return path

    def __iter__(self):
        return _itertools.product(range(self.width), range(self.height))

    def __contains__(self, position):
        x, y = position
        return (0 <= x < self.width) and (0 <= y < self.height)



class AStar(object):
    """A* pathfinder

    Using this class requires a callback detailed in L{AStar.__init__}

    @undocumented: getPath
    """

    __slots__ = ('_as_parameter_', '_callback', '__weakref__')



    def __init__(self, width, height, callback,
                 diagnalCost=_math.sqrt(2), advanced=False):
        """Create an A* pathfinder using a callback.

        Before crating this instance you should make one of two types of
        callbacks:
         - A function that returns the cost to move to (x, y)
        or
         - A function that returns the cost to move between
           (destX, destY, sourceX, sourceY)
        If path is blocked the function should return zero or None.
        When using the second type of callback be sure to set advanced=True

        @type width: int
        @param width: width of the pathfinding area in tiles
        @type height: int
        @param height: height of the pathfinding area in tiles

        @type callback: function
        @param callback: A callback taking parameters depending on the setting
                         of 'advanced' and returning the cost of
                         movement for an open tile or zero for a
                         blocked tile.

        @type diagnalCost: float
        @param diagnalCost: Multiplier for diagonal movement.

                            Can be set to zero to disable diagonal movement
                            entirely.

        @type advanced: boolean
        @param advanced: A simple callback with 2 positional parameters may not
                         provide enough information.  Setting this to True will
                         call the callback with 2 additional parameters giving
                         you both the destination and the source of movement.

                         When True the callback will need to accept
                         (destX, destY, sourceX, sourceY) as parameters.
                         Instead of just (destX, destY).

        """
        if not diagnalCost: # set None or False to zero
            diagnalCost = 0.0
        if advanced:
            def newCallback(sourceX, sourceY, destX, destY, null):
                pathCost = callback(destX, destY, sourceX, sourceY)
                if pathCost:
                    return pathCost
                return 0.0
        else:
            def newCallback(sourceX, sourceY, destX, destY, null):
                pathCost = callback(destX, destY) # expecting a float or 0
                if pathCost:
                    return pathCost
                return 0.0
        # float(int, int, int, int, void*)
        self._callback = _ffi.callback('TCOD_path_func_t')(newCallback)

        self._as_parameter_ = _lib.TCOD_path_new_using_function(width, height,
                                     self._callback, _ffi.NULL, diagnalCost)

    def __del__(self):
        if self._as_parameter_:
            _lib.TCOD_path_delete(self._as_parameter_)
            self._as_parameter_ = None

    def get_path(self, origX, origY, destX, destY):
        """
        Get the shortest path from origXY to destXY.

        @rtype: [(x, y), ...]
        @return: Returns a list walking the path from origXY to destXY.
                 This excludes the starting point and includes the destination.

                 If no path is found then an empty list is returned.
        """
        found = _lib.TCOD_path_compute(self._as_parameter_, origX, origY, destX, destY)
        if not found:
            return [] # path not found
        x, y = _ffi.new('int *'), _ffi.new('int *')
        recalculate = True
        path = []
        while _lib.TCOD_path_walk(self._as_parameter_, x, y, recalculate):
            path.append((x[0], y[0]))
        return path

def quick_fov(x, y, callback, fov='PERMISSIVE', radius=7.5, lightWalls=True, sphere=True):
    """All field-of-view functionality in one call.

    Before using this call be sure to make a function, lambda, or method that takes 2
    positional parameters and returns True if light can pass through the tile or False
    for light-blocking tiles and for indexes that are out of bounds of the
    dungeon.

    This function is 'quick' as in no hassle but can quickly become a very slow
    function call if a large radius is used or the callback provided itself
    isn't optimized.

    Always check if the index is in bounds both in the callback and in the
    returned values.  These values can go into the negatives as well.

    @type x: int
    @param x: x center of the field-of-view
    @type y: int
    @param y: y center of the field-of-view
    @type callback: function
    @param callback: This should be a function that takes two positional arguments x,y
                     and returns True if the tile at that position is transparent
                     or False if the tile blocks light or is out of bounds.
    @type fov: string
    @param fov: The type of field-of-view to be used.  Available types are:

                'BASIC', 'DIAMOND', 'SHADOW', 'RESTRICTIVE', 'PERMISSIVE',
                'PERMISSIVE0', 'PERMISSIVE1', ..., 'PERMISSIVE8'
    @type radius: float
    @param radius: Raduis of the field-of-view.

                   When sphere is True a floating point can be used to fine-tune
                   the range.  Otherwise the radius is just rounded up.

                   Be careful as a large radius has an exponential affect on
                   how long this function takes.
    @type lightWalls: boolean
    @param lightWalls: Include or exclude wall tiles in the field-of-view.
    @type sphere: boolean
    @param sphere: True for a spherical field-of-view.  False for a square one.

    @rtype: set((x, y), ...)
    @return: Returns a set of (x, y) points that are within the field-of-view.
    """
    trueRadius = radius
    radius = int(_math.ceil(radius))
    mapSize = radius * 2 + 1
    fov = _get_fov_type(fov)

    setProp = _lib.TCOD_map_set_properties # make local
    inFOV = _lib.TCOD_map_is_in_fov

    tcodMap = _lib.TCOD_map_new(mapSize, mapSize)
    try:
        # pass no.1, write callback data to the tcodMap
        for x_, y_ in _itertools.product(range(mapSize), range(mapSize)):
            pos = (x_ + x - radius,
                   y_ + y - radius)
            transparent = bool(callback(*pos))
            setProp(tcodMap, x_, y_, transparent, False)

        # pass no.2, compute fov and build a list of points
        _lib.TCOD_map_compute_fov(tcodMap, radius, radius, radius, lightWalls, fov)
        touched = set() # points touched by field of view
        for x_, y_ in _itertools.product(range(mapSize), range(mapSize)):
            if sphere and _math.hypot(x_ - radius, y_ - radius) > trueRadius:
                continue
            if inFOV(tcodMap, x_, y_):
                touched.add((x_ + x - radius, y_ + y - radius))
    finally:
        _lib.TCOD_map_delete(tcodMap)
    return touched

def bresenham(x1, y1, x2, y2):
    """
    Return a list of points in a bresenham line.

    Implementation hastily copied from RogueBasin.

    @return: Returns a list of (x, y) points, including both the start and
             endpoints.
    """
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


__all__ = [_var for _var in locals().keys() if _var[0] != '_']

quickFOV = _style.backport(quick_fov)
AStar.getPath = _style.backport(AStar.get_path)
