"""
    Rogue-like map utilitys such as line-of-sight, field-of-view, and path-finding.

    .. deprecated:: 3.2
        The features provided here are better realized in the
        :any:`tcod.map` and :any:`tcod.path` modules.
"""

from __future__ import absolute_import

import itertools as _itertools
import math as _math

import numpy as np

from tcod import ffi as _ffi
from tcod import lib as _lib
from tcod import ffi, lib

from tcod._internal import deprecate
import tcod.map
import tcod.path
import tdl as _tdl
from . import style as _style

_FOVTYPES = {'BASIC' : 0, 'DIAMOND': 1, 'SHADOW': 2, 'RESTRICTIVE': 12,
             'PERMISSIVE': 11}

def _get_fov_type(fov):
    "Return a FOV from a string"
    oldFOV = fov
    fov = str(fov).upper()
    if fov in _FOVTYPES:
        return _FOVTYPES[fov]
    if fov[:10] == 'PERMISSIVE' and fov[10].isdigit() and fov[10] != '9':
        return 4 + int(fov[10])
    raise _tdl.TDLError('No such fov option as %s' % oldFOV)

class Map(tcod.map.Map):
    """Field-of-view and path-finding on stored data.

    .. versionchanged:: 4.1
        `transparent`, `walkable`, and `fov` are now numpy boolean arrays.

    .. versionchanged:: 4.3
        Added `order` parameter.

    .. deprecated:: 3.2
        :any:`tcod.map.Map` should be used instead.

    Set map conditions with the walkable and transparency attributes, this
    object can be iterated and checked for containment similar to consoles.

    For example, you can set all tiles and transparent and walkable with the
    following code:

    Example:
        >>> import tdl.map
        >>> map_ = tdl.map.Map(80, 60)
        >>> map_.transparent[:] = True
        >>> map_.walkable[:] = True

    Attributes:
        transparent: Map transparency

            Access this attribute with ``map.transparent[x,y]``

            Set to True to allow field-of-view rays, False will
            block field-of-view.

            Transparent tiles only affect field-of-view.
        walkable: Map accessibility

            Access this attribute with ``map.walkable[x,y]``

            Set to True to allow path-finding through that tile,
            False will block passage to that tile.

            Walkable tiles only affect path-finding.

        fov: Map tiles touched by a field-of-view computation.

            Access this attribute with ``map.fov[x,y]``

            Is True if a the tile is if view, otherwise False.

            You can set to this attribute if you want, but you'll typically
            be using it to read the field-of-view of a :any:`compute_fov` call.
    """

    def __init__(self, width, height, order='F'):
        super(Map, self).__init__(width, height, order)

    def compute_fov(self, x, y, fov='PERMISSIVE', radius=None,
                    light_walls=True, sphere=True, cumulative=False):
        """Compute the field-of-view of this Map and return an iterator of the
        points touched.

        Args:
            x (int): Point of view, x-coordinate.
            y (int): Point of view, y-coordinate.
            fov (Text): The type of field-of-view to be used.

                Available types are:
                'BASIC', 'DIAMOND', 'SHADOW', 'RESTRICTIVE', 'PERMISSIVE',
                'PERMISSIVE0', 'PERMISSIVE1', ..., 'PERMISSIVE8'
            radius (Optional[int]): Maximum view distance from the point of
                view.

                A value of 0 will give an infinite distance.
            light_walls (bool): Light up walls, or only the floor.
            sphere (bool): If True the lit area will be round instead of
                square.
            cumulative (bool): If True the lit cells will accumulate instead
                of being cleared before the computation.

        Returns:
            Iterator[Tuple[int, int]]: An iterator of (x, y) points of tiles
                touched by the field-of-view.
        """
        # refresh cdata
        if radius is None: # infinite radius
            radius = 0
        if cumulative:
            fov_copy = self.fov.copy()
        lib.TCOD_map_compute_fov(
            self.map_c, x, y, radius, light_walls, _get_fov_type(fov))
        if cumulative:
            self.fov[:] |= fov_copy
        return zip(*np.where(self.fov))


    def compute_path(self, start_x, start_y, dest_x, dest_y,
                     diagonal_cost=_math.sqrt(2)):
        """Get the shortest path between two points.

        Args:
            start_x (int): Starting x-position.
            start_y (int): Starting y-position.
            dest_x (int): Destination x-position.
            dest_y (int): Destination y-position.
            diagonal_cost (float): Multiplier for diagonal movement.

                Can be set to zero to disable diagonal movement entirely.

        Returns:
            List[Tuple[int, int]]: The shortest list of points to the
                destination position from the starting position.

            The start point is not included in this list.
        """
        return tcod.path.AStar(self, diagonal_cost).get_path(start_x, start_y,
                                                             dest_x, dest_y)

    def __iter__(self):
        return _itertools.product(range(self.width), range(self.height))

    def __contains__(self, position):
        x, y = position
        return (0 <= x < self.width) and (0 <= y < self.height)



class AStar(tcod.path.AStar):
    """An A* pathfinder using a callback.

    .. deprecated:: 3.2
        See :any:`tcod.path`.

    Before crating this instance you should make one of two types of
    callbacks:

    - A function that returns the cost to move to (x, y)
    - A function that returns the cost to move between
      (destX, destY, sourceX, sourceY)

    If path is blocked the function should return zero or None.
    When using the second type of callback be sure to set advanced=True

    Args:
        width (int): Width of the pathfinding area (in tiles.)
        height (int): Height of the pathfinding area (in tiles.)
        callback (Union[Callable[[int, int], float],
                        Callable[[int, int, int, int], float]]): A callback
            returning the cost of a tile or edge.

            A callback taking parameters depending on the setting
            of 'advanced' and returning the cost of
            movement for an open tile or zero for a
            blocked tile.
        diagnalCost (float): Multiplier for diagonal movement.

            Can be set to zero to disable diagonal movement entirely.
        advanced (bool): Give 2 additional parameters to the callback.

            A simple callback with 2 positional parameters may not
            provide enough information.  Setting this to True will
            call the callback with 2 additional parameters giving
            you both the destination and the source of movement.

            When True the callback will need to accept
            (destX, destY, sourceX, sourceY) as parameters.
            Instead of just (destX, destY).
    """

    class __DeprecatedEdgeCost(tcod.path.EdgeCostCallback):
        _CALLBACK_P = lib._pycall_path_swap_src_dest

    class __DeprecatedNodeCost(tcod.path.EdgeCostCallback):
        _CALLBACK_P = lib._pycall_path_dest_only

    def __init__(self, width, height, callback,
                 diagnalCost=_math.sqrt(2), advanced=False):
        if advanced:
            cost = self.__DeprecatedEdgeCost(callback, (width, height))
        else:
            cost = self.__DeprecatedNodeCost(callback, (width, height))
        super(AStar, self).__init__(cost, diagnalCost or 0.0)

    def get_path(self, origX, origY, destX, destY):
        """
        Get the shortest path from origXY to destXY.

        Returns:
            List[Tuple[int, int]]: Returns a list walking the path from orig
                to dest.

                This excludes the starting point and includes the destination.

                If no path is found then an empty list is returned.
        """
        return super(AStar, self).get_path(origX, origY, destX, destY)

@deprecate("This function is very slow.")
def quick_fov(x, y, callback, fov='PERMISSIVE', radius=7.5, lightWalls=True,
              sphere=True):
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

    Args:
        x (int): x center of the field-of-view
        y (int): y center of the field-of-view
        callback (Callable[[int, int], bool]):

            This should be a function that takes two positional arguments x,y
            and returns True if the tile at that position is transparent
            or False if the tile blocks light or is out of bounds.
        fov (Text): The type of field-of-view to be used.

            Available types are:
            'BASIC', 'DIAMOND', 'SHADOW', 'RESTRICTIVE', 'PERMISSIVE',
            'PERMISSIVE0', 'PERMISSIVE1', ..., 'PERMISSIVE8'
        radius (float) Radius of the field-of-view.

            When sphere is True a floating point can be used to fine-tune
            the range.  Otherwise the radius is just rounded up.

            Be careful as a large radius has an exponential affect on
            how long this function takes.
        lightWalls (bool): Include or exclude wall tiles in the field-of-view.
        sphere (bool): True for a spherical field-of-view.
            False for a square one.

    Returns:
        Set[Tuple[int, int]]: A set of (x, y) points that are within the
            field-of-view.
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

    Returns:
        List[Tuple[int, int]]: A list of (x, y) points,
            including both the start and end-points.
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


quickFOV = _style.backport(quick_fov)
AStar.getPath = _style.backport(AStar.get_path)
