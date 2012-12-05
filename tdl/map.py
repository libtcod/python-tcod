"""
    Rogue-like map utilitys such as line-of-sight, field-of-view, and path-finding.
    
"""
import array
import ctypes
import itertools
import math

from .__tcod import _lib, _PATHCALL

def _getFOVType(fov):
    "Return a FOV from a string"
    oldFOV = fov
    fov = str(fov).upper()
    FOVTYPES = {'BASIC' : 0, 'DIAMOND': 1, 'SHADOW': 2, 'RESTRICTIVE': 12,
                'PERMISSIVE': 11}
    if fov in FOVTYPES:
        fov = FOVTYPES[fov]
    elif fov[:10] == 'PERMISSIVE' and fov[10].isdigit() and fov[10] != '9':
        fov = 4 + int(fov[10])
    else:
        raise TDLError('No such fov option as %s' % oldFOV)
    return fov

# class Map(object):
    
    # def __init__(self, width, height, callback=None):
        # self._width = int(width)
        # self._height = int(height)
        # self._size = self._width * self._height
        # self._tcodMap = _lib.TCOD_map_new(width, height)
        # self._as_parameter_ = self._tcodMap
        # self._callback = callback
        # #self._clean = set()
        # #self._walkable = array.array('b', [0] * self._size)
        # #self._transparent = array.array('b', [0] * self._size)
        
    # def __del__(self):
        # _lib.TCOD_map_delete(self)
        
    # def _pointsInRadius(self, x, y, radius):
        # 'returns a list of (x, y) items'
        # x = range(max(0, x - radius), min(x + radius + 1, self._width))
        # y = range(max(0, y - radius), min(y + radius + 1, self._height))
        # return itertools.product(x, y)
    
    # def _pointsInRadiusC(self, x, y, radius):
        # 'returns a list of ((x, ctypeX), (y, ctypeY)) items'
        # c_int = ctypes.c_int
        # x = ((i, c_int(i)) for i in
             # range(max(0, x - radius), min(x + radius + 1, self._width)))
        # y = ((i, c_int(i)) for i in
             # range(max(0, y - radius), min(y + radius + 1, self._height)))
        # return itertools.product(x, y)
        
    # def setFromCallbacks(self, walkableCall, transparentCall):
        # for x, y in itertools.product(range(self._width), range(self._height)):
            # _lib.TCOD_map_set_properties(self._tcodMap, x, y,
                                         # transparentCall(x, y),
                                         # walkableCall(x, y))
        
    # def set(self, x, y, walkable, transparent):
        # #walkable = bool(walkable)
        # #transparent = bool(transparent)
        # _lib.TCOD_map_set_properties(self._as_parameter_,
                                     # x, y, walkable, transparent)
        
    # def _updateMap(self, x, y, radius):
        # if not self._callback:
            # return
        # c_bool = ctypes.c_bool
        # for (x, cX),(y, cY) in self._pointsInRadiusC(x, y, radius):
            # #if (x, y) not in self._clean:
            # #    self._clean.add((x,y))
                # transparent = c_bool(self._callback(x, y))
                # _lib.TCOD_map_set_properties(self._as_parameter_,
                                         # cX, cY, transparent, transparent)
        
    # def computeFOV(self, x, y, fov='PERMISSIVE', radius=8, lightWalls=True):
        # """
        
        # @type x: int
        # @param x:
        # @type y: int
        # @param y:
        # @type fov: string
        # @type radius: int
        # @type lightWalls: boolean
        
        # @rtype: list
        # @return: Returns a list of (x, y) coordinates that are within the field-of-view
        # """
        # fov = _getFOVType(fov)
            
        # self._updateMap(x, y, radius)
        # _lib.TCOD_map_compute_fov(self, x, y, radius, lightWalls, fov)
        # return self._listFOV(x, y, radius)
            
    # def _iterFOV(self, x, y, radius):
        # inFOV = _lib.TCOD_map_is_in_fov
        # map = self._as_parameter_
        # for (x, cX),(y, cY) in self._pointsInRadiusC(x, y, radius):
            # if inFOV(map, cX, cY):
                # yield(x, y)
                
    # def _listFOV(self, x, y, radius):
        # return list(self._iterFOV(x, y, radius))

class AStar(object):

    def __init__(self, width, height, callback, diagnalCost=math.sqrt(2)):
        def newCallback(fromX, fromY, toX, toY, null):
            pathCost = callback(toX, toY) # expecting a float or 0
            if pathCost:
                return pathCost
            return 0.0
        self.callback = _PATHCALL(newCallback)
        self._as_parameter_ = _lib.TCOD_path_new_using_function(width, height,
                                     self.callback, None, diagnalCost)

    #@classmethod
    #def FromMap(cls, map, diagnalCost=math.sqrt(2)):
    #    self = cls.__new__(cls)
    #    self.callback = None
    #    self._as_parameter_ = _lib.TCOD_path_new_using_map(map, diagnalCost)
    #    return self
                                     
    def __del__(self):
        _lib.TCOD_path_delete(self)
        
    def getPath(self, origX, origY, destX, destY):
        found = _lib.TCOD_path_compute(self, origX, origY, destX, destY)
        if not found:
            return [] # path not found
        x, y = ctypes.c_int(), ctypes.c_int()
        xRef, yRef = ctypes.byref(x), ctypes.byref(y)
        recalculate = ctypes.c_bool(False)
        path = []
        while _lib.TCOD_path_walk(self, xRef, yRef, recalculate):
            path.append((x.value, y.value))
        return path
        
                                                
    
class Dijkstra(object):

    def __init__(self, width, height, callback, diagnalCost=math.sqrt(2)):
        def newCallback(fromX, fromY, toX, toY, null):
            pathCost = callback(toX, toY) # expecting a float or 0
            return pathCost
        self.callback = _PATHCALL(newCallback)
        self._as_parameter_ = _lib.TCOD_dijkstra_new_using_function(width, height,
                                     self.callback, None, diagnalCost)
        # add code to compute here with x,y
    
    #@classmethod
    #def FromMap(cls, map, diagnalCost=math.sqrt(2)):
    #    self = cls.__new__(cls)
    #    self.callback = None
    #    self._as_parameter_ = _lib.TCOD_dijkstra_new_using_map(map, diagnalCost)
    #    return self
        
    def __del__(self):
        _lib.TCOD_dijkstra_delete(self)
        
    def setPole(self, x, y):
        self.x, self.y = x, y
        _lib.TCOD_dijkstra_compute(self, x, y)
        
    def getPathFrom(self, startX, startY):
        pass
        
    def getPathTo(self, destX, destY):
        pass
    
def quickFOV(x, y, callback, fov='PERMISSIVE', radius=7.5, lightWalls=True, sphere=True):
    """All field-of-view functionality in one call.
    
    @type x: int
    @param x: x origin of the field-of-view
    @type y: int
    @param y: y origin of the field-of-view
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
    
    @rtype: iterator
    @return: Returns an iterator of (x, y) points that are within the field-of-view
    """
    trueRadius = radius
    radius = math.ceil(radius)
    mapSize = radius * 2 + 1
    fov = _getFOVType(fov)
    
    setProp = _lib.TCOD_map_set_properties # make local
    inFOV = _lib.TCOD_map_is_in_fov
    
    cTrue = ctypes.c_bool(1)
    cFalse = ctypes.c_bool(False)
    try:
        tcodMap = _lib.TCOD_map_new(mapSize, mapSize)
        # pass one, write callback data to the tcodMap
        for (x_, cX), (y_, cY) in itertools.product(((i, ctypes.c_int(i)) for i in range(mapSize)),
                                                    ((i, ctypes.c_int(i)) for i in range(mapSize))):
            
            pos = (x_ + x - radius, 
                   y_ + y - radius)
            transparent = bool(callback(*pos))
            setProp(tcodMap, cX, cY, transparent, cFalse)
        
        # pass two, compute fov and build a list of points
        _lib.TCOD_map_compute_fov(tcodMap, radius, radius, radius, lightWalls, fov)
        touched = [] # points touched by field of view
        for (x_, cX),(y_, cY) in itertools.product(((i, ctypes.c_int(i)) for i in range(mapSize)),
                                                   ((i, ctypes.c_int(i)) for i in range(mapSize))):
            if sphere and math.hypot(x_ - radius, y_ - radius) > trueRadius:
                continue
            if inFOV(tcodMap, cX, cY):
                touched.append((x_ + x - radius, y_ + y - radius))
    finally:
        _lib.TCOD_map_delete(tcodMap)
    return touched
    
# def bresenham(x1, y1, x2, y2):
    # points = []
    # issteep = abs(y2-y1) > abs(x2-x1)
    # if issteep:
        # x1, y1 = y1, x1
        # x2, y2 = y2, x2
    # rev = False
    # if x1 > x2:
        # x1, x2 = x2, x1
        # y1, y2 = y2, y1
        # rev = True
    # deltax = x2 - x1
    # deltay = abs(y2-y1)
    # error = int(deltax / 2)
    # y = y1
    # ystep = None
    # if y1 < y2:
        # ystep = 1
    # else:
        # ystep = -1
    # for x in range(x1, x2 + 1):
        # if issteep:
            # points.append((y, x))
        # else:
            # points.append((x, y))
        # error -= deltay
        # if error < 0:
            # y += ystep
            # error += deltax
    # # Reverse the list if the coordinates were reversed
    # if rev:
        # points.reverse()
    # return points
    
__all__ = ['AStar', 'Dijkstra', 'quickFOV']
