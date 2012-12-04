
import math
import ctypes

from .__tcod import _lib, _PATHCALL

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

    @classmethod
    def FromMap(cls, map, diagnalCost=math.sqrt(2)):
        self = cls.__new__(cls)
        self.callback = None
        self._as_parameter_ = _lib.TCOD_path_new_using_map(map, diagnalCost)
        return self
                                     
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
    
    @classmethod
    def FromMap(cls, map, diagnalCost=math.sqrt(2)):
        self = cls.__new__(cls)
        self.callback = None
        self._as_parameter_ = _lib.TCOD_dijkstra_new_using_map(map, diagnalCost)
        return self
        
    def __del__(self):
        _lib.TCOD_dijkstra_delete(self)
        
    def setPole(self, x, y):
        self.x, self.y = x, y
        _lib.TCOD_dijkstra_compute(self, x, y)
        
    def getPathFrom(self, startX, startY):
        pass
        
    def getPathTo(self, destX, destY):
        pass
    
def bresenham(x1, y1, x2, y2):
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