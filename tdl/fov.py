
import itertools
import ctypes
import array

from .__tcod import _lib

class Map(object):
    
    def __init__(self, width, height, callback=None):
        self._width = int(width)
        self._height = int(height)
        self._size = self._width * self._height
        self._tcodMap = _lib.TCOD_map_new(width, height)
        self._as_parameter_ = self._tcodMap
        self._callback = callback
        self._clean = set()
        #self._walkable = array.array('b', [0] * self._size)
        #self._transparent = array.array('b', [0] * self._size)
        
    def __del__(self):
        _lib.TCOD_map_delete(self)
        
    def _pointsInRadius(self, x, y, radius):
        'returns a list of (x, y) items'
        x = range(max(0, x - radius), min(x + radius + 1, self._width))
        y = range(max(0, y - radius), min(y + radius + 1, self._height))
        return itertools.product(x, y)
    
    def _pointsInRadiusC(self, x, y, radius):
        'returns a list of ((x, ctypeX), (y, ctypeY)) items'
        c_int = ctypes.c_int
        x = ((i, c_int(i)) for i in
             range(max(0, x - radius), min(x + radius + 1, self._width)))
        y = ((i, c_int(i)) for i in
             range(max(0, y - radius), min(y + radius + 1, self._height)))
        return itertools.product(x, y)
        
    def setFromCallbacks(self, walkableCall, transparentCall):
        for x, y in itertools.product(range(self._width), range(self._height)):
            _lib.TCOD_map_set_properties(self._tcodMap, x, y,
                                         walkableCall(x, y),
                                         transparentCall(x, y))
        
    def set(self, x, y, walkable, transparent):
        #walkable = bool(walkable)
        #transparent = bool(transparent)
        _lib.TCOD_map_set_properties(self._as_parameter_,
                                     x, y, walkable, transparent)
        
    def _updateMap(self, x, y, radius):
        if not self._callback:
            return
        c_bool = ctypes.c_bool
        for (x, cX),(y, cY) in self._pointsInRadiusC(x, y, radius):
            #if (x, y) not in self._clean:
            #    self._clean.add((x,y))
                transparent = c_bool(self._callback(x, y))
                _lib.TCOD_map_set_properties(self._as_parameter_,
                                         cX, cY, transparent, transparent)
        
    def computeFOV(self, x, y, fov='PERMISSIVE', radius=8, lightWalls=True):
        oldFOV = fov
        fov = str(fov).upper()
        FOVTYPES = {'BASIC' : 0, 'DIAMOND': 1, 'SHADOW': 2, 'RESTRICTIVE': 12,
                    'PERMISSIVE': 11}
        if fov in FOVTYPES:
            fov = FOVTYPES[fov]
        elif fov[:10] == 'PERMISSIVE' and fov[10].isdigit():
            fov = 4 + int(fov[10])
        else:
            raise TDLError('No such fov option as %s' % oldFOV)
            
        self._updateMap(x, y, radius)
        _lib.TCOD_map_compute_fov(self, x, y, radius, lightWalls, fov)
        return self._listFOV(x, y, radius)
            
    def _iterFOV(self, x, y, radius):
        inFOV = _lib.TCOD_map_is_in_fov
        map = self._as_parameter_
        for (x, cX),(y, cY) in self._pointsInRadiusC(x, y, radius):
            if inFOV(map, cX, cY):
                yield(x, y)
                
    def _listFOV(self, x, y, radius):
        return list(self._iterFOV(x, y, radius))

__all__ = [Map]