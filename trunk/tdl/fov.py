
import itertools

from .__tcod import _lib

class Map(object):
    
    def __init__(self, width, height):
        self._width = int(width)
        self._height = int(height)
        self._size = self._width * self._height
        self._tcodMap = _lib.TCOD_map_new(width, height)
        self._as_parameter_ = self._tcodMap
        self._fovConf = (0, 0, 0) # fov settings (x, y, radius)
        #self._walkable = array.array('b', [0] * self._size)
        #self._transparent = array.array('b', [0] * self._size)
        
    def __del__(self):
        _lib.TCOD_map_delete(self)
        
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
            raise TDLError('No such fov as %s' % oldFOV)
        _lib.TCOD_map_compute_fov(self, x, y, radius, lightWalls, fov)
        self._fovConf = (x, y, radius)
        return self._iterFOV()
            
    def _iterFOV(self):
        x, y, radius = self._fovConf
        inFOV = _lib.TCOD_map_is_in_fov
        map = self._as_parameter_
        for x,y in itertools.product(range(x - radius, x + radius + 1),
                                     range(y - radius, y + radius + 1)):
            if inFOV(map, x, y):
                yield(x, y)
