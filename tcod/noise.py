
from __future__ import absolute_import as _

from tcod.tcod import _cdata
from tcod.tcod import _CDataWrapper
from tcod.libtcod import ffi, lib

# Noise implementation constants
SIMPLE = 0
FBM = 1
TURBULENCE = 2

class Noise(_CDataWrapper):
    """
    .. versionadded:: 2.0

    The ``hurst`` exponent describes the raggedness of the resultant noise,
    with a higher value leading to a smoother noise.
    Not used with NOISE_IMP_SIMPLE.

    ``lacunarity`` is a multiplier that determines how fast the noise
    frequency increases for each successive octave.
    Not used with NOISE_IMP_SIMPLE.

    Args:
        dimentions (int): Must be from 1 to 4.
        algorithm (int): Defaults to NOISE_SIMPLEX
        implementation (int): Defaults to NOISE_IMP_SIMPLE
        hurst (float): The hurst exponent.  Should be in the 0.0-1.0 range.
        lacunarity (float): The noise lacunarity.
        octaves (float): The level of detail on fBm and turbulence
                         implementations.
        rand (Optional[Random]): A Random instance, or None.
    """
    def __init__(self, *args, **kargs):
        self.octaves = 4
        self.implementation = SIMPLE
        self._cdata_random = None # keep alive the random cdata instance
        self._algorithm = None
        self._dimentions = None
        self._hurst = None
        self._lacunarity = None
        super(Noise, self).__init__(*args, **kargs)
        if not self.cdata:
            self._init(*args, **kargs)

    def _init(self, dimentions, algorithm=2, implementation=SIMPLE,
              hurst=0.5, lacunarity=2.0, octaves=4, rand=None):
        self._cdata_random = _cdata(rand)
        self.implementation = implementation
        self._dimentions = dimentions
        self._hurst = hurst
        self._lacunarity = lacunarity
        self.octaves = octaves
        self.cdata = ffi.gc(lib.TCOD_noise_new(self._dimentions, self._hurst,
                                               self._lacunarity,
                                               self._cdata_random),
                            lib.TCOD_noise_delete)
        self.algorithm = algorithm

    @property
    def algorithm(self):
        return self._algorithm
    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
        lib.TCOD_noise_set_type(self.cdata, value)

    @property
    def dimentions(self):
        return self._dimentions

    @property
    def hurst(self):
        return self._hurst

    @property
    def lacunarity(self):
        return self._lacunarity

    def get_point(self, x=0, y=0, z=0, w=0):
        """Return the noise value at the (x, y, z, w) point.

        Args:
            x (float): The position on the 1st axis.
            y (float): The position on the 2nd axis.
            z (float): The position on the 3rd axis.
            w (float): The position on the 4th axis.
        """
        if self.implementation == SIMPLE:
            return lib.TCOD_noise_get(self.cdata, (x, y, z, w))
        elif self.implementation == FBM:
            return lib.TCOD_noise_get_fbm(self.cdata, (x, y, z, w),
                                          self.octaves)
        elif self.implementation == TURBULENCE:
            return lib.TCOD_noise_get_turbulence(self.cdata, (x, y, z, w),
                                                 self.octaves)
        raise RuntimeError('implementation must be one of tcod.NOISE_IMP_*')

