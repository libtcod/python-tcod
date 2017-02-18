
from __future__ import absolute_import

import operator

import numpy as np

from tcod.tcod import _cdata
from tcod.libtcod import ffi, lib

# Noise implementation constants
SIMPLE = 0
FBM = 1
TURBULENCE = 2

class Noise(object):
    """
    .. versionadded:: 2.0

    The ``hurst`` exponent describes the raggedness of the resultant noise,
    with a higher value leading to a smoother noise.
    Not used with NOISE_IMP_SIMPLE.

    ``lacunarity`` is a multiplier that determines how fast the noise
    frequency increases for each successive octave.
    Not used with NOISE_IMP_SIMPLE.

    Args:
        dimensions (int): Must be from 1 to 4.
        algorithm (int): Defaults to NOISE_SIMPLEX
        implementation (int): Defaults to NOISE_IMP_SIMPLE
        hurst (float): The hurst exponent.  Should be in the 0.0-1.0 range.
        lacunarity (float): The noise lacunarity.
        octaves (float): The level of detail on fBm and turbulence
                         implementations.
        rand (Optional[Random]): A Random instance, or None.
    """

    def __init__(self, dimensions, algorithm=2, implementation=SIMPLE,
                 hurst=0.5, lacunarity=2.0, octaves=4, rand=None):
        if not 0 < dimensions <= 4:
            raise ValueError('dimensions must be in range 0 < n <= 4, got %r' %
                             (dimensions,))
        self._random = rand
        self._random_c = _cdata(rand)
        self._algorithm = algorithm
        self._hurst = hurst
        self._lacunarity = lacunarity
        self.noise_c = ffi.gc(
            lib.TCOD_noise_new(dimensions, hurst, lacunarity, self._random_c),
            lib.TCOD_noise_delete)
        self._tdl_noise_c = ffi.new('TDLNoise*', (self.noise_c,
                                                  dimensions,
                                                  0,
                                                  octaves))
        self.implementation = implementation # sanity check

    @property
    def dimensions(self):
        return self._tdl_noise_c.dimensions

    @property
    def algorithm(self):
        return self._algorithm
    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
        lib.TCOD_noise_set_type(self.noise_c, value)

    @property
    def implementation(self):
        return self._tdl_noise_c.implementation
    @implementation.setter
    def implementation(self, value):
        if not 0 <= value < 3:
            raise ValueError('%r is not a valid implementation. ' % (value,))
        self._tdl_noise_c.implementation = value

    @property
    def hurst(self):
        return self._hurst

    @property
    def lacunarity(self):
        return self._lacunarity

    @property
    def octaves(self):
        return self._tdl_noise_c.octaves
    @octaves.setter
    def octaves(self, value):
        self._tdl_noise_c.octaves = value

    def get_point(self, x=0, y=0, z=0, w=0):
        """Return the noise value at the (x, y, z, w) point.

        Args:
            x (float): The position on the 1st axis.
            y (float): The position on the 2nd axis.
            z (float): The position on the 3rd axis.
            w (float): The position on the 4th axis.
        """
        return lib.NoiseGetSample(self._tdl_noise_c, (x, y, z, w))

    def sample_mgrid(self, mgrid):
        """Sample a mesh-grid array and return the result.

        Args:
            mgrid (numpy.ndarray): A mesh-grid array of points to sample.

        Returns:
            numpy.ndarray: A float32 array of sampled points
                           with the shape: ``mgrid.shape[:-1]``.

        .. versionadded:: 2.2
        """
        mgrid = np.ascontiguousarray(mgrid, np.float32)
        if mgrid.shape[0] != self.dimensions:
            raise ValueError('mgrid.shape[0] must equal self.dimensions, '
                             '%r[0] != %r' % (mgrid.shape, self.dimensions))
        out = np.ndarray(mgrid.shape[1:], np.float32)
        if mgrid.shape[1:] != out.shape:
            raise ValueError('mgrid.shape[1:] must equal out.shape, '
                             '%r[1:] != %r' % (mgrid.shape, out.shape))
        lib.NoiseSampleMeshGrid(self._tdl_noise_c, out.size,
                                ffi.cast('float*', mgrid.ctypes.data),
                                ffi.cast('float*', out.ctypes.data))
        return out

    def sample_ogrid(self, ogrid):
        """Sample an open mesh-grid array and return the result.

        Args
            ogrid (Sequence[numpy.ndarray]): An open mesh-grid.

        Returns:
            numpy.ndarray:  A float32 array of sampled points.  Shape is based
                            on the lengths of the open mesh-grid arrays.

        .. versionadded:: 2.2
        """
        if len(ogrid) != self.dimensions:
            raise ValueError('len(ogrid) must equal self.dimensions, '
                             '%r != %r' % (len(ogrid), self.dimensions))
        ogrids = [np.ascontiguousarray(array, np.float32) for array in ogrid]
        out = np.ndarray([array.size for array in ogrids], np.float32)
        lib.NoiseSampleOpenMeshGrid(
            self._tdl_noise_c,
            len(ogrids),
            out.shape,
            [ffi.cast('float*', array.ctypes.data) for array in ogrids],
            ffi.cast('float*', out.ctypes.data),
            )
        return out
