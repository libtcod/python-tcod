
from __future__ import absolute_import

import operator

import numpy as np

from tcod.tcod import _cdata
from tcod.libtcod import ffi, lib
import tcod.libtcod

"""Noise implementation constants"""
SIMPLE = 0
FBM = 1
TURBULENCE = 2

class Noise(object):
    """

    The ``hurst`` exponent describes the raggedness of the resultant noise,
    with a higher value leading to a smoother noise.
    Not used with tcod.noise.SIMPLE.

    ``lacunarity`` is a multiplier that determines how fast the noise
    frequency increases for each successive octave.
    Not used with tcod.noise.SIMPLE.

    Args:
        dimensions (int): Must be from 1 to 4.
        algorithm (int): Defaults to NOISE_SIMPLEX
        implementation (int): Defaults to tcod.noise.SIMPLE
        hurst (float): The hurst exponent.  Should be in the 0.0-1.0 range.
        lacunarity (float): The noise lacunarity.
        octaves (float): The level of detail on fBm and turbulence
                         implementations.
        rand (Optional[Random]): A Random instance, or None.

    .. versionadded:: 2.0
    """

    def __init__(self, dimensions, algorithm=2, implementation=SIMPLE,
                 hurst=0.5, lacunarity=2.0, octaves=4, rand=None):
        if not 0 < dimensions <= 4:
            raise ValueError('dimensions must be in range 0 < n <= 4, got %r' %
                             (dimensions,))
        self._random = rand
        self._random_c = _cdata(rand)
        self._algorithm = algorithm
        self.noise_c = ffi.gc(
            ffi.cast(
                'perlin_data_t*',
                lib.TCOD_noise_new(dimensions, hurst, lacunarity,
                                   self._random_c),
                ),
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
    def dimentions(self): # deprecated
        return self.dimensions

    @property
    def algorithm(self):
        return self.noise_c.noise_type
    @algorithm.setter
    def algorithm(self, value):
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
        return self.noise_c.H

    @property
    def lacunarity(self):
        return self.noise_c.lacunarity

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

        The :any:`sample_ogrid` method performs better as there is a lot of
        overhead when working with large mesh-grids.

        Args:
            mgrid (numpy.ndarray): A mesh-grid array of points to sample.
                A contiguous array of type :any:`numpy.float32` is preferred.

        Returns:
            numpy.ndarray: An array of sampled points
                           with the shape: ``mgrid.shape[:-1]``.
                           The ``dtype`` is :any:`numpy.float32`.

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
            numpy.ndarray: An array of sampled points.  The ``shape`` is based
                           on the lengths of the open mesh-grid arrays.
                           The ``dtype`` is :any:`numpy.float32`.

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

    def __getstate__(self):
        if self.dimensions < 4 and self.noise_c.waveletTileData == ffi.NULL:
            # Trigger a side effect of wavelet, so that copies will be synced.
            saved_algo = self.algorithm
            self.algorithm = tcod.libtcod.NOISE_WAVELET
            self.get_point()
            self.algorithm = saved_algo

        waveletTileData = None
        if self.noise_c.waveletTileData != ffi.NULL:
            waveletTileData = ffi.buffer(
                self.noise_c.waveletTileData[0:32*32*32])[:]
        return (
            self._random,
            self.implementation,
            self.octaves,
            self.noise_c.ndim,
            ffi.buffer(self.noise_c.map)[:],
            ffi.buffer(self.noise_c.buffer)[:],
            self.noise_c.H,
            self.noise_c.lacunarity,
            ffi.buffer(self.noise_c.exponent)[:],
            waveletTileData,
            self.noise_c.noise_type,
            )

    def __setstate__(self, state):
        self._random = state[0]
        self._random_c = _cdata(self._random)
        self.noise_c = ffi.new('perlin_data_t*')
        self.noise_c.ndim = state[3]
        ffi.buffer(self.noise_c.map)[:] = state[4]
        ffi.buffer(self.noise_c.buffer)[:] = state[5]
        self.noise_c.H = state[6]
        self.noise_c.lacunarity = state[7]
        ffi.buffer(self.noise_c.exponent)[:] = state[8]
        if state[9]:
            # high change of this being prematurely garbage collected!
            self.__waveletTileData = ffi.new('float[]', 32*32*32)
            ffi.buffer(self.__waveletTileData)[:] = state[9]
        self.noise_c.noise_type = state[10]
        self._tdl_noise_c = ffi.new('TDLNoise*',
                                    (self.noise_c, self.noise_c.ndim,
                                     state[1], state[2]))
