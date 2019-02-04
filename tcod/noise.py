"""
The :any:`Noise.sample_mgrid` and :any:`Noise.sample_ogrid` methods are
multi-threaded operations when the Python runtime supports OpenMP.
Even when single threaded these methods will perform much better than
multiple calls to :any:`Noise.get_point`.

Example::

    import numpy as np
    import tcod
    import tcod.noise

    noise = tcod.noise.Noise(
        dimensions=2,
        algorithm=tcod.NOISE_SIMPLEX,
        implementation=tcod.noise.TURBULENCE,
        hurst=0.5,
        lacunarity=2.0,
        octaves=4,
        seed=None,
        )

    # Create a 5x5 open multi-dimensional mesh-grid.
    ogrid = [np.arange(5, dtype=np.float32),
             np.arange(5, dtype=np.float32)]
    print(ogrid)

    # Scale the grid.
    ogrid[0] *= 0.25
    ogrid[1] *= 0.25

    # Return the sampled noise from this grid of points.
    samples = noise.sample_ogrid(ogrid)
    print(samples)
"""
from typing import Any, Optional

import numpy as np

from tcod._internal import deprecate
from tcod.libtcod import ffi, lib
import tcod.constants
import tcod.random

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
        seed (Optional[Random]): A Random instance, or None.

    Attributes:
        noise_c (CData): A cffi pointer to a TCOD_noise_t object.
    """

    def __init__(
        self,
        dimensions: int,
        algorithm: int = 2,
        implementation: int = SIMPLE,
        hurst: float = 0.5,
        lacunarity: float = 2.0,
        octaves: float = 4,
        seed: Optional[tcod.random.Random] = None,
    ):
        if not 0 < dimensions <= 4:
            raise ValueError(
                "dimensions must be in range 0 < n <= 4, got %r"
                % (dimensions,)
            )
        self._random = seed
        _random_c = seed.random_c if seed else ffi.NULL
        self._algorithm = algorithm
        self.noise_c = ffi.gc(
            ffi.cast(
                "struct TCOD_Noise*",
                lib.TCOD_noise_new(dimensions, hurst, lacunarity, _random_c),
            ),
            lib.TCOD_noise_delete,
        )
        self._tdl_noise_c = ffi.new(
            "TDLNoise*", (self.noise_c, dimensions, 0, octaves)
        )
        self.implementation = implementation  # sanity check

    @property
    def dimensions(self) -> int:
        return int(self._tdl_noise_c.dimensions)

    @property  # type: ignore
    @deprecate("This is a misspelling of 'dimensions'.")
    def dimentions(self) -> int:
        return self.dimensions

    @property
    def algorithm(self) -> int:
        return int(self.noise_c.noise_type)

    @algorithm.setter
    def algorithm(self, value: int) -> None:
        lib.TCOD_noise_set_type(self.noise_c, value)

    @property
    def implementation(self) -> int:
        return int(self._tdl_noise_c.implementation)

    @implementation.setter
    def implementation(self, value: int) -> None:
        if not 0 <= value < 3:
            raise ValueError("%r is not a valid implementation. " % (value,))
        self._tdl_noise_c.implementation = value

    @property
    def hurst(self) -> float:
        return float(self.noise_c.H)

    @property
    def lacunarity(self) -> float:
        return float(self.noise_c.lacunarity)

    @property
    def octaves(self) -> float:
        return float(self._tdl_noise_c.octaves)

    @octaves.setter
    def octaves(self, value: float) -> None:
        self._tdl_noise_c.octaves = value

    def get_point(
        self, x: float = 0, y: float = 0, z: float = 0, w: float = 0
    ) -> float:
        """Return the noise value at the (x, y, z, w) point.

        Args:
            x (float): The position on the 1st axis.
            y (float): The position on the 2nd axis.
            z (float): The position on the 3rd axis.
            w (float): The position on the 4th axis.
        """
        return float(lib.NoiseGetSample(self._tdl_noise_c, (x, y, z, w)))

    def sample_mgrid(self, mgrid: np.array) -> np.array:
        """Sample a mesh-grid array and return the result.

        The :any:`sample_ogrid` method performs better as there is a lot of
        overhead when working with large mesh-grids.

        Args:
            mgrid (numpy.ndarray): A mesh-grid array of points to sample.
                A contiguous array of type `numpy.float32` is preferred.

        Returns:
            numpy.ndarray: An array of sampled points.

                This array has the shape: ``mgrid.shape[:-1]``.
                The ``dtype`` is `numpy.float32`.
        """
        mgrid = np.ascontiguousarray(mgrid, np.float32)
        if mgrid.shape[0] != self.dimensions:
            raise ValueError(
                "mgrid.shape[0] must equal self.dimensions, "
                "%r[0] != %r" % (mgrid.shape, self.dimensions)
            )
        out = np.ndarray(mgrid.shape[1:], np.float32)
        if mgrid.shape[1:] != out.shape:
            raise ValueError(
                "mgrid.shape[1:] must equal out.shape, "
                "%r[1:] != %r" % (mgrid.shape, out.shape)
            )
        lib.NoiseSampleMeshGrid(
            self._tdl_noise_c,
            out.size,
            ffi.cast("float*", mgrid.ctypes.data),
            ffi.cast("float*", out.ctypes.data),
        )
        return out

    def sample_ogrid(self, ogrid: np.array) -> np.array:
        """Sample an open mesh-grid array and return the result.

        Args
            ogrid (Sequence[Sequence[float]]): An open mesh-grid.

        Returns:
            numpy.ndarray: An array of sampled points.

                The ``shape`` is based on the lengths of the open mesh-grid
                arrays.
                The ``dtype`` is `numpy.float32`.
        """
        if len(ogrid) != self.dimensions:
            raise ValueError(
                "len(ogrid) must equal self.dimensions, "
                "%r != %r" % (len(ogrid), self.dimensions)
            )
        ogrids = [np.ascontiguousarray(array, np.float32) for array in ogrid]
        out = np.ndarray([array.size for array in ogrids], np.float32)
        lib.NoiseSampleOpenMeshGrid(
            self._tdl_noise_c,
            len(ogrids),
            out.shape,
            [ffi.cast("float*", array.ctypes.data) for array in ogrids],
            ffi.cast("float*", out.ctypes.data),
        )
        return out

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        if self.dimensions < 4 and self.noise_c.waveletTileData == ffi.NULL:
            # Trigger a side effect of wavelet, so that copies will be synced.
            saved_algo = self.algorithm
            self.algorithm = tcod.constants.NOISE_WAVELET
            self.get_point()
            self.algorithm = saved_algo

        waveletTileData = None
        if self.noise_c.waveletTileData != ffi.NULL:
            waveletTileData = list(
                self.noise_c.waveletTileData[0 : 32 * 32 * 32]
            )
            state["_waveletTileData"] = waveletTileData

        state["noise_c"] = {
            "ndim": self.noise_c.ndim,
            "map": list(self.noise_c.map),
            "buffer": [list(sub_buffer) for sub_buffer in self.noise_c.buffer],
            "H": self.noise_c.H,
            "lacunarity": self.noise_c.lacunarity,
            "exponent": list(self.noise_c.exponent),
            "waveletTileData": waveletTileData,
            "noise_type": self.noise_c.noise_type,
        }
        state["_tdl_noise_c"] = {
            "dimensions": self._tdl_noise_c.dimensions,
            "implementation": self._tdl_noise_c.implementation,
            "octaves": self._tdl_noise_c.octaves,
        }
        return state

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):  # deprecated format
            return self._setstate_old(state)
        # unpack wavelet tile data if it exists
        if "_waveletTileData" in state:
            state["_waveletTileData"] = ffi.new(
                "float[]", state["_waveletTileData"]
            )
            state["noise_c"]["waveletTileData"] = state["_waveletTileData"]
        else:
            state["noise_c"]["waveletTileData"] = ffi.NULL

        # unpack TCOD_Noise and link to Random instance
        state["noise_c"]["rand"] = state["_random"].random_c
        state["noise_c"] = ffi.new("struct TCOD_Noise*", state["noise_c"])

        # unpack TDLNoise and link to libtcod noise
        state["_tdl_noise_c"]["noise"] = state["noise_c"]
        state["_tdl_noise_c"] = ffi.new("TDLNoise*", state["_tdl_noise_c"])
        self.__dict__.update(state)

    def _setstate_old(self, state: Any) -> None:
        self._random = state[0]
        self.noise_c = ffi.new("struct TCOD_Noise*")
        self.noise_c.ndim = state[3]
        ffi.buffer(self.noise_c.map)[:] = state[4]
        ffi.buffer(self.noise_c.buffer)[:] = state[5]
        self.noise_c.H = state[6]
        self.noise_c.lacunarity = state[7]
        ffi.buffer(self.noise_c.exponent)[:] = state[8]
        if state[9]:
            # high change of this being prematurely garbage collected!
            self.__waveletTileData = ffi.new("float[]", 32 * 32 * 32)
            ffi.buffer(self.__waveletTileData)[:] = state[9]
        self.noise_c.noise_type = state[10]
        self._tdl_noise_c = ffi.new(
            "TDLNoise*", (self.noise_c, self.noise_c.ndim, state[1], state[2])
        )
