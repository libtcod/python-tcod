"""
    This module provides advanced noise generation.
    
    Noise is sometimes used for over-world generation, height-maps, and
    cloud/mist/smoke effects among other things.

    You can see examples of the available noise algorithms in the libtcod
    documentation
    U{here<http://doryen.eptalys.net/data/libtcod/doc/1.5.1/html2/noise.html>}.
"""


import random
import itertools
import ctypes

import tdl
from tdl.__tcod import _lib

_MERSENNE_TWISTER = 1
_CARRY_WITH_MULTIPLY = 2

_MAX_DIMENSIONS = 4
_MAX_OCTAVES = 128

_NOISE_TYPES = {'PERLIN': 1, 'SIMPLEX': 2, 'WAVELET': 4}
_NOISE_MODES = {'FLAT': _lib.TCOD_noise_get,
                'FBM': _lib.TCOD_noise_get_fbm,
                'TURBULENCE': _lib.TCOD_noise_get_turbulence}

class Noise(object):
    """An advanced noise generator.
    """
    
    def __init__(self, algorithm='PERLIN', mode='FLAT',
                 hurst=0.5, lacunarity=2.0, octaves=4.0, seed=None, dimensions=4):
        """Create a new noise generator specifying a noise algorithm and how
        it's used.
        
        @type algorithm: string
        @param algorithm: The primary noise algorithm to be used.
                          
                          Can be one of 'PERLIN', 'SIMPLEX', 'WAVELET'
                           - 'PERLIN' -
                             A popular noise generator.
                             
                           - 'SIMPLEX' -
                             In theory this is a slightly faster generator with
                             less noticeable directional artifacts.
                             
                           - 'WAVELET'
                             A noise generator designed to reduce aliasing and
                             not lose detail when summed into a fractal
                             (as with the 'FBM' and 'TURBULENCE' modes.)
                             
                             This works faster at higher dimensions.
        
        @type mode: string
        @param mode: A secondary parameter to determine how noise is generated.
                     
                     Can be one of 'FLAT', 'FBM', 'TURBULENCE'
                      - 'FLAT' -
                        Generates the simplest form of noise.
                        This mode does not use the hurst, lacunarity,
                        and octaves parameters.
                        
                      - 'FBM' -
                        Generates fractal brownian motion.
                        
                      - 'TURBULENCE' -
                        Generates detailed noise with smoother and more
                        natural transitions.
                      
        @type hurst: float
        @param hurst: The hurst exponent describes the raggedness of the
                      resultant noise, with a higher value leading to a
                      smoother noise.
                      It should be in the 0.0-1.0 range.
        
                      This is only used in 'FBM' and 'TURBULENCE' modes.
                      
        @type lacunarity: float
        @param lacunarity: A multiplier that determines how quickly the
                           frequency increases for each successive octave.

                           The frequency of each successive octave is equal to
                           the product of the previous octave's frequency and
                           the lacunarity value.
                      
                           This is only used in 'FBM' and 'TURBULENCE' modes.
                      
        @type octaves: float
        @param octaves: Controls the amount of detail in the noise.
                        
                        This is only used in 'FBM' and 'TURBULENCE' modes.
                        
        @type seed: object
        @param seed: You can use any hashable object to be a seed for the
                     noise generator.
                     
                     If None is used then a random seed will be generated.
        """
        if algorithm.upper() not in _NOISE_TYPES:
            raise tdl.TDLError('No such noise algorithm as %s' % algorithm)
        self._algorithm = algorithm.upper()
        
        if mode.upper() not in _NOISE_MODES:
            raise tdl.TDLError('No such mode as %s' % mode)
        self._mode = mode.upper()
        
        if seed is None:
            seed = random.getrandbits(32)
        else:
            seed = hash(seed)
        self._seed = seed
        # convert values into ctypes to speed up later functions
        self._dimensions = min(_MAX_DIMENSIONS, int(dimensions))
        if self._algorithm == 'WAVELET':
            self._dimensions = min(self._dimensions, 3) # Wavelet only goes up to 3
        self._random = _lib.TCOD_random_new_from_seed(_MERSENNE_TWISTER, self._seed)
        self._hurst = ctypes.c_float(hurst)
        self._lacunarity = ctypes.c_float(lacunarity)
        self._noise = _lib.TCOD_noise_new(self._dimensions, self._hurst,
                                          self._lacunarity, self._random)
        _lib.TCOD_noise_set_type(self._noise, _NOISE_TYPES[self._algorithm])
        self._noiseFunc = _NOISE_MODES[self._mode]
        self._octaves = ctypes.c_float(octaves)
        self._useOctaves = (self._mode != 'FLAT')
        self._cFloatArray = ctypes.c_float * self._dimensions
        self._array = self._cFloatArray()
        
    def __copy__(self):
        # using the pickle method is a convenient way to clone this object
        self.__class__(self.__getstate__())
        
    def __getstate__(self):
        return (self._algorithm, self._mode,
                self._hurst.value, self._lacunarity.value, self._octaves.value,
                self._seed, self._dimensions)
        
    def __setstate__(self, state):
        self.__init__(*state)
        
    def getPoint(self, *position):
        """Return the noise value of a specific position.
        
        Example usage: value = noise.getPoint(x, y, z)
        @type position: floats
        @param position: 
        
        @rtype: float
        @return: Returns the noise value at position.
                 This will be a floating point in the 0.0-1.0 range.
        """
        #array = self._array
        #for d, pos in enumerate(position):
        #    array[d] = pos
        array = self._cFloatArray(*position)
        if self._useOctaves:
            return (self._noiseFunc(self._noise, array, self._octaves) + 1) * 0.5
        return (self._noiseFunc(self._noise, array) + 1) * 0.5
        
    def __del__(self):
        _lib.TCOD_random_delete(self._random)
        _lib.TCOD_noise_delete(self._noise)
    
__all__ = ['Noise']
