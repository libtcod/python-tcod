"""
    Random module docs.
"""

from __future__ import absolute_import as _

import array
import time

from tcod.libtcod import ffi, lib
from tcod.libtcod import RNG_MT as MERSENNE_TWISTER
from tcod.libtcod import RNG_CMWC as COMPLEMENTARY_MULTIPLY_WITH_CARRY


class Random(object):
    """The libtcod random number generator.
    .. versionadded:: 2.0

    If all you need is a random number generator then it's recommended
    that you use the :any:`random` module from the Python standard library.

    If ``seed`` is None then a random seed will be generated.

    Args:
        algorithm (int): The algorithm to use.
        seed (Optional[Hashable]):
            Could be a 32-bit integer, but any hashable object is accepted.
    """
    def __init__(self, algorithm, seed=None):
        """Create a new instance using this algorithm and seed."""
        if seed is None:
            seed = time.time() + time.clock()
        self.cdata = ffi.gc(
            ffi.cast('mersenne_data_t*',
                     lib.TCOD_random_new_from_seed(algorithm,
                                                   hash(seed) % (1 << 32))),
            lib.TCOD_random_delete)

    @classmethod
    def _new_from_cdata(cls, cdata):
        """Return a new instance encapsulating this cdata."""
        self = object.__new__(cls)
        self.cdata = cdata
        return self

    def randint(self, low, high):
        """Return a random integer within the linear range: low <= n <= high.

        Args:
            low (int): The lower bound of the random range.
            high (int): The upper bound of the random range.

        Returns:
            int: A random integer.
        """
        return lib.TCOD_random_get_i(self.cdata, low, high)

    def uniform(self, low, high):
        """Return a random floating number in the range: low <= n <= high.

        Args:
            low (int): The lower bound of the random range.
            high (int): The upper bound of the random range.

        Returns:
            float: A random float.
        """
        return lib.TCOD_random_get_double(self.cdata, low, high)

    def guass(self, mu, sigma):
        """Return a random number using Gaussian distribution.

        Args:
            mu (float): The median returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random float.
        """
        return lib.TCOD_random_get_gaussian_double(self.cdata, mu, sigma)

    def inverse_guass(self, mu, sigma):
        """Return a random Gaussian number using the Box-Muller transform.

        Args:
            mu (float): The median returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random float.
        """
        return lib.TCOD_random_get_gaussian_double_inv(self.cdata, mu, sigma)

    def __getstate__(self):
        """Pack the self.cdata attribute into a portable state."""
        return {'cdata': (self.cdata.algo, self.cdata.distribution,
                          list(self.cdata.mt), self.cdata.cur_mt,
                          list(self.cdata.Q), self.cdata.c, self.cdata.cur)}

    def __setstate__(self, state):
        """Create a new cdata object with the stored paramaters."""
        self.cdata = ffi.new('mersenne_data_t*', state['cdata'])
