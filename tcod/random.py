
from __future__ import absolute_import as _

from tcod.tcod import _CDataWrapper
from tcod.libtcod import ffi, lib


class Random(_CDataWrapper):
    """
    .. versionadded:: 2.0

    If all you need is a random number generator then it's recommended
    that you use the :any:`random` module from the Python standard library.

    Args:
        seed (Hashable): The RNG seed.  Should be a 32-bit integer, but any
                         hashable object is accepted.
        algorithm (int): The algorithm to use.
    """
    def __init__(self, *args, **kargs):
        super(Random, self).__init__(*args, **kargs)
        if not self.cdata:
            self._init(*args, **kargs)

    def _init(self, seed, algorithm):
        self.cdata = ffi.gc(lib.TCOD_random_new_from_seed(algorithm,
                                                          hash(seed)),
                            lib.TCOD_random_delete)


    def random_int(self, low, high, mean=None):
        """Return a random integer from a linear or triangular range.

        Args:
            low (int): The lower bound of the random range, inclusive.
            high (int): The upper bound of the random range, inclusive.
            mean (Optional[int]): The mean return value, or None.

        Returns:
            int: A random number from the given range: low <= n <= high.
        """
        lib.TCOD_random_set_distribution(self.cdata,
                                         lib.TCOD_DISTRIBUTION_LINEAR)
        if mean is None:
            return lib.TCOD_random_get_int(self.cdata, low, high)
        return lib.TCOD_random_get_int_mean(self.cdata, low, high, mean)


    def random_float(self, low, high, mean=None):
        """Return a random float from a linear or triangular range.

        Args:
            low (float): The lower bound of the random range.
            high (float): The upper bound of the random range.
            mean (Optional[float]): The mean return value, or None.

        Returns:
            float: A random number from the given range: low <= n <= high.
        """
        lib.TCOD_random_set_distribution(self.cdata,
                                         lib.TCOD_DISTRIBUTION_LINEAR)
        if mean is None:
            return lib.TCOD_random_get_double(self.cdata, low, high)
        return lib.TCOD_random_get_double_mean(self.cdata, low, high, mean)

    def gaussian(self, mu, sigma):
        """Return a number from a random gaussian distribution.

        Args:
            mu (float): The mean returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random number derived from the given parameters.
        """
        lib.TCOD_random_set_distribution(self.cdata,
                                         lib.TCOD_DISTRIBUTION_GAUSSIAN)
        return lib.TCOD_random_get_double(self.cdata, mu, sigma)

    def inverse_gaussian(self, mu, sigma):
        """Return a number from a random inverse gaussian distribution.

        Args:
            mu (float): The mean returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random number derived from the given parameters.
        """
        lib.TCOD_random_set_distribution(self.cdata,
            lib.TCOD_DISTRIBUTION_GAUSSIAN_INVERSE)
        return lib.TCOD_random_get_double(self.cdata, mu, sigma)

    def gaussian_range(self, low, high, mean=None):
        """Return a random gaussian number clamped to a range.

        When ``mean`` is None it will be automatically determined
        from the ``low`` and ``high`` parameters.

        Args:
            low (float): The lower bound of the random range.
            high (float): The upper bound of the random range.
            mean (Optional[float]): The mean return value, or None.

        Returns:
            float: A clamped gaussian number.
        """
        lib.TCOD_random_set_distribution(self.cdata,
            lib.TCOD_DISTRIBUTION_GAUSSIAN_RANGE)
        if mean is None:
            return lib.TCOD_random_get_double(self.cdata, low, high)
        return lib.TCOD_random_get_double_mean(self.cdata, low, high, mean)

    def inverse_gaussian_range(self, low, high, mean=None):
        """Return a random inverted gaussian number clamped to a range.

        When ``mean`` is None it will be automatically determined
        from the ``low`` and ``high`` parameters.

        Args:
            low (float): The lower bound of the random range.
            high (float): The upper bound of the random range.
            mean (Optional[float]): The mean return value, or None.

        Returns:
            float: A clamped inverse gaussian number.
        """
        lib.TCOD_random_set_distribution(self.cdata,
            lib.TCOD_DISTRIBUTION_GAUSSIAN_RANGE_INVERSE)
        if mean is None:
            return lib.TCOD_random_get_double(self.cdata, low, high)
        return lib.TCOD_random_get_double_mean(self.cdata, low, high, mean)
