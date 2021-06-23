"""Ports of the libtcod random number generator.

Usually it's recommend to the Python's standard library `random` module
instead of this one.

However, you will need to use these generators to get deterministic results
from the :any:`Noise` and :any:`BSP` classes.
"""
import os
import random
import warnings
from typing import Any, Hashable, Optional

import tcod.constants
from tcod.loader import ffi, lib

MERSENNE_TWISTER = tcod.constants.RNG_MT
COMPLEMENTARY_MULTIPLY_WITH_CARRY = tcod.constants.RNG_CMWC
MULTIPLY_WITH_CARRY = tcod.constants.RNG_CMWC


class Random(object):
    """The libtcod random number generator.

    `algorithm` defaults to Mersenne Twister, it can be one of:

    * tcod.random.MERSENNE_TWISTER
    * tcod.random.MULTIPLY_WITH_CARRY

    `seed` is a 32-bit number or any Python hashable object like a string.
    Using the same seed will cause the generator to return deterministic
    values.  The default `seed` of None will generate a random seed instead.

    Attributes:
        random_c (CData): A cffi pointer to a TCOD_random_t object.

    .. warning::
        A non-integer seed is only deterministic if the environment variable
        ``PYTHONHASHSEED`` is set.  In the future this function will only
        accept `int`'s as a seed.

    .. versionchanged:: 9.1
        Added `tcod.random.MULTIPLY_WITH_CARRY` constant.
        `algorithm` parameter now defaults to `tcod.random.MERSENNE_TWISTER`.
    """

    def __init__(
        self,
        algorithm: int = MERSENNE_TWISTER,
        seed: Optional[Hashable] = None,
    ):
        """Create a new instance using this algorithm and seed."""
        if seed is None:
            seed = random.getrandbits(32)
        elif not isinstance(seed, int):
            warnings.warn(
                "In the future this class will only accept integer seeds.",
                DeprecationWarning,
                stacklevel=2,
            )
            if __debug__ and "PYTHONHASHSEED" not in os.environ:
                warnings.warn(
                    "Python's hash algorithm is not configured to be"
                    " deterministic so this non-integer seed will not be"
                    " deterministic."
                    "\nYou should do one of the following to fix this error:"
                    "\n* Use an integer as a seed instead (recommended.)"
                    "\n* Set the PYTHONHASHSEED environment variable before"
                    " starting Python.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            seed = hash(seed)

        self.random_c = ffi.gc(
            ffi.cast(
                "mersenne_data_t*",
                lib.TCOD_random_new_from_seed(algorithm, seed & 0xFFFFFFFF),
            ),
            lib.TCOD_random_delete,
        )

    @classmethod
    def _new_from_cdata(cls, cdata: Any) -> "Random":
        """Return a new instance encapsulating this cdata."""
        self = object.__new__(cls)  # type: "Random"
        self.random_c = cdata
        return self

    def randint(self, low: int, high: int) -> int:
        """Return a random integer within the linear range: low <= n <= high.

        Args:
            low (int): The lower bound of the random range.
            high (int): The upper bound of the random range.

        Returns:
            int: A random integer.
        """
        return int(lib.TCOD_random_get_i(self.random_c, low, high))

    def uniform(self, low: float, high: float) -> float:
        """Return a random floating number in the range: low <= n <= high.

        Args:
            low (float): The lower bound of the random range.
            high (float): The upper bound of the random range.

        Returns:
            float: A random float.
        """
        return float(lib.TCOD_random_get_double(self.random_c, low, high))

    def guass(self, mu: float, sigma: float) -> float:
        """Return a random number using Gaussian distribution.

        Args:
            mu (float): The median returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random float.
        """
        return float(lib.TCOD_random_get_gaussian_double(self.random_c, mu, sigma))

    def inverse_guass(self, mu: float, sigma: float) -> float:
        """Return a random Gaussian number using the Box-Muller transform.

        Args:
            mu (float): The median returned value.
            sigma (float): The standard deviation.

        Returns:
            float: A random float.
        """
        return float(lib.TCOD_random_get_gaussian_double_inv(self.random_c, mu, sigma))

    def __getstate__(self) -> Any:
        """Pack the self.random_c attribute into a portable state."""
        state = self.__dict__.copy()
        state["random_c"] = {
            "algo": self.random_c.algo,
            "distribution": self.random_c.distribution,
            "mt": list(self.random_c.mt),
            "cur_mt": self.random_c.cur_mt,
            "Q": list(self.random_c.Q),
            "c": self.random_c.c,
            "cur": self.random_c.cur,
        }
        return state

    def __setstate__(self, state: Any) -> None:
        """Create a new cdata object with the stored paramaters."""
        try:
            cdata = state["random_c"]
        except KeyError:  # old/deprecated format
            cdata = state["cdata"]
            del state["cdata"]
        state["random_c"] = ffi.new("mersenne_data_t*", cdata)
        self.__dict__.update(state)
