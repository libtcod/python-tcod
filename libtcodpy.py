"""Module alias for tcod."""
import warnings

from tcod import *  # noqa: F4
from tcod.libtcodpy import __getattr__  # noqa: F401

warnings.warn(
    "'import tcod as libtcodpy' is preferred.",
    DeprecationWarning,
    stacklevel=2,
)
