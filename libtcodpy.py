"""Deprecated module alias for tcod.libtcodpy, use 'import tcod as libtcodpy' instead."""

import warnings

from tcod.libtcodpy import *  # noqa: F403
from tcod.libtcodpy import __getattr__  # noqa: F401

warnings.warn(
    "'import tcod as libtcodpy' is preferred.",
    DeprecationWarning,
    stacklevel=2,
)
