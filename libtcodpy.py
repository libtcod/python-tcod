"""This module just an alias for tcod"""
import warnings

warnings.warn(
    "'import tcod as libtcodpy' is preferred.",
    DeprecationWarning,
    stacklevel=2,
)
from tcod import *  # noqa: F4
