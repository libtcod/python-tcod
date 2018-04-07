"""This module just an alias for tcod"""
import warnings
warnings.warn("`import tcod` is preferred.", DeprecationWarning, stacklevel=2)
from tcod import *
