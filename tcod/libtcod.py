"""This module handles loading of the libtcod cffi API.
"""
from __future__ import absolute_import as _

import sys as _sys
import os as _os

import platform as _platform


from tcod import __path__

if _sys.platform == 'win32':
    # add Windows dll's to PATH
    _bits, _linkage = _platform.architecture()
    _os.environ['PATH'] = '%s;%s' % (
        _os.path.join(__path__[0], 'x86' if _bits == '32bit' else 'x64'),
        _os.environ['PATH'],
        )

from tcod.constants import *

NOISE_DEFAULT_HURST = 0.5
NOISE_DEFAULT_LACUNARITY = 2.0

def FOV_PERMISSIVE(p) :
    return FOV_PERMISSIVE_0+p

def BKGND_ALPHA(a):
    return BKGND_ALPH | (int(a * 255) << 8)

def BKGND_ADDALPHA(a):
    return BKGND_ADDA | (int(a * 255) << 8)

if not _os.environ.get('READTHEDOCS'):
    # Allows an import without building the cffi module first.
    from tcod._libtcod import lib, ffi
