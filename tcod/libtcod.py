
import os as _os
import sys as _sys

import ctypes as _ctypes
import platform as _platform

from . import __path__

# add Windows dll's to PATH
if 'win32' in _sys.platform:
    _bits, _linkage = _platform.architecture()
    _os.environ['PATH'] += (';' + \
        _os.path.join(__path__[0], 'x86/' if _bits == '32bit' else 'x64'))

from . import _libtcod

_ffi = ffi = _libtcod.ffi
_lib = lib = _libtcod.lib

def _unpack_char_p(char_p):
    if char_p == _ffi.NULL:
        return ''
    return ffi.string(char_p).decode()

def _int(int_or_str):
    'return an integer where a single character string may be expected'
    if isinstance(int_or_str, str):
        return ord(int_or_str)
    if isinstance(int_or_str, bytes):
        return int_or_str[0]
    return int(int_or_str)

if _sys.version_info[0] == 2: # Python 2
    def _str(string):
        if isinstance(string, unicode):
            return string.encode()
        return string

    def _unicode(string):
        if not isinstance(string, unicode):
            return string.decode()
        return string

else: # Python 3
    def _str(string):
        if isinstance(string, str):
            return string.encode()
        return string

    def _unicode(string):
        if isinstance(string, bytes):
            return string.decode()
        return string
