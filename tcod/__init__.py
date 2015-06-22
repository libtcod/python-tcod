"""
    This module provides a simple CFFI API to libtcod.
    
    This port has large partial support for libtcod's C functions.
    Use tcod/libtcod_cdef.h in the source distribution to see specially what
    functions were exported and what new functions have been added by TDL.
    
    The ffi and lib variables should be familiar to anyone that has used CFFI
    before, otherwise it's time to read up on how they work:
    https://cffi.readthedocs.org/en/latest/using.html
    
    Bring any issues or requests to GitHub:
    https://github.com/HexDecimal/libtcod-cffi
"""
import sys as _sys
import os as _os

import platform as _platform

def _get_lib_path_crossplatform():
    '''Locate the right DLL path for this OS'''
    bits, linkage = _platform.architecture()
    if 'win32' in _sys.platform:
        return 'lib/win32/'
    elif 'linux' in _sys.platform:
        if bits == '32bit':
            return 'lib/linux32/'
        elif bits == '64bit':
            return 'lib/linux64/'
    elif 'darwin' in _sys.platform:
        return 'lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (_sys.platform, bits, linkage))

def _import_library_functions(lib):
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            g[name[5:]] = getattr(lib, name)
        elif name[:4] == 'TCOD': # short constant names
            g[name[4:]] = getattr(lib, name)
    
# add dll's to PATH
_os.environ['PATH'] += ';' + _os.path.join(__path__[0],
                                           _get_lib_path_crossplatform())

# import the right .pyd file for this Python implementation
try:
    import _libtcod # PyPy
except ImportError:
    # get implementation specific version of _libtcod.pyd
    import importlib as _importlib
    _module_name = '._libtcod'
    if _platform.python_implementation() == 'CPython':
        _module_name += '_cp%i%i' % _sys.version_info[:2]
        if _platform.architecture()[0] == '64bit':
            _module_name += '_x64'

    _libtcod = _importlib.import_module(_module_name, 'tcod')

ffi = _libtcod.ffi
lib = _libtcod.lib

# make a fancy function importer, then never use it!
#_import_library_functions(lib)

__all__ = [name for name in list(globals()) if name[0] != '_']
