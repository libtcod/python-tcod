
import sys as _sys
import os as _os

import platform

def _get_library_crossplatform():
    bits, linkage = platform.architecture()
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
    
_os.environ['PATH'] += ';' + _os.path.join(__path__[0],
                                           _get_library_crossplatform())

try:
    import _libtcod
except ImportError:
    # get implementation specific version of _libtcod.pyd
    import importlib
    module_name = '._libtcod'
    if platform.python_implementation() == 'CPython':
        module_name += '_cp%i%i' % _sys.version_info[:2]

    _libtcod = importlib.import_module(module_name, 'tdl')

ffi = _libtcod.ffi
lib = _libtcod.lib
_import_library_functions(lib)

__all__ = [name for name in globals() if name[0] != '_']