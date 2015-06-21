
import sys
import os

import platform

from . import __path__

def _get_library_crossplatform():
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return 'lib/win32/'
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return 'lib/linux32/'
        elif bits == '64bit':
            return 'lib/linux64/'
    elif 'darwin' in sys.platform:
        return 'lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))

def _import_library_functions(lib):
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            g[name[5:]] = getattr(lib, name)
    
os.environ['PATH'] += ';' + os.path.join(__path__[0],
                                         _get_library_crossplatform())

try:
    import _libtcod
except ImportError:
    # get implementation specific version of _libtcod.pyd
    import importlib
    module_name = '._libtcod'
    if platform.python_implementation() == 'CPython':
        module_name += '_cp%i%i' % sys.version_info[:2]

    _libtcod = importlib.import_module(module_name, 'tdl')

_ffi = ffi = _libtcod.ffi
_lib = lib = _libtcod.lib
_import_library_functions(lib)
