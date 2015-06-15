
import os as _os
from ._tcod_cffi import ffi

try: # decide how files are unpacked depending on if we have the pkg_resources module
    from pkg_resources import resource_filename
    def _unpackfile(filename):
        return resource_filename(__name__, filename)
except ImportError:
    from tdl import __path__
    def _unpackfile(filename):
        return os.path.abspath(os.path.join(__path__[0], filename))

def _unpackFramework(framework, path):
    """get framework.tar file, remove ".tar" and add path"""
    return os.path.abspath(os.path.join(_unpackfile(framework)[:-4], path))

def _loadDLL(dll):
    """shorter version of file unpacking and linking"""
    return ffi.LoadLibrary(_unpackfile(dll))


ffi.dlopen(_unpackfile('lib/win32/zlib1.dll'))
ffi.dlopen(_unpackfile('lib/win32/SDL.dll'))
lib = ffi.dlopen(_unpackfile('lib/win32/libtcod-mingw.dll'))

_ffi = ffi
_lib = lib