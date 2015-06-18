
import sys
import os

import platform

from . import __path__

try: # decide how files are unpacked depending on if we have the pkg_resources module
    from pkg_resources import resource_filename
    def _unpackfile(filename):
        return resource_filename(__name__, filename)
except ImportError:
    #from tdl import __path__
    def _unpackfile(filename):
        return os.path.abspath(os.path.join(__path__[0], filename))

def _unpackFramework(framework, path):
    """get framework.tar file, remove ".tar" and add path"""
    return os.path.abspath(os.path.join(_unpackfile(framework)[:-4], path))

def _loadDLL(dll):
    """shorter version of file unpacking and linking"""
    return ffi.LoadLibrary(_unpackfile(dll))

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

os.environ['PATH'] += ';' + os.path.join(__path__[0],
                                         _get_library_crossplatform())

from ._libtcod import ffi, lib

_ffi = ffi
_lib = lib
