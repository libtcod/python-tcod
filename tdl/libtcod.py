
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

os.environ['PATH'] += ';' + os.path.join(__path__[0],
                                         _get_library_crossplatform())

from ._libtcod import ffi, lib

_ffi = ffi
_lib = lib
