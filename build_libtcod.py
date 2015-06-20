#!/usr/bin/env python3

import sys

import platform

from cffi import FFI


def _get_library_crossplatform():
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return 'tdl/lib/win32/'
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return 'tdl/lib/linux32/'
        elif bits == '64bit':
            return 'tdl/lib/linux64/'
    elif 'darwin' in sys.platform:
        return 'tdl/lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))

ffi = FFI()
ffi.cdef(open('tdl/tdl_cdef.h', 'r').read())
ffi.set_source('tdl._libtcod', open('tdl/tdl_source.c', 'r').read(),
include_dirs=['tdl/include/', 'tdl/include/Release/'],
library_dirs=[_get_library_crossplatform()],
libraries=['libtcod-VS'])


if __name__ == "__main__":
    ffi.compile()