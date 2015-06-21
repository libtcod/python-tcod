#!/usr/bin/env python3

import sys

import platform

from cffi import FFI

module_name = 'libtcod-cffi._libtcod'
if platform.python_implementation() == 'CPython':
    module_name += '_cp%i%i' % sys.version_info[:2]

def _get_library_dirs_crossplatform():
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return 'libtcod-cffi/lib/win32/'
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return 'libtcod-cffi/lib/linux32/'
        elif bits == '64bit':
            return 'libtcod-cffi/lib/linux64/'
    elif 'darwin' in sys.platform:
        return 'libtcod-cffi/lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))

def _get_libraries_crossplatform():
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return ['libtcod-VS']
    elif 'linux' in sys.platform:
        return ['libtcod']
    elif 'darwin' in sys.platform:
        return ['libtcod']
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))
    
ffi = FFI()
ffi.cdef(open('libtcod-cffi/tdl_cdef.h', 'r').read())
ffi.set_source(module_name, open('libtcod-cffi/tdl_source.c', 'r').read(),
include_dirs=['libtcod-cffi/include/', 'Release/libtcod-cffi/'],
library_dirs=[_get_library_dirs_crossplatform()],
libraries=_get_libraries_crossplatform())

if __name__ == "__main__":
    ffi.compile()
