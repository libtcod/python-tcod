#!/usr/bin/env python3

import sys

import platform

from cffi import FFI

module_name = 'tcod._libtcod'
if platform.python_implementation() == 'CPython':
    module_name += '_cp%i%i' % sys.version_info[:2]
    if platform.architecture()[0] == '64bit':
        module_name += '_x64'

def _get_library_dirs_crossplatform():
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return 'tcod/lib/win32/'
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return 'tcod/lib/linux32/'
        elif bits == '64bit':
            return 'tcod/lib/linux64/'
    elif 'darwin' in sys.platform:
        return 'tcod/lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))

def _get_libraries_crossplatform():
    bits, linkage = platform.architecture()
    if sys.platform  in ['win32', 'win64']:
        return ['libtcod-VS']
    elif 'linux' in sys.platform:
        return ['libtcod']
    elif 'darwin' in sys.platform:
        return ['libtcod']
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))
    
ffi = FFI()
ffi.cdef(open('tcod/libtcod_cdef.h', 'r').read())
ffi.set_source(module_name, open('tcod/tdl_source.c', 'r').read(),
include_dirs=['tcod/include/libtcod-1.5', 'Release/tcod/'],
library_dirs=[_get_library_dirs_crossplatform()],
libraries=_get_libraries_crossplatform(),
extra_compile_args=['/O2', '/W3'])

if __name__ == "__main__":
    ffi.compile()
