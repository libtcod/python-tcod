#!/usr/bin/env python3

import os
import sys

import platform

from cffi import FFI

BITSIZE, LINKAGE = platform.architecture()

def _get_library_dirs_crossplatform():
    if 'win32' in sys.platform:
        return ['src/lib/win32/']
    elif 'linux' in sys.platform:
        if BITSIZE == '32bit':
            return ['src/lib/linux32/']
        elif BITSIZE == '64bit':
            return ['src/lib/linux64/']
    elif 'darwin' in sys.platform:
        return ['src/lib/darwin/']
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, BITSIZE, LINKAGE))

def _get_libraries_crossplatform():
    return ['SDL', 'OpenGL32']
    if sys.platform == 'win32':
        return ['libtcod-VS']
    elif 'linux' in sys.platform:
        return ['tcod']
    elif 'darwin' in sys.platform:
        return ['tcod']
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, BITSIZE, LINKAGE))


def walk_sources(directory):
    for path, dirs, files in os.walk(directory):
        for source in files:
            if not source.endswith('.c'):
                continue
            yield os.path.join(path, source)

def find_sources(directory):
    return [os.path.join(directory, source)
            for source in os.listdir(directory)
            if source.endswith('.c')]

module_name = 'tcod._libtcod'
include_dirs = ['Release/tcod/',
                'dependencies/libtcod-1.5.1/include/',
                'dependencies/libtcod-1.5.1/src/png/']
extra_compile_args = []
sources = []

sources += [file for file in walk_sources('dependencies/libtcod-1.5.1/src')
            if 'sys_sfml_c' not in file]
sources += find_sources('dependencies/zlib-1.2.8/')

libraries = ['SDL']
library_dirs = _get_library_dirs_crossplatform()
define_macros = [('LIBTCOD_EXPORTS', None)]

with open('src/tdl_source.c', 'r') as file_source:
    source = file_source.read()

if sys.platform == 'win32':
    libraries += ['User32', 'OpenGL32']

if 'linux' in sys.platform:
    libraries += ['GL']

if sys.platform == 'darwin':
    extra_compile_args += ['-framework', 'OpenGL']

    library_dirs += ['src/SDL.framework/Versions/A/']

# included SDL headers are for whatever OS's don't easily come with them
if sys.platform in ['win32', 'darwin']:
    include_dirs += ['dependencies/SDL-1.2.15/include', 'dependencies/zlib-1.2.8/']

    if BITSIZE == '32bit':
        library_dirs += [os.path.realpath('dependencies/SDL-1.2.15/lib/x86')]
    else:
        library_dirs += [os.path.realpath('dependencies/SDL-1.2.15/lib/x64')]

ffi = FFI()
with open('src/libtcod_cdef.h', 'r') as file_cdef:
    ffi.cdef(file_cdef.read())
ffi.set_source(
    module_name, source,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    sources=sources,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
)

if __name__ == "__main__":
    ffi.compile()
