#!/usr/bin/env python3

import os
import sys

from cffi import FFI
import subprocess
import platform
from pycparser import c_parser, c_ast, parse_file, c_generator

BITSIZE, LINKAGE = platform.architecture()

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
include_dirs = [
                'tcod/',
                'libtcod/include/',
                'libtcod/src/png/',
                'libtcod/src/zlib/',
                '/usr/include/SDL2/',
                ]

extra_parse_args = []
extra_compile_args = []
extra_link_args = []
sources = []

sources += [file for file in walk_sources('libtcod/src')
            if 'sys_sfml_c' not in file
            and 'sdl12' not in file
            ]

libraries = []
library_dirs = []
define_macros = [('LIBTCOD_EXPORTS', None),
                 ('TCOD_SDL2', None),
                 ('NO_OPENGL', None),
                 ('TCOD_NO_MACOSX_SDL_MAIN', None),
                 ('_CRT_SECURE_NO_WARNINGS', None),
                 ]

sources += find_sources('tcod/')

if sys.platform == 'win32':
    libraries += ['User32', 'OpenGL32']

if 'linux' in sys.platform:
    libraries += ['GL']

if sys.platform == 'darwin':
    extra_link_args += ['-framework', 'OpenGL']

libraries += ['SDL2']

# included SDL headers are for whatever OS's don't easily come with them

if sys.platform in ['win32', 'darwin']:
    include_dirs += ['dependencies/SDL2-2.0.4/include']

    if BITSIZE == '32bit':
        library_dirs += [os.path.realpath('dependencies/SDL2-2.0.4/lib/x86')]
    else:
        library_dirs += [os.path.realpath('dependencies/SDL2-2.0.4/lib/x64')]

if sys.platform in ['win32', 'darwin']:
    include_dirs += ['libtcod/src/zlib/']

if sys.platform != 'win32':
    extra_parse_args += subprocess.check_output(['sdl2-config', '--cflags'],
                                              universal_newlines=True
                                              ).strip().split()
    extra_compile_args += extra_parse_args
    extra_link_args += subprocess.check_output(['sdl2-config', '--libs'],
                                               universal_newlines=True
                                               ).strip().split()

class CustomPostParser(c_ast.NodeVisitor):

    def __init__(self):
        self.ast = None
        self.typedefs = None

    def parse(self, ast):
        self.ast = ast
        self.typedefs = []
        self.visit(ast)
        return ast

    def visit_Typedef(self, node):
        start_node = node
        if node.name in ['wchar_t', 'size_t']:
            # remove fake typedef placeholders
            self.ast.ext.remove(node)
        else:
            self.generic_visit(node)
            if node.name in self.typedefs:
                print('warning: %s redefined' % node.name)
                self.ast.ext.remove(node)
            self.typedefs.append(node.name)

    def visit_EnumeratorList(self, node):
        """Replace enumerator expressions with '...' stubs."""
        for type, enum in node.children():
            if enum.value is None:
                pass
            elif isinstance(enum.value, (c_ast.BinaryOp, c_ast.UnaryOp)):
                enum.value = c_ast.Constant('int', '...')
            elif hasattr(enum.value, 'type'):
                enum.value = c_ast.Constant(enum.value.type, '...')

    def visit_Decl(self, node):
        if node.name is None:
            self.generic_visit(node)
        elif (node.name and 'vsprint' in node.name or
              node.name in ['SDL_vsscanf',
                            'SDL_vsnprintf',
                            'SDL_LogMessageV']):
            # exclude va_list related functions
            self.ast.ext.remove(node)
        elif node.name in ['screen']:
            # exclude outdated 'extern SDL_Surface* screen;' line
            self.ast.ext.remove(node)
        else:
            self.generic_visit(node)

    def visit_FuncDef(self, node):
        """Exclude function definitions.  Should be declarations only."""
        self.ast.ext.remove(node)

def get_cdef():
    generator = c_generator.CGenerator()
    return generator.visit(get_ast())

def get_ast():
    global extra_parse_args
    if 'win32' in sys.platform:
        extra_parse_args += [r'-Idependencies/SDL2-2.0.4/include']
    ast = parse_file(filename='tcod/tcod.h', use_cpp=True,
                     cpp_args=[r'-Idependencies/fake_libc_include',
                               r'-Ilibtcod/include',
                               r'-DDECLSPEC=',
                               r'-DSDLCALL=',
                               r'-DTCODLIB_API=',
                               r'-DTCOD_NO_MACOSX_SDL_MAIN=',
                               r'-DTCOD_SDL2=',
                               r'-DNO_OPENGL',
                               r'-DSDL_FORCE_INLINE=',
                               r'-U__GNUC__',
                               r'-D_SDL_assert_h',
                               r'-D_SDL_thread_h',
                               r'-DDOXYGEN_SHOULD_IGNORE_THIS',
                               r'-DMAC_OS_X_VERSION_MIN_REQUIRED=9999',
                               ] + extra_parse_args)
    ast = CustomPostParser().parse(ast)
    return ast

ffi = FFI()
ffi.cdef(get_cdef())
ffi.cdef('''
extern "Python" {
    static bool pycall_parser_new_struct(TCOD_parser_struct_t str,const char *name);
    static bool pycall_parser_new_flag(const char *name);
    static bool pycall_parser_new_property(const char *propname, TCOD_value_type_t type, TCOD_value_t value);
    static bool pycall_parser_end_struct(TCOD_parser_struct_t str, const char *name);
    static void pycall_parser_error(const char *msg);

    static bool _pycall_bsp_callback(TCOD_bsp_t *node, void *userData);

    static float _pycall_path_func( int xFrom, int yFrom, int xTo, int yTo, void *user_data );

    static bool _pycall_line_listener(int x, int y);

    static void _pycall_sdl_hook(void *);
}
''')
ffi.set_source(
    module_name, '#include <tcod.h>',
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    sources=sources,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

if __name__ == "__main__":
    ffi.compile()
