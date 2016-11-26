#!/usr/bin/env python3

import os
import sys

import platform
from pycparser import c_parser, c_ast, parse_file, c_generator
from cffi import FFI

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
                'Release/tcod/',
                'libtcod/include/',
                'libtcod/src/png/',
                'libtcod/src/zlib/',
                '/usr/include/SDL2/',
                ]

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
            self.ast.ext.remove(node) # remove wchar_t placeholder
        else:
            self.generic_visit(node)
            if node.name in self.typedefs:
                print('%s redefined' % node.name)
            self.typedefs.append(node.name)

    def visit_EnumeratorList(self, node):
        """Replace enumerator expressions with stubs."""
        for type, enum in node.children():
            if enum.value is None:
                pass
            elif isinstance(enum.value, c_ast.BinaryOp):
                enum.value = c_ast.Constant('int', '...')
            else:
                enum.value = c_ast.Constant(enum.value.type, '...')

    def visit_FuncDecl(self, node):
        pass

    def visit_FuncDef(self, node):
        """Remvoe function definitions."""
        self.ast.ext.remove(node)

def get_cdef():
    generator = c_generator.CGenerator()
    return generator.visit(get_ast())

def get_ast():
    if 'win32' in sys.platform:
        sdl_include = r'-Idependencies/SDL2-2.0.4/include'
    else:
        sdl_include = r'-I/usr/include/SDL2'
    ast = parse_file(filename='tcod/tcod.h', use_cpp=True,
                     cpp_args=[r'-Idependencies/fake_libc_include',
                               r'-Ilibtcod/include',
                               sdl_include,
                               r'-DDECLSPEC=',
                               r'-DSDLCALL=',
                               r'-DTCODLIB_API=',
                               r'-DTCOD_NO_MACOSX_SDL_MAIN=',
                               r'-DTCOD_SDL2=',
                               r'-DNO_OPENGL',
                               r'-DSDL_FORCE_INLINE=',
                               r'-U__GNUC__',
                               r'-D_SDL_thread_h',
                               r'-DDOXYGEN_SHOULD_IGNORE_THIS',
                               r'-DSDL_MAIN_HANDLED',
                               ])
    return CustomPostParser().parse(ast)

def resolve_ast(ast, consts):
    if isinstance(ast, c_ast.Constant):
        return int(ast.value)
    elif isinstance(ast, c_ast.ID):
        return consts[ast.name]
    elif isinstance(ast, c_ast.BinaryOp):
        return resolve_ast(ast.left, consts) | resolve_ast(ast.right, consts)
    else:
        raise RuntimeError('Unexpected ast node: %r' % ast)


ffi = FFI()
try:
    ffi.cdef(get_cdef())
except Exception as exc:
    #print(dir(exc.args[1]))
    #print(exc.args[1].children()[0][1].name)
    #print(exc.args[1].show())
    raise
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
