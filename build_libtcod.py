#!/usr/bin/env python3

import os
import sys

import glob
import re
from typing import List, Tuple, Any, Dict, Iterator

from cffi import FFI
from pycparser import c_parser, c_ast, parse_file, c_generator

import shutil
import subprocess
import platform
import zipfile

import parse_sdl2

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

# The SDL2 version to parse and export symbols from.
SDL2_PARSE_VERSION = os.environ.get("SDL_VERSION", "2.0.5")
# The SDL2 version to include in binary distributions.
SDL2_BUNDLE_VERSION = os.environ.get("SDL_VERSION", "2.0.9")

CFFI_HEADER = "tcod/cffi.h"
CFFI_EXTRA_CDEFS = "tcod/cdef.h"

BITSIZE, LINKAGE = platform.architecture()


def walk_sources(directory):
    for path, dirs, files in os.walk(directory):
        for source in files:
            if source.endswith(".c") or source.endswith(".cpp"):
                yield os.path.join(path, source)


def find_sources(directory):
    return [
        os.path.join(directory, source)
        for source in os.listdir(directory)
        if source.endswith(".c")
    ]


def get_sdl2_file(version):
    if sys.platform == "win32":
        sdl2_file = "SDL2-devel-%s-VC.zip" % (version,)
    else:
        assert sys.platform == "darwin"
        sdl2_file = "SDL2-%s.dmg" % (version,)
    sdl2_local_file = os.path.join("dependencies", sdl2_file)
    sdl2_remote_file = "https://www.libsdl.org/release/%s" % sdl2_file
    if not os.path.exists(sdl2_local_file):
        print("Downloading %s" % sdl2_remote_file)
        urlretrieve(sdl2_remote_file, sdl2_local_file)
    return sdl2_local_file


def unpack_sdl2(version):
    sdl2_path = "dependencies/SDL2-%s" % (version,)
    if sys.platform == "darwin":
        sdl2_dir = sdl2_path
        sdl2_path += "/SDL2.framework"
    if os.path.exists(sdl2_path):
        return sdl2_path
    sdl2_arc = get_sdl2_file(version)
    print("Extracting %s" % sdl2_arc)
    if sdl2_arc.endswith(".zip"):
        with zipfile.ZipFile(sdl2_arc) as zf:
            zf.extractall("dependencies/")
    else:
        assert sdl2_arc.endswith(".dmg")
        subprocess.check_call(["hdiutil", "mount", sdl2_arc])
        subprocess.check_call(["mkdir", "-p", sdl2_dir])
        subprocess.check_call(
            ["cp", "-r", "/Volumes/SDL2/SDL2.framework", sdl2_dir]
        )
        subprocess.check_call(["hdiutil", "unmount", "/Volumes/SDL2"])
    return sdl2_path


module_name = "tcod._libtcod"
include_dirs = [".", "libtcod/src/vendor/", "libtcod/src/vendor/zlib/"]

extra_parse_args = []
extra_compile_args = []
extra_link_args = []
sources = []

libraries = []
library_dirs = []
define_macros = []

sources += walk_sources("tcod/")
sources += walk_sources("tdl/")
sources += ["libtcod/src/libtcod_c.c"]
sources += ["libtcod/src/libtcod.cpp"]
sources += ["libtcod/src/vendor/glad.c"]
sources += ["libtcod/src/vendor/lodepng.cpp"]
sources += ["libtcod/src/vendor/utf8proc/utf8proc.c"]
sources += glob.glob("libtcod/src/vendor/zlib/*.c")

if sys.platform == "win32":
    libraries += ["User32"]
    define_macros.append(("TCODLIB_API", ""))
    define_macros.append(("_CRT_SECURE_NO_WARNINGS", None))

if sys.platform == "darwin":
    extra_link_args += ["-framework", "SDL2"]
else:
    libraries += ["SDL2"]

# included SDL headers are for whatever OS's don't easily come with them

if sys.platform in ["win32", "darwin"]:
    SDL2_PARSE_PATH = unpack_sdl2(SDL2_PARSE_VERSION)
    SDL2_BUNDLE_PATH = unpack_sdl2(SDL2_BUNDLE_VERSION)
    include_dirs.append("libtcod/src/zlib/")

if sys.platform == "win32":
    SDL2_INCLUDE = os.path.join(SDL2_PARSE_PATH, "include")
elif sys.platform == "darwin":
    SDL2_INCLUDE = os.path.join(SDL2_PARSE_PATH, "Versions/A/Headers")
else:
    match = re.match(
        r".*-I(\S+)",
        subprocess.check_output(
            ["sdl2-config", "--cflags"], universal_newlines=True
        ),
    )
    assert match
    SDL2_INCLUDE, = match.groups()

if sys.platform == "win32":
    include_dirs.append(SDL2_INCLUDE)
    ARCH_MAPPING = {"32bit": "x86", "64bit": "x64"}
    SDL2_LIB_DIR = os.path.join(
        SDL2_BUNDLE_PATH, "lib/", ARCH_MAPPING[BITSIZE]
    )
    library_dirs.append(SDL2_LIB_DIR)
    SDL2_LIB_DEST = os.path.join("tcod", ARCH_MAPPING[BITSIZE])
    if not os.path.exists(SDL2_LIB_DEST):
        os.mkdir(SDL2_LIB_DEST)
    shutil.copy(os.path.join(SDL2_LIB_DIR, "SDL2.dll"), SDL2_LIB_DEST)


def fix_header(filepath):
    """Removes leading whitespace from a MacOS header file.

    This whitespace is causing issues with directives on some platforms.
    """
    with open(filepath, "r+") as f:
        current = f.read()
        fixed = "\n".join(line.strip() for line in current.split("\n"))
        if current == fixed:
            return
        f.seek(0)
        f.truncate()
        f.write(fixed)


if sys.platform == "darwin":
    HEADER_DIR = os.path.join(SDL2_PARSE_PATH, "Headers")
    fix_header(os.path.join(HEADER_DIR, "SDL_assert.h"))
    fix_header(os.path.join(HEADER_DIR, "SDL_config_macosx.h"))
    include_dirs.append(HEADER_DIR)
    extra_link_args += ["-F%s/.." % SDL2_BUNDLE_PATH]
    extra_link_args += ["-rpath", "%s/.." % SDL2_BUNDLE_PATH]
    extra_link_args += ["-rpath", "/usr/local/opt/llvm/lib/"]

if sys.platform not in ["win32", "darwin"]:
    extra_parse_args += (
        subprocess.check_output(
            ["sdl2-config", "--cflags"], universal_newlines=True
        )
        .strip()
        .split()
    )
    extra_compile_args += extra_parse_args
    extra_link_args += (
        subprocess.check_output(
            ["sdl2-config", "--libs"], universal_newlines=True
        )
        .strip()
        .split()
    )


class CustomPostParser(c_ast.NodeVisitor):
    def __init__(self):
        self.ast = None
        self.typedefs = []
        self.removeable_typedefs = []
        self.funcdefs = []

    def parse(self, ast):
        self.ast = ast
        self.visit(ast)
        for node in self.funcdefs:
            ast.ext.remove(node)
        for node in self.removeable_typedefs:
            ast.ext.remove(node)
        return ast

    def visit_Typedef(self, node):
        if node.name in ["wchar_t", "size_t"]:
            # remove fake typedef placeholders
            self.removeable_typedefs.append(node)
        else:
            self.generic_visit(node)
            if node.name in self.typedefs:
                print("warning: %s redefined" % node.name)
                self.removeable_typedefs.append(node)
            self.typedefs.append(node.name)

    def visit_EnumeratorList(self, node):
        """Replace enumerator expressions with '...' stubs."""
        for type, enum in node.children():
            if enum.value is None:
                pass
            elif isinstance(enum.value, (c_ast.BinaryOp, c_ast.UnaryOp)):
                enum.value = c_ast.Constant("int", "...")
            elif hasattr(enum.value, "type"):
                enum.value = c_ast.Constant(enum.value.type, "...")

    def visit_ArrayDecl(self, node):
        if not node.dim:
            return
        if isinstance(node.dim, (c_ast.BinaryOp, c_ast.UnaryOp)):
            node.dim = c_ast.Constant("int", "...")

    def visit_Decl(self, node):
        if node.name is None:
            self.generic_visit(node)
        elif (
            node.name
            and "vsprint" in node.name
            or node.name
            in ["SDL_vsscanf", "SDL_vsnprintf", "SDL_LogMessageV", "alloca"]
        ):
            # exclude va_list related functions
            self.ast.ext.remove(node)
        elif node.name in ["screen"]:
            # exclude outdated 'extern SDL_Surface* screen;' line
            self.ast.ext.remove(node)
        else:
            self.generic_visit(node)

    def visit_FuncDef(self, node):
        """Exclude function definitions.  Should be declarations only."""
        self.funcdefs.append(node)


def get_cdef():
    generator = c_generator.CGenerator()
    cdef = generator.visit(get_ast())
    cdef = re.sub(
        pattern=r"typedef int (ptrdiff_t);",
        repl=r"typedef int... \1;",
        string=cdef,
    )
    return cdef


def get_ast():
    global extra_parse_args
    if "win32" in sys.platform:
        extra_parse_args += [r"-I%s/include" % SDL2_PARSE_PATH]
    if "darwin" in sys.platform:
        extra_parse_args += [r"-I%s/Headers" % SDL2_PARSE_PATH]

    ast = parse_file(
        filename=CFFI_HEADER,
        use_cpp=True,
        cpp_args=[
            r"-Idependencies/fake_libc_include",
            r"-DDECLSPEC=",
            r"-DSDLCALL=",
            r"-DTCODLIB_API=",
            r"-DSDL_FORCE_INLINE=",
            r"-U__GNUC__",
            r"-D_SDL_thread_h",
            r"-DDOXYGEN_SHOULD_IGNORE_THIS",
            r"-DMAC_OS_X_VERSION_MIN_REQUIRED=1060",
            r"-D__attribute__(x)=",
            r"-D_PSTDINT_H_INCLUDED",
        ]
        + extra_parse_args,
    )
    ast = CustomPostParser().parse(ast)
    return ast


# Can force the use of OpenMP with this variable.
try:
    USE_OPENMP = eval(os.environ.get("USE_OPENMP", "None").title())
except Exception:
    USE_OPENMP = None

tdl_build = os.environ.get("TDL_BUILD", "RELEASE").upper()

MSVC_CFLAGS = {"DEBUG": ["/Od"], "RELEASE": ["/GL", "/O2", "/GS-", "/wd4996"]}
MSVC_LDFLAGS = {"DEBUG": [], "RELEASE": ["/LTCG"]}
GCC_CFLAGS = {
    "DEBUG": ["-Og", "-g", "-fPIC"],
    "RELEASE": ["-flto", "-O3", "-g", "-fPIC", "-Wno-deprecated-declarations"],
}

if sys.platform == "win32" and "--compiler=mingw32" not in sys.argv:
    extra_compile_args.extend(MSVC_CFLAGS[tdl_build])
    extra_link_args.extend(MSVC_LDFLAGS[tdl_build])

    if USE_OPENMP is None:
        USE_OPENMP = sys.version_info[:2] >= (3, 5)

    if USE_OPENMP:
        extra_compile_args.append("/openmp")
else:
    extra_compile_args.extend(GCC_CFLAGS[tdl_build])
    extra_link_args.extend(GCC_CFLAGS[tdl_build])
    if USE_OPENMP is None:
        USE_OPENMP = sys.platform != "darwin"

    if USE_OPENMP:
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

ffi = FFI()
parse_sdl2.add_to_ffi(ffi, SDL2_INCLUDE)
ffi.cdef(get_cdef())
ffi.cdef(open(CFFI_EXTRA_CDEFS, "r").read())
ffi.set_source(
    module_name,
    "#include <tcod/cffi.h>\n#include <SDL.h>",
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    sources=sources,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

CONSTANT_MODULE_HEADER = '''"""
Constants from the libtcod C API.

This module is auto-generated by `build_libtcod.py`.
"""
from tcod.color import Color

'''

EVENT_CONSTANT_MODULE_HEADER = '''"""
Event constants from SDL's C API.

This module is auto-generated by `build_libtcod.py`.
"""
'''


def find_sdl_attrs(prefix: str) -> Iterator[Tuple[str, Any]]:
    """Return names and values from `tcod.lib`.

    `prefix` is used to filter out which names to copy.
    """
    from tcod._libtcod import lib

    if prefix.startswith("SDL_"):
        name_starts_at = 4
    elif prefix.startswith("SDL"):
        name_starts_at = 3
    else:
        name_starts_at = 0
    for attr in dir(lib):
        if attr.startswith(prefix):
            yield attr[name_starts_at:], getattr(lib, attr)


def parse_sdl_attrs(prefix: str, all_names: List[str]) -> Tuple[str, str]:
    """Return the name/value pairs, and the final dictionary string for the
    library attributes with `prefix`.

    Append matching names to the `all_names` list.
    """
    names = []
    lookup = []
    for name, value in sorted(
        find_sdl_attrs(prefix), key=lambda item: item[1]
    ):
        all_names.append(name)
        names.append("%s = %s" % (name, value))
        lookup.append('%s: "%s"' % (value, name))
    names = "\n".join(names)
    lookup = "{\n    %s,\n}" % (",\n    ".join(lookup),)
    return names, lookup


def write_library_constants():
    """Write libtcod constants into the tcod.constants module."""
    from tcod._libtcod import lib, ffi
    import tcod.color

    with open("tcod/constants.py", "w") as f:
        all_names = []
        f.write(CONSTANT_MODULE_HEADER)
        for name in dir(lib):
            value = getattr(lib, name)
            if name[:5] == "TCOD_":
                if name.isupper():  # const names
                    f.write("%s = %r\n" % (name[5:], value))
                    all_names.append(name[5:])
            elif name.startswith("FOV"):  # fov const names
                f.write("%s = %r\n" % (name, value))
                all_names.append(name)
            elif name[:6] == "TCODK_":  # key name
                f.write("KEY_%s = %r\n" % (name[6:], value))
                all_names.append("KEY_%s" % name[6:])

        f.write("\n# --- colors ---\n")
        for name in dir(lib):
            if name[:5] != "TCOD_":
                continue
            value = getattr(lib, name)
            if not isinstance(value, ffi.CData):
                continue
            if ffi.typeof(value) != ffi.typeof("TCOD_color_t"):
                continue
            color = tcod.color.Color._new_from_cdata(value)
            f.write("%s = %r\n" % (name[5:], color))
            all_names.append(name[5:])

        all_names = ",\n    ".join('"%s"' % name for name in all_names)
        f.write("\n__all__ = [\n    %s,\n]\n" % (all_names,))

    with open("tcod/event_constants.py", "w") as f:
        all_names = []
        f.write(EVENT_CONSTANT_MODULE_HEADER)
        f.write("# --- SDL scancodes ---\n")
        f.write(
            "%s\n_REVERSE_SCANCODE_TABLE = %s\n"
            % parse_sdl_attrs("SDL_SCANCODE", all_names)
        )

        f.write("\n# --- SDL keyboard symbols ---\n")
        f.write(
            "%s\n_REVERSE_SYM_TABLE = %s\n"
            % parse_sdl_attrs("SDLK", all_names)
        )

        f.write("\n# --- SDL keyboard modifiers ---\n")
        f.write(
            "%s\n_REVERSE_MOD_TABLE = %s\n"
            % parse_sdl_attrs("KMOD", all_names)
        )

        f.write("\n# --- SDL wheel ---\n")
        f.write(
            "%s\n_REVERSE_WHEEL_TABLE = %s\n"
            % parse_sdl_attrs("SDL_MOUSEWHEEL", all_names)
        )
        all_names = ",\n    ".join('"%s"' % name for name in all_names)
        f.write("\n__all__ = [\n    %s,\n]\n" % (all_names,))


if __name__ == "__main__":
    write_library_constants()
