#!/usr/bin/env python3

import os
import sys

import glob
import re
from typing import List, Tuple, Any, Dict, Iterable, Iterator, Set

from cffi import FFI  # type: ignore

import shutil
import subprocess
import platform
import zipfile

import parse_sdl2

try:
    from urllib import urlretrieve  # type: ignore
except ImportError:
    from urllib.request import urlretrieve

# The SDL2 version to parse and export symbols from.
SDL2_PARSE_VERSION = os.environ.get("SDL_VERSION", "2.0.5")
# The SDL2 version to include in binary distributions.
SDL2_BUNDLE_VERSION = os.environ.get("SDL_VERSION", "2.0.14")

HEADER_PARSE_PATHS = ("tcod/", "libtcod/src/libtcod/")
HEADER_PARSE_EXCLUDES = ("gl2_ext_.h", "renderer_gl_internal.h", "event.h")

BITSIZE, LINKAGE = platform.architecture()

# Regular expressions to parse the headers for cffi.
RE_COMMENT = re.compile(r"\s*/\*.*?\*/|\s*//*?$", re.DOTALL | re.MULTILINE)
RE_CPLUSPLUS = re.compile(
    r"#ifdef __cplusplus.*?#endif.*?$", re.DOTALL | re.MULTILINE
)
RE_PREPROCESSOR = re.compile(
    r"(?!#define\s+\w+\s+\d+$)#.*?(?<!\\)$", re.DOTALL | re.MULTILINE
)
RE_INCLUDE = re.compile(r'#include "([^"]*)"')
RE_TAGS = re.compile(
    r"TCODLIB_C?API|TCOD_PUBLIC|TCOD_NODISCARD|TCOD_DEPRECATED_NOMESSAGE"
    r"|(TCOD_DEPRECATED|TCODLIB_FORMAT)\([^)]*\)|__restrict"
)
RE_VAFUNC = re.compile(r".*\(.*va_list.*\);")
RE_INLINE = re.compile(
    r"(^.*?inline.*?\(.*?\))\s*\{.*?\}$", re.DOTALL | re.MULTILINE
)


class ParsedHeader:
    """Header manager class for parsing headers.

    Holds parsed sources and keeps information needed to resolve header order.
    """

    # Class dictionary of all parsed headers.
    all_headers = {}  # type: Dict[str, "ParsedHeader"]

    def __init__(self, path: str) -> None:
        self.path = path = os.path.normpath(path)
        directory = os.path.dirname(path)
        depends = set()
        with open(self.path, "r") as f:
            header = f.read()
        header = RE_COMMENT.sub("", header)
        header = RE_CPLUSPLUS.sub("", header)
        for dependancy in RE_INCLUDE.findall(header):
            depends.add(os.path.normpath(os.path.join(directory, dependancy)))
        header = RE_PREPROCESSOR.sub("", header)
        header = RE_TAGS.sub("", header)
        header = RE_VAFUNC.sub("", header)
        header = RE_INLINE.sub(r"\1;", header)
        self.header = header.strip()
        self.depends = frozenset(depends)
        self.all_headers[self.path] = self

    def parsed_depends(self) -> Iterator["ParsedHeader"]:
        """Return dependencies excluding ones that were not loaded."""
        for dep in self.depends:
            try:
                yield self.all_headers[dep]
            except KeyError:
                pass

    def __str__(self) -> str:
        return "Parsed harder at '%s'\n Depends on: %s" % (
            self.path,
            "\n\t".join(self.depends),
        )

    def __repr__(self) -> str:
        return "ParsedHeader(%s)" % (self.path,)


def walk_includes(directory: str) -> Iterator[ParsedHeader]:
    """Parse all the include files in a directory and subdirectories."""
    for path, dirs, files in os.walk(directory):
        for file in files:
            if file in HEADER_PARSE_EXCLUDES:
                continue
            if file.endswith(".h"):
                yield ParsedHeader(os.path.join(path, file))


def resolve_dependencies(
    includes: Iterable[ParsedHeader],
) -> List[ParsedHeader]:
    """Sort headers by their correct include order."""
    unresolved = set(includes)
    resolved = set()  # type: Set[ParsedHeader]
    result = []
    while unresolved:
        for item in unresolved:
            if frozenset(item.parsed_depends()).issubset(resolved):
                resolved.add(item)
                result.append(item)
        if not unresolved & resolved:
            raise RuntimeError(
                "Could not resolve header load order.\n"
                "Possible cyclic dependency with the unresolved headers:\n%s"
                % (unresolved,)
            )
        unresolved -= resolved
    return result


def parse_includes() -> List[ParsedHeader]:
    """Collect all parsed header files and return them.

    Reads HEADER_PARSE_PATHS and HEADER_PARSE_EXCLUDES."""
    includes = []  # type: List[ParsedHeader]
    for dirpath in HEADER_PARSE_PATHS:
        includes.extend(walk_includes(dirpath))
    return resolve_dependencies(includes)


def walk_sources(directory: str, cpp: bool) -> Iterator[str]:
    for path, dirs, files in os.walk(directory):
        for source in files:
            if source.endswith(".c"):
                yield os.path.join(path, source)
            elif cpp and source.endswith(".cpp"):
                yield os.path.join(path, source)


def find_sources(directory: str) -> List[str]:
    return [
        os.path.join(directory, source)
        for source in os.listdir(directory)
        if source.endswith(".c")
    ]


def get_sdl2_file(version: str) -> str:
    if sys.platform == "win32":
        sdl2_file = "SDL2-devel-%s-VC.zip" % (version,)
    else:
        assert sys.platform == "darwin"
        sdl2_file = "SDL2-%s.dmg" % (version,)
    sdl2_local_file = os.path.join("dependencies", sdl2_file)
    sdl2_remote_file = "https://www.libsdl.org/release/%s" % sdl2_file
    if not os.path.exists(sdl2_local_file):
        print("Downloading %s" % sdl2_remote_file)
        os.makedirs("dependencies/", exist_ok=True)
        urlretrieve(sdl2_remote_file, sdl2_local_file)
    return sdl2_local_file


def unpack_sdl2(version: str) -> str:
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


includes = parse_includes()

module_name = "tcod._libtcod"
include_dirs = [".", "libtcod/src/vendor/", "libtcod/src/vendor/zlib/"]

extra_parse_args = []
extra_compile_args = []
extra_link_args = []
sources = []  # type: List[str]

libraries = []
library_dirs = []
define_macros = [("Py_LIMITED_API", 0x03050000)]  # type: List[Tuple[str, Any]]

sources += walk_sources("tcod/", cpp=True)
sources += walk_sources("libtcod/src/libtcod/", cpp=False)
sources += ["libtcod/src/vendor/stb.c"]
sources += ["libtcod/src/vendor/glad.c"]
sources += ["libtcod/src/vendor/lodepng.c"]
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
    matches = re.findall(
        r"-I(\S+)",
        subprocess.check_output(
            ["sdl2-config", "--cflags"], universal_newlines=True
        ),
    )
    assert matches

    SDL2_INCLUDE = None
    for match in matches:
        if os.path.isfile(os.path.join(match, "SDL_stdinc.h")):
            SDL2_INCLUDE = match
    assert SDL2_INCLUDE

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


def fix_header(filepath: str) -> None:
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

    # Fix "implicit declaration of function 'close'" in zlib.
    define_macros.append(("HAVE_UNISTD_H", 1))

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

tdl_build = os.environ.get("TDL_BUILD", "RELEASE").upper()

MSVC_CFLAGS = {"DEBUG": ["/Od"], "RELEASE": ["/GL", "/O2", "/GS-", "/wd4996"]}
MSVC_LDFLAGS = {"DEBUG": [], "RELEASE": ["/LTCG"]}
GCC_CFLAGS = {
    "DEBUG": ["-std=c99", "-Og", "-g", "-fPIC"],
    "RELEASE": [
        "-std=c99",
        "-flto",
        "-O3",
        "-g",
        "-fPIC",
        "-Wno-deprecated-declarations",
        "-Wno-discarded-qualifiers",  # Ignore discarded restrict qualifiers.
    ],
}

if sys.platform == "win32" and "--compiler=mingw32" not in sys.argv:
    extra_compile_args.extend(MSVC_CFLAGS[tdl_build])
    extra_link_args.extend(MSVC_LDFLAGS[tdl_build])
else:
    extra_compile_args.extend(GCC_CFLAGS[tdl_build])
    extra_link_args.extend(GCC_CFLAGS[tdl_build])

ffi = FFI()
parse_sdl2.add_to_ffi(ffi, SDL2_INCLUDE)
for include in includes:
    try:
        ffi.cdef(include.header)
    except Exception:
        # Print the source, for debugging.
        print("Error with: %s" % include.path)
        for i, line in enumerate(include.header.split("\n"), 1):
            print("%03i %s" % (i, line))
        raise
ffi.cdef("""
#define TCOD_COMPILEDVERSION ...
""")
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
    py_limited_api=True,
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
    from tcod._libtcod import lib  # type: ignore

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


EXCLUDE_CONSTANTS = [
    "TCOD_MAJOR_VERSION",
    "TCOD_MINOR_VERSION",
    "TCOD_PATCHLEVEL",
    "TCOD_COMPILEDVERSION",
    "TCOD_PATHFINDER_MAX_DIMENSIONS",
    "TCOD_KEY_TEXT_SIZE",
    "TCOD_NOISE_MAX_DIMENSIONS",
    "TCOD_NOISE_MAX_OCTAVES",
]

EXCLUDE_CONSTANT_PREFIXES = [
    "TCOD_E_",
    "TCOD_HEAP_",
    "TCOD_LEX_",
    "TCOD_CHARMAP_",
]


def update_module_all(filename: str, new_all: str) -> None:
    """Update the __all__ of a file with the constants from new_all."""
    RE_CONSTANTS_ALL = re.compile(
        r"(.*# --- From constants.py ---).*(# --- End constants.py ---.*)",
        re.DOTALL,
    )
    with open(filename, "r") as f:
        match = RE_CONSTANTS_ALL.match(f.read())
    assert match, "Can't determine __all__ subsection in %s!" % (filename,)
    header, footer = match.groups()
    with open(filename, "w") as f:
        f.write("%s\n    %s,\n    %s" % (header, new_all, footer))


def write_library_constants() -> None:
    """Write libtcod constants into the tcod.constants module."""
    from tcod._libtcod import lib, ffi
    import tcod.color

    with open("tcod/constants.py", "w") as f:
        all_names = []
        f.write(CONSTANT_MODULE_HEADER)
        for name in dir(lib):
            # To exclude specific names use either EXCLUDE_CONSTANTS or
            # EXCLUDE_CONSTANT_PREFIXES before editing this.
            if name.endswith("_"):
                continue
            if name in EXCLUDE_CONSTANTS:
                continue
            if any(
                name.startswith(prefix) for prefix in EXCLUDE_CONSTANT_PREFIXES
            ):
                continue
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
        update_module_all("tcod/__init__.py", all_names)
        update_module_all("tcod/libtcodpy.py", all_names)

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
