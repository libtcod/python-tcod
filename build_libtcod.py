#!/usr/bin/env python
"""Parse and compile libtcod and SDL sources for CFFI."""

from __future__ import annotations

import ast
import contextlib
import glob
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import attrs
import pycparser  # type: ignore[import-untyped]
import pycparser.c_ast  # type: ignore[import-untyped]
import pycparser.c_generator  # type: ignore[import-untyped]
from cffi import FFI

# ruff: noqa: T201

sys.path.append(str(Path(__file__).parent))  # Allow importing local modules.

import build_sdl

Py_LIMITED_API = 0x03100000

HEADER_PARSE_PATHS = ("tcod/", "libtcod/src/libtcod/")
HEADER_PARSE_EXCLUDES = ("gl2_ext_.h", "renderer_gl_internal.h", "event.h")

BIT_SIZE, LINKAGE = platform.architecture()

# Regular expressions to parse the headers for cffi.
RE_COMMENT = re.compile(r"\s*/\*.*?\*/|\s*//*?$", re.DOTALL | re.MULTILINE)
RE_CPLUSPLUS = re.compile(r"#ifdef __cplusplus.*?#endif.*?$", re.DOTALL | re.MULTILINE)
RE_PREPROCESSOR = re.compile(r"(?!#define\s+\w+\s+\d+$)#.*?(?<!\\)$", re.DOTALL | re.MULTILINE)
RE_INCLUDE = re.compile(r'#include "([^"]*)"')
RE_TAGS = re.compile(
    r"TCODLIB_C?API|TCOD_PUBLIC|TCOD_NODISCARD|TCOD_DEPRECATED_NOMESSAGE|TCOD_DEPRECATED_ENUM"
    r"|(TCOD_DEPRECATED\(\".*?\"\))"
    r"|(TCOD_DEPRECATED|TCODLIB_FORMAT)\([^)]*\)|__restrict"
)
RE_VAFUNC = re.compile(r"^[^;]*\([^;]*va_list.*\);", re.MULTILINE)
RE_INLINE = re.compile(r"(^.*?inline.*?\(.*?\))\s*\{.*?\}$", re.DOTALL | re.MULTILINE)


class ParsedHeader:
    """Header manager class for parsing headers.

    Holds parsed sources and keeps information needed to resolve header order.
    """

    # Class dictionary of all parsed headers.
    all_headers: ClassVar[dict[Path, ParsedHeader]] = {}

    def __init__(self, path: Path) -> None:
        """Initialize and organize a header file."""
        self.path = path = path.resolve(strict=True)
        directory = path.parent
        depends = set()
        header = self.path.read_text(encoding="utf-8")
        header = RE_COMMENT.sub("", header)
        header = RE_CPLUSPLUS.sub("", header)
        for dependency in RE_INCLUDE.findall(header):
            depends.add((directory / str(dependency)).resolve(strict=True))
        header = RE_PREPROCESSOR.sub("", header)
        header = RE_TAGS.sub("", header)
        header = RE_VAFUNC.sub("", header)
        header = RE_INLINE.sub(r"\1;", header)
        self.header = header.strip()
        self.depends = frozenset(depends)
        self.all_headers[self.path] = self

    def parsed_depends(self) -> Iterator[ParsedHeader]:
        """Return dependencies excluding ones that were not loaded."""
        for dep in self.depends:
            with contextlib.suppress(KeyError):
                yield self.all_headers[dep]

    def __str__(self) -> str:
        """Return useful info on this object."""
        return "Parsed harder at '{}'\n Depends on: {}".format(
            self.path,
            "\n\t".join(str(d) for d in self.depends),
        )

    def __repr__(self) -> str:
        """Return the representation of this object."""
        return f"ParsedHeader({self.path!r})"


def walk_includes(directory: str) -> Iterator[ParsedHeader]:
    """Parse all the include files in a directory and subdirectories."""
    for path, _dirs, files in os.walk(directory):
        for file in files:
            if file in HEADER_PARSE_EXCLUDES:
                continue
            if file.endswith(".h"):
                yield ParsedHeader(Path(path, file).resolve(strict=True))


def resolve_dependencies(
    includes: Iterable[ParsedHeader],
) -> list[ParsedHeader]:
    """Sort headers by their correct include order."""
    unresolved = set(includes)
    resolved: set[ParsedHeader] = set()
    result = []
    while unresolved:
        for item in unresolved:
            if frozenset(item.parsed_depends()).issubset(resolved):
                resolved.add(item)
                result.append(item)
        if not unresolved & resolved:
            msg = (
                "Could not resolve header load order."
                "\nPossible cyclic dependency with the unresolved headers:"
                f"\n{unresolved}"
            )
            raise RuntimeError(msg)
        unresolved -= resolved
    return result


def parse_includes() -> list[ParsedHeader]:
    """Collect all parsed header files and return them.

    Reads HEADER_PARSE_PATHS and HEADER_PARSE_EXCLUDES.
    """
    includes: list[ParsedHeader] = []
    for dirpath in HEADER_PARSE_PATHS:
        includes.extend(walk_includes(dirpath))
    return resolve_dependencies(includes)


def walk_sources(directory: str) -> Iterator[str]:
    """Iterate over the C sources of a directory recursively."""
    for path, _dirs, files in os.walk(directory):
        for source in files:
            if source.endswith(".c"):
                yield str(Path(path, source))


includes = parse_includes()

module_name = "tcod._libtcod"
include_dirs: list[str] = [
    ".",
    "libtcod/src/vendor/",
    "libtcod/src/vendor/utf8proc",
    "libtcod/src/vendor/zlib/",
    *build_sdl.include_dirs,
]

extra_compile_args: list[str] = [*build_sdl.extra_compile_args]
extra_link_args: list[str] = [*build_sdl.extra_link_args]
sources: list[str] = []

libraries: list[str] = [*build_sdl.libraries]
library_dirs: list[str] = [*build_sdl.library_dirs]
define_macros: list[tuple[str, Any]] = []

if "PYODIDE" not in os.environ:
    # Unable to apply Py_LIMITED_API to Pyodide in cffi<=1.17.1
    # https://github.com/python-cffi/cffi/issues/179
    define_macros.append(("Py_LIMITED_API", Py_LIMITED_API))

sources += walk_sources("tcod/")
sources += walk_sources("libtcod/src/libtcod/")
sources += ["libtcod/src/vendor/stb.c"]
sources += ["libtcod/src/vendor/lodepng.c"]
sources += ["libtcod/src/vendor/utf8proc/utf8proc.c"]
sources += glob.glob("libtcod/src/vendor/zlib/*.c")

if sys.platform == "win32":
    libraries += ["User32"]
    define_macros.append(("TCODLIB_API", ""))
    define_macros.append(("_CRT_SECURE_NO_WARNINGS", None))

if sys.platform in ["win32", "darwin"]:
    include_dirs.append("libtcod/src/zlib/")


if sys.platform != "win32":
    # Fix implicit declaration of multiple functions in zlib.
    define_macros.append(("HAVE_UNISTD_H", 1))


tdl_build = os.environ.get("TDL_BUILD", "RELEASE").upper()

MSVC_CFLAGS = {"DEBUG": ["/Od"], "RELEASE": ["/GL", "/O2", "/GS-", "/wd4996"]}
MSVC_LDFLAGS: dict[str, list[str]] = {"DEBUG": [], "RELEASE": ["/LTCG"]}
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
sdl_cdef, sdl_strings = build_sdl.get_cdef()
ffi.cdef(sdl_cdef)
for include in includes:
    try:
        ffi.cdef(include.header)
    except Exception:  # noqa: PERF203
        # Print the source, for debugging.
        print(f"Error with: {include.path}")
        for i, line in enumerate(include.header.split("\n"), 1):
            print(f"{i:03i} {line}")
        raise
ffi.cdef(
    """
#define TCOD_COMPILEDVERSION ...
"""
)
ffi.set_source(
    module_name,
    """\
#include <tcod/cffi.h>
#define SDL_oldnames_h_
#include <SDL3/SDL.h>""",
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    sources=sources,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    py_limited_api=True,
)

CONSTANT_MODULE_HEADER = '''"""Constants from the libtcod C API.

This module is auto-generated by `build_libtcod.py`.
"""

from tcod.color import Color

'''

EVENT_CONSTANT_MODULE_HEADER = '''"""Event constants from SDL's C API.

This module is auto-generated by `build_libtcod.py`.
"""
'''


def find_sdl_attrs(prefix: str) -> Iterator[tuple[str, int | str | Any]]:
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


def parse_sdl_attrs(prefix: str, all_names: list[str] | None) -> tuple[str, str]:
    """Return the name/value pairs, and the final dictionary string for the library attributes with `prefix`.

    Append matching names to the `all_names` list.
    """
    names = []
    lookup = []
    for name, value in sorted(find_sdl_attrs(prefix), key=lambda item: item[1]):
        if name == "KMOD_RESERVED":
            continue
        if all_names is not None:
            all_names.append(name)
        names.append(f"{name} = {value}")
        lookup.append(f'{value}: "{name}"')
    return "\n".join(names), "{{\n    {},\n}}".format(",\n    ".join(lookup))


EXCLUDE_CONSTANTS = [
    "TCOD_MAJOR_VERSION",
    "TCOD_MINOR_VERSION",
    "TCOD_PATCHLEVEL",
    "TCOD_COMPILEDVERSION",
    "TCOD_PATHFINDER_MAX_DIMENSIONS",
    "TCOD_KEY_TEXT_SIZE",
    "TCOD_NOISE_MAX_DIMENSIONS",
    "TCOD_NOISE_MAX_OCTAVES",
    "TCOD_FALLBACK_FONT_SIZE",
]

EXCLUDE_CONSTANT_PREFIXES = [
    "TCOD_E_",
    "TCOD_HEAP_",
    "TCOD_LEX_",
    "TCOD_CHARMAP_",
    "TCOD_LOG_",
]


def update_module_all(filename: Path, new_all: str) -> None:
    """Update the __all__ of a file with the constants from new_all."""
    RE_CONSTANTS_ALL = re.compile(
        r"(.*# --- From constants.py ---).*(# --- End constants.py ---.*)",
        re.DOTALL,
    )
    match = RE_CONSTANTS_ALL.match(filename.read_text(encoding="utf-8"))
    assert match, f"Can't determine __all__ subsection in {filename}!"
    header, footer = match.groups()
    filename.write_text(f"{header}\n    {new_all},\n    {footer}", encoding="utf-8")


def generate_enums(prefix: str) -> Iterator[str]:
    """Generate attribute assignments suitable for a Python enum."""
    for symbol, value in sorted(find_sdl_attrs(prefix), key=lambda item: item[1]):
        _, name = symbol.split("_", 1)
        if name.isdigit():
            name = f"N{name}"
        if name in "IOl":  # Ignore ambiguous variable name warnings.
            yield f"{name} = {value}  # noqa: E741"
        else:
            yield f"{name} = {value}"


def write_library_constants() -> None:
    """Write libtcod constants into the tcod.constants module."""
    import tcod.color
    from tcod._libtcod import ffi, lib

    with Path("tcod/constants.py").open("w", encoding="utf-8") as f:
        all_names = []
        f.write(CONSTANT_MODULE_HEADER)
        for name in dir(lib):
            # To exclude specific names use either EXCLUDE_CONSTANTS or
            # EXCLUDE_CONSTANT_PREFIXES before editing this.
            if name.endswith("_"):
                continue
            if name in EXCLUDE_CONSTANTS:
                continue
            if any(name.startswith(prefix) for prefix in EXCLUDE_CONSTANT_PREFIXES):
                continue
            value = getattr(lib, name)
            if name[:5] == "TCOD_":
                if name.isupper():  # const names
                    f.write(f"{name[5:]} = {value!r}\n")
                    all_names.append(name[5:])
            elif name.startswith("FOV"):  # fov const names
                f.write(f"{name} = {value!r}\n")
                all_names.append(name)
            elif name[:6] == "TCODK_":  # key name
                f.write(f"KEY_{name[6:]} = {value!r}\n")
                all_names.append(f"KEY_{name[6:]}")

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
            f.write(f"{name[5:]} = {color!r}\n")

        all_names_merged = ",\n    ".join(f'"{name}"' for name in all_names)
        f.write(f"\n__all__ = [  # noqa: RUF022\n    {all_names_merged},\n]\n")
        update_module_all(Path("tcod/libtcodpy.py"), all_names_merged)

    with Path("tcod/event_constants.py").open("w", encoding="utf-8") as f:
        all_names = []
        f.write(EVENT_CONSTANT_MODULE_HEADER)
        f.write("\n# --- SDL scancodes ---\n")
        f.write(f"""{parse_sdl_attrs("SDL_SCANCODE", None)[0]}\n""")

        f.write("\n# --- SDL keyboard symbols ---\n")
        f.write(f"""{parse_sdl_attrs("SDLK_", None)[0]}\n""")

        f.write("\n# --- SDL keyboard modifiers ---\n")
        f.write("{}\n_REVERSE_MOD_TABLE = {}\n".format(*parse_sdl_attrs("SDL_KMOD", None)))

        f.write("\n# --- SDL wheel ---\n")
        f.write("{}\n_REVERSE_WHEEL_TABLE = {}\n".format(*parse_sdl_attrs("SDL_MOUSEWHEEL", all_names)))
        all_names_merged = ",\n    ".join(f'"{name}"' for name in all_names)
        f.write(f"\n__all__ = [  # noqa: RUF022\n    {all_names_merged},\n]\n")

    event_py = Path("tcod/event.py").read_text(encoding="utf-8")

    event_py = re.sub(
        r"(?<=# --- SDL scancodes ---\n    ).*?(?=\n    # --- end ---\n)",
        "\n    ".join(generate_enums("SDL_SCANCODE")),
        event_py,
        flags=re.DOTALL,
    )
    event_py = re.sub(
        r"(?<=# --- SDL keyboard symbols ---\n    ).*?(?=\n    # --- end ---\n)",
        "\n    ".join(generate_enums("SDLK")),
        event_py,
        flags=re.DOTALL,
    )

    Path("tcod/event.py").write_text(event_py, encoding="utf-8")

    with Path("tcod/sdl/constants.py").open("w", encoding="utf-8") as f:
        f.write('"""SDL private constants."""\n\n')
        for name, value in sdl_strings.items():
            f.write(f"{name} = {ast.literal_eval(value)!r}\n")

    subprocess.run(["ruff", "format", "--silent", Path("tcod/sdl/constants.py")], check=True)  # noqa: S603, S607


def _fix_reserved_name(name: str) -> str:
    """Add underscores to reserved Python keywords."""
    assert isinstance(name, str)
    if name in ("def", "in"):
        return name + "_"
    return name


@attrs.define(frozen=True)
class ConvertedParam:
    name: str = attrs.field(converter=_fix_reserved_name)
    hint: str
    original: str


def _type_from_names(names: list[str]) -> str:
    if not names:
        return ""
    if names[-1] == "void":
        return "None"
    if names in (["unsigned", "char"], ["bool"]):
        return "bool"
    if names[-1] in ("size_t", "int", "ptrdiff_t"):
        return "int"
    if names[-1] in ("float", "double"):
        return "float"
    return "Any"


def _param_as_hint(node: pycparser.c_ast.Node, default_name: str) -> ConvertedParam:
    original = pycparser.c_generator.CGenerator().visit(node)
    name: str
    names: list[str]
    match node:
        case pycparser.c_ast.Typename(type=pycparser.c_ast.TypeDecl(type=pycparser.c_ast.IdentifierType(names=names))):
            # Unnamed type
            return ConvertedParam(default_name, _type_from_names(names), original)
        case pycparser.c_ast.Decl(
            name=name, type=pycparser.c_ast.TypeDecl(type=pycparser.c_ast.IdentifierType(names=names))
        ):
            # Named type
            return ConvertedParam(name, _type_from_names(names), original)
        case pycparser.c_ast.Decl(
            name=name,
            type=pycparser.c_ast.ArrayDecl(
                type=pycparser.c_ast.TypeDecl(type=pycparser.c_ast.IdentifierType(names=names))
            ),
        ):
            # Named array
            return ConvertedParam(name, "Any", original)
        case pycparser.c_ast.Decl(name=name, type=pycparser.c_ast.PtrDecl()):
            # Named pointer
            return ConvertedParam(name, "Any", original)
        case pycparser.c_ast.Typename(name=name, type=pycparser.c_ast.PtrDecl()):
            # Forwarded struct
            return ConvertedParam(name or default_name, "Any", original)
        case pycparser.c_ast.TypeDecl(type=pycparser.c_ast.IdentifierType(names=names)):
            # Return type
            return ConvertedParam(default_name, _type_from_names(names), original)
        case pycparser.c_ast.PtrDecl():
            # Return pointer
            return ConvertedParam(default_name, "Any", original)
        case pycparser.c_ast.EllipsisParam():
            # C variable args
            return ConvertedParam("*__args", "Any", original)
        case _:
            raise AssertionError


class DefinitionCollector(pycparser.c_ast.NodeVisitor):  # type: ignore[misc]
    """Gathers functions and names from C headers."""

    def __init__(self) -> None:
        """Initialize the object with empty values."""
        self.functions: list[str] = []
        """Indented Python function definitions."""
        self.variables: set[str] = set()
        """Python variable definitions."""

    def parse_defines(self, string: str, /) -> None:
        """Parse C define directives into hinted names."""
        for match in re.finditer(r"#define\s+(\S+)\s+(\S+)\s*", string):
            name, value = match.groups()
            if value == "...":
                self.variables.add(f"{name}: Final[int]")
            else:
                self.variables.add(f"{name}: Final[Literal[{value}]] = {value}")

    def visit_Decl(self, node: pycparser.c_ast.Decl) -> None:  # noqa: N802
        """Parse C FFI functions into type hinted Python functions."""
        match node:
            case pycparser.c_ast.Decl(
                type=pycparser.c_ast.FuncDecl(),
            ):
                assert isinstance(node.type.args, pycparser.c_ast.ParamList), type(node.type.args)
                arg_hints = [_param_as_hint(param, f"arg{i}") for i, param in enumerate(node.type.args.params)]
                return_hint = _param_as_hint(node.type.type, "")
                if len(arg_hints) == 1 and arg_hints[0].hint == "None":  # Remove void parameter
                    arg_hints = []

                python_params = [f"{p.name}: {p.hint}" for p in arg_hints]
                if python_params:
                    if arg_hints[-1].name.startswith("*"):
                        python_params.insert(-1, "/")
                    else:
                        python_params.append("/")
                c_def = pycparser.c_generator.CGenerator().visit(node)
                python_def = f"""def {node.name}({", ".join(python_params)}) -> {return_hint.hint}:"""
                self.functions.append(f'''    {python_def}\n        """{c_def}"""''')

    def visit_Enumerator(self, node: pycparser.c_ast.Enumerator) -> None:  # noqa: N802
        """Parse C enums into hinted names."""
        name: str | None
        value: str | int
        match node:
            case pycparser.c_ast.Enumerator(name=name, value=None):
                self.variables.add(f"{name}: Final[int]")
            case pycparser.c_ast.Enumerator(name=name, value=pycparser.c_ast.ID()):
                self.variables.add(f"{name}: Final[int]")
            case pycparser.c_ast.Enumerator(name=name, value=pycparser.c_ast.Constant(value=value)):
                value = int(str(value).removesuffix("u"), base=0)
                self.variables.add(f"{name}: Final[Literal[{value}]] = {value}")
            case pycparser.c_ast.Enumerator(
                name=name, value=pycparser.c_ast.UnaryOp(op="-", expr=pycparser.c_ast.Constant(value=value))
            ):
                value = -int(str(value).removesuffix("u"), base=0)
                self.variables.add(f"{name}: Final[Literal[{value}]] = {value}")
            case pycparser.c_ast.Enumerator(name=name):
                self.variables.add(f"{name}: Final[int]")
            case _:
                raise AssertionError


def write_hints() -> None:
    """Write a custom _libtcod.pyi file from C definitions."""
    function_collector = DefinitionCollector()
    c = pycparser.CParser()

    # Parse SDL headers
    cdef = sdl_cdef
    cdef = cdef.replace("int...", "int")
    cdef = (
        """
typedef int bool;
typedef int int8_t;
typedef int uint8_t;
typedef int int16_t;
typedef int uint16_t;
typedef int int32_t;
typedef int uint32_t;
typedef int int64_t;
typedef int uint64_t;
typedef int wchar_t;
typedef int intptr_t;
"""
        + cdef
    )
    for match in re.finditer(r"SDL_PIXELFORMAT_\w+", cdef):
        function_collector.variables.add(f"{match.group()}: int")
    cdef = re.sub(r"(typedef enum SDL_PixelFormat).*(SDL_PixelFormat;)", r"\1 \2", cdef, flags=re.DOTALL)
    cdef = cdef.replace("padding[...]", "padding[]")
    cdef = cdef.replace("...;} SDL_TouchFingerEvent;", "} SDL_TouchFingerEvent;")
    function_collector.parse_defines(cdef)
    cdef = re.sub(r"\n#define .*", "", cdef)
    cdef = re.sub(r"""extern "Python" \{(.*?)\}""", r"\1", cdef, flags=re.DOTALL)
    cdef = re.sub(r"//.*", "", cdef)
    cdef = cdef.replace("...;", ";")
    ast = c.parse(cdef)
    function_collector.visit(ast)

    # Parse libtcod headers
    cdef = "\n".join(include.header for include in includes)
    function_collector.parse_defines(cdef)
    cdef = re.sub(r"\n?#define .*", "", cdef)
    cdef = re.sub(r"//.*", "", cdef)
    cdef = (
        """
typedef int int8_t;
typedef int uint8_t;
typedef int int16_t;
typedef int uint16_t;
typedef int int32_t;
typedef int uint32_t;
typedef int int64_t;
typedef int uint64_t;
typedef int wchar_t;
typedef int intptr_t;
typedef int ptrdiff_t;
typedef int size_t;
typedef unsigned char bool;
typedef void* SDL_PropertiesID;
"""
        + cdef
    )
    cdef = re.sub(r"""extern "Python" \{(.*?)\}""", r"\1", cdef, flags=re.DOTALL)
    function_collector.visit(c.parse(cdef))
    function_collector.variables.add("TCOD_ctx: Any")
    function_collector.variables.add("TCOD_COMPILEDVERSION: Final[int]")

    # Write PYI file
    out_functions = """\n\n    @staticmethod\n""".join(sorted(function_collector.functions))
    out_variables = "\n    ".join(sorted(function_collector.variables))

    pyi = f"""\
# Autogenerated with build_libtcod.py
from typing import Any, Final, Literal

# pyi files for CFFI ports are not standard
# ruff: noqa: A002, ANN401, D402, D403, D415, N801, N802, N803, N815, PLW0211, PYI021

class _lib:
    @staticmethod
{out_functions}

    {out_variables}

lib: _lib
ffi: Any
"""
    Path("tcod/_libtcod.pyi").write_text(pyi)
    subprocess.run(["ruff", "format", "--silent", Path("tcod/_libtcod.pyi")], check=True)  # noqa: S603, S607


if __name__ == "__main__":
    write_hints()
    write_library_constants()
