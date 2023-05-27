#!/usr/bin/env python3
"""Build script to parse SDL headers and generate CFFI bindings."""
from __future__ import annotations

import io
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pcpp  # type: ignore
import requests

# ruff: noqa: S603, S607  # This script calls a lot of programs.

BIT_SIZE, LINKAGE = platform.architecture()

# Reject versions of SDL older than this, update the requirements in the readme if you change this.
SDL_MIN_VERSION = (2, 0, 10)
# The SDL2 version to parse and export symbols from.
SDL2_PARSE_VERSION = os.environ.get("SDL_VERSION", "2.0.20")
# The SDL2 version to include in binary distributions.
SDL2_BUNDLE_VERSION = os.environ.get("SDL_VERSION", "2.26.0")


# Used to remove excessive newlines in debug outputs.
RE_NEWLINES = re.compile(r"\n\n+")
# Functions using va_list need to be culled.
RE_VAFUNC = re.compile(r"^.*?\([^()]*va_list[^()]*\);$", re.MULTILINE)
# Static inline functions need to be culled.
RE_INLINE = re.compile(r"^static inline.*?^}$", re.MULTILINE | re.DOTALL)
# Most SDL_PIXELFORMAT names need their values scrubbed.
RE_PIXELFORMAT = re.compile(r"(?P<name>SDL_PIXELFORMAT_\w+) =[^,}]*")
# Most SDLK names need their values scrubbed.
RE_SDLK = re.compile(r"(?P<name>SDLK_\w+) =.*?(?=,\n|}\n)")
# Remove compile time assertions from the cdef.
RE_ASSERT = re.compile(r"^.*SDL_compile_time_assert.*$", re.MULTILINE)
# Padding values need to be scrubbed.
RE_PADDING = re.compile(r"padding\[[^;]*\];")

# These structs have an unusual size when packed by SDL on 32-bit platforms.
FLEXIBLE_STRUCTS = (
    "SDL_AudioCVT",
    "SDL_TouchFingerEvent",
    "SDL_MultiGestureEvent",
    "SDL_DollarGestureEvent",
)

# Other defined names which sometimes cause issues when parsed.
IGNORE_DEFINES = frozenset(
    (
        "SDL_DEPRECATED",
        "SDL_INLINE",
        "SDL_FORCE_INLINE",
        "SDL_FALLTHROUGH",
        # Might show up in parsing and not in source.
        "SDL_ANDROID_EXTERNAL_STORAGE_READ",
        "SDL_ANDROID_EXTERNAL_STORAGE_WRITE",
        "SDL_ASSEMBLY_ROUTINES",
        "SDL_RWOPS_VITAFILE",
        # Prevent double definition.
        "SDL_FALSE",
        "SDL_TRUE",
        # Ignore floating point symbols.
        "SDL_FLT_EPSILON",
        # Conditional config flags which might be missing.
        "SDL_VIDEO_RENDER_D3D12",
        "SDL_SENSOR_WINDOWS",
        "SDL_SENSOR_DUMMY",
    )
)


def check_sdl_version() -> None:
    """Check the local SDL version on Linux distributions."""
    if not sys.platform.startswith("linux"):
        return
    needed_version = f"{SDL_MIN_VERSION[0]}.{SDL_MIN_VERSION[1]}.{SDL_MIN_VERSION[2]}"
    try:
        sdl_version_str = subprocess.check_output(["sdl2-config", "--version"], universal_newlines=True).strip()
    except FileNotFoundError as exc:
        msg = (
            "libsdl2-dev or equivalent must be installed on your system and must be at least version"
            f" {needed_version}.\nsdl2-config must be on PATH."
        )
        raise RuntimeError(msg) from exc
    print(f"Found SDL {sdl_version_str}.")
    sdl_version = tuple(int(s) for s in sdl_version_str.split("."))
    if sdl_version < SDL_MIN_VERSION:
        msg = f"SDL version must be at least {needed_version}, (found {sdl_version_str})"
        raise RuntimeError(msg)


def get_sdl2_file(version: str) -> Path:
    """Return a path to an SDL2 archive for the current platform.  The archive is downloaded if missing."""
    if sys.platform == "win32":
        sdl2_file = f"SDL2-devel-{version}-VC.zip"
    else:
        assert sys.platform == "darwin"
        sdl2_file = f"SDL2-{version}.dmg"
    sdl2_local_file = Path("dependencies", sdl2_file)
    sdl2_remote_file = f"https://www.libsdl.org/release/{sdl2_file}"
    if not sdl2_local_file.exists():
        print(f"Downloading {sdl2_remote_file}")
        Path("dependencies/").mkdir(parents=True, exist_ok=True)
        with requests.get(sdl2_remote_file) as response:  # noqa: S113
            response.raise_for_status()
            sdl2_local_file.write_bytes(response.content)
    return sdl2_local_file


def unpack_sdl2(version: str) -> Path:
    """Return the path to an extracted SDL distribution.  Creates it if missing."""
    sdl2_path = Path(f"dependencies/SDL2-{version}")
    if sys.platform == "darwin":
        sdl2_dir = sdl2_path
        sdl2_path /= "SDL2.framework"
    if sdl2_path.exists():
        return sdl2_path
    sdl2_arc = get_sdl2_file(version)
    print(f"Extracting {sdl2_arc}")
    if sdl2_arc.suffix == ".zip":
        with zipfile.ZipFile(sdl2_arc) as zf:
            zf.extractall("dependencies/")
    elif sys.platform == "darwin":
        assert sdl2_arc.suffix == ".dmg"
        subprocess.check_call(["hdiutil", "mount", sdl2_arc])
        subprocess.check_call(["mkdir", "-p", sdl2_dir])
        subprocess.check_call(["cp", "-r", "/Volumes/SDL2/SDL2.framework", sdl2_dir])
        subprocess.check_call(["hdiutil", "unmount", "/Volumes/SDL2"])
    return sdl2_path


class SDLParser(pcpp.Preprocessor):  # type: ignore
    """A modified preprocessor to output code in a format for CFFI."""

    def __init__(self) -> None:
        """Initialise the object with empty values."""
        super().__init__()
        self.line_directive = None  # Don't output line directives.
        self.known_string_defines: dict[str, str] = {}
        self.known_defines: set[str] = set()

    def get_output(self) -> str:
        """Return this objects current tokens as a string."""
        with io.StringIO() as buffer:
            self.write(buffer)
            for name in self.known_defines:
                buffer.write(f"#define {name} ...\n")
            return buffer.getvalue()

    def on_include_not_found(self, is_malformed: bool, is_system_include: bool, curdir: str, includepath: str) -> None:
        """Remove bad includes such as stddef.h and stdarg.h."""
        raise pcpp.OutputDirective(pcpp.Action.IgnoreAndRemove)

    def _should_track_define(self, tokens: list[Any]) -> bool:
        if len(tokens) < 3:
            return False
        if tokens[0].value in IGNORE_DEFINES:
            return False
        if not tokens[0].value.isupper():
            return False  # Function-like name, such as SDL_snprintf.
        if tokens[0].value.startswith("_") or tokens[0].value.endswith("_"):
            return False  # Private name.
        if tokens[2].value.startswith("_") or tokens[2].value.endswith("_"):
            return False  # Likely calls a private function.
        if tokens[1].type == "CPP_LPAREN":
            return False  # Function-like macro.
        if len(tokens) >= 4 and tokens[2].type == "CPP_INTEGER" and tokens[3].type == "CPP_DOT":
            return False  # Value is a floating point number.
        if tokens[0].value.startswith("SDL_PR") and (tokens[0].value.endswith("32") or tokens[0].value.endswith("64")):
            return False  # Data type for printing, which is not needed.
        return bool(
            tokens[0].value.startswith("KMOD_")
            or tokens[0].value.startswith("SDL_")
            or tokens[0].value.startswith("AUDIO_")
        )

    def on_directive_handle(
        self, directive: Any, tokens: list[Any], if_passthru: bool, preceding_tokens: list[Any]  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Catch and store definitions."""
        if directive.value == "define" and self._should_track_define(tokens):
            if tokens[2].type == "CPP_STRING":
                self.known_string_defines[tokens[0].value] = tokens[2].value
            else:
                self.known_defines.add(tokens[0].value)
        return super().on_directive_handle(directive, tokens, if_passthru, preceding_tokens)


check_sdl_version()

if sys.platform in ["win32", "darwin"]:
    SDL2_PARSE_PATH = unpack_sdl2(SDL2_PARSE_VERSION)
    SDL2_BUNDLE_PATH = unpack_sdl2(SDL2_BUNDLE_VERSION)

SDL2_INCLUDE: Path
if sys.platform == "win32":
    SDL2_INCLUDE = SDL2_PARSE_PATH / "include"
elif sys.platform == "darwin":
    SDL2_INCLUDE = SDL2_PARSE_PATH / "Versions/A/Headers"
else:  # Unix
    matches = re.findall(
        r"-I(\S+)",
        subprocess.check_output(["sdl2-config", "--cflags"], universal_newlines=True),
    )
    assert matches

    for match in matches:
        if Path(match, "SDL_stdinc.h").is_file():
            SDL2_INCLUDE = match
    assert SDL2_INCLUDE


EXTRA_CDEF = """
#define SDLK_SCANCODE_MASK ...

extern "Python" {
// SDL_AudioCallback callback.
void _sdl_audio_callback(void* userdata, Uint8* stream, int len);
// SDL to Python log function.
void _sdl_log_output_function(void *userdata, int category, SDL_LogPriority priority, const char *message);
// Generic event watcher callback.
int _sdl_event_watcher(void* userdata, SDL_Event* event);
}
"""


def get_cdef() -> str:
    """Return the parsed code of SDL for CFFI."""
    parser = SDLParser()
    parser.add_path(SDL2_INCLUDE)
    parser.parse(
        """
    // Remove extern keyword.
    #define extern
    // Ignore some SDL assert statements.
    #define DOXYGEN_SHOULD_IGNORE_THIS

    #define _SIZE_T_DEFINED_
    typedef int... size_t;

    // Skip these headers.
    #define SDL_atomic_h_
    #define SDL_thread_h_

    #include <SDL.h>
    """
    )
    sdl2_cdef = parser.get_output()
    sdl2_cdef = RE_VAFUNC.sub("", sdl2_cdef)
    sdl2_cdef = RE_INLINE.sub("", sdl2_cdef)
    sdl2_cdef = RE_PIXELFORMAT.sub(r"\g<name> = ...", sdl2_cdef)
    sdl2_cdef = RE_SDLK.sub(r"\g<name> = ...", sdl2_cdef)
    sdl2_cdef = RE_NEWLINES.sub("\n", sdl2_cdef)
    sdl2_cdef = RE_ASSERT.sub("", sdl2_cdef)
    sdl2_cdef = RE_PADDING.sub("padding[...];", sdl2_cdef)
    sdl2_cdef = (
        sdl2_cdef.replace("int SDL_main(int argc, char *argv[]);", "")
        .replace("typedef unsigned int uintptr_t;", "typedef int... uintptr_t;")
        .replace("typedef unsigned int size_t;", "typedef int... size_t;")
    )
    for name in FLEXIBLE_STRUCTS:
        sdl2_cdef = sdl2_cdef.replace(f"}} {name};", f"...;}} {name};")
    return sdl2_cdef + EXTRA_CDEF


include_dirs: list[str] = []
extra_compile_args: list[str] = []
extra_link_args: list[str] = []

libraries: list[str] = []
library_dirs: list[str] = []


if sys.platform == "darwin":
    extra_link_args += ["-framework", "SDL2"]
else:
    libraries += ["SDL2"]

# Bundle the Windows SDL2 DLL.
if sys.platform == "win32":
    include_dirs.append(str(SDL2_INCLUDE))
    ARCH_MAPPING = {"32bit": "x86", "64bit": "x64"}
    SDL2_LIB_DIR = Path(SDL2_BUNDLE_PATH, "lib/", ARCH_MAPPING[BIT_SIZE])
    library_dirs.append(str(SDL2_LIB_DIR))
    SDL2_LIB_DEST = Path("tcod", ARCH_MAPPING[BIT_SIZE])
    SDL2_LIB_DEST.mkdir(exist_ok=True)
    SDL2_LIB_DEST_FILE = SDL2_LIB_DEST / "SDL2.dll"
    SDL2_LIB_FILE = SDL2_LIB_DIR / "SDL2.dll"
    if not SDL2_LIB_DEST_FILE.exists() or SDL2_LIB_FILE.read_bytes() != SDL2_LIB_DEST_FILE.read_bytes():
        shutil.copy(SDL2_LIB_FILE, SDL2_LIB_DEST_FILE)

# Link to the SDL2 framework on MacOS.
# Delocate will bundle the binaries in a later step.
if sys.platform == "darwin":
    HEADER_DIR = Path(SDL2_PARSE_PATH, "Headers")
    include_dirs.append(HEADER_DIR)
    extra_link_args += [f"-F{SDL2_BUNDLE_PATH}/.."]
    extra_link_args += ["-rpath", f"{SDL2_BUNDLE_PATH}/.."]
    extra_link_args += ["-rpath", "/usr/local/opt/llvm/lib/"]

# Use sdl2-config to link to SDL2 on Linux.
if sys.platform not in ["win32", "darwin"]:
    extra_compile_args += subprocess.check_output(["sdl2-config", "--cflags"], universal_newlines=True).strip().split()
    extra_link_args += subprocess.check_output(["sdl2-config", "--libs"], universal_newlines=True).strip().split()
