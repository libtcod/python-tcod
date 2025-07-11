#!/usr/bin/env python
"""Build script to parse SDL headers and generate CFFI bindings."""

from __future__ import annotations

import io
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pcpp  # type: ignore[import-untyped]
import requests

# This script calls a lot of programs.
# ruff: noqa: S603, S607

# Ignore f-strings in logging, these will eventually be replaced with t-strings.
# ruff: noqa: G004

logger = logging.getLogger(__name__)


BIT_SIZE, LINKAGE = platform.architecture()

# Reject versions of SDL older than this, update the requirements in the readme if you change this.
SDL_MIN_VERSION = (3, 2, 0)
# The SDL version to parse and export symbols from.
SDL_PARSE_VERSION = os.environ.get("SDL_VERSION", "3.2.16")
# The SDL version to include in binary distributions.
SDL_BUNDLE_VERSION = os.environ.get("SDL_VERSION", "3.2.16")


# Used to remove excessive newlines in debug outputs.
RE_NEWLINES = re.compile(r"\n\n+")
# Functions using va_list need to be culled.
RE_VAFUNC = re.compile(r"^.*?\([^()]*va_list[^()]*\)\s*;\s*$", re.MULTILINE)
# Static inline functions need to be culled.
RE_INLINE = re.compile(r"^static inline.*?^}$", re.MULTILINE | re.DOTALL)
# Most SDL_PIXELFORMAT names need their values scrubbed.
RE_PIXELFORMAT = re.compile(r"(?P<name>SDL_PIXELFORMAT_\w+) =[^,}]*")
# Most SDLK names need their values scrubbed.
RE_SDLK = re.compile(r"(?P<name>SDLK_\w+) =.*?(?=,\n|}\n)")
# Remove compile time assertions from the cdef.
RE_ASSERT = re.compile(r"^.*SDL_COMPILE_TIME_ASSERT.*$", re.MULTILINE)
# Padding values need to be scrubbed.
RE_PADDING = re.compile(r"padding\[[^;]*\];")

# These structs have an unusual size when packed by SDL on 32-bit platforms.
FLEXIBLE_STRUCTS = (
    "SDL_CommonEvent",
    "SDL_DisplayEvent",
    "SDL_WindowEvent",
    "SDL_KeyboardDeviceEvent",
    "SDL_KeyboardEvent",
    "SDL_TextEditingEvent",
    "SDL_TextEditingCandidatesEvent",
    "SDL_TextInputEvent",
    "SDL_MouseDeviceEvent",
    "SDL_MouseMotionEvent",
    "SDL_MouseButtonEvent",
    "SDL_MouseWheelEvent",
    "SDL_JoyAxisEvent",
    "SDL_JoyBallEvent",
    "SDL_JoyHatEvent",
    "SDL_JoyButtonEvent",
    "SDL_JoyDeviceEvent",
    "SDL_JoyBatteryEvent",
    "SDL_GamepadAxisEvent",
    "SDL_GamepadButtonEvent",
    "SDL_GamepadDeviceEvent",
    "SDL_GamepadTouchpadEvent",
    "SDL_GamepadSensorEvent",
    "SDL_AudioDeviceEvent",
    "SDL_CameraDeviceEvent",
    "SDL_RenderEvent",
    "SDL_TouchFingerEvent",
    "SDL_PenProximityEvent",
    "SDL_PenMotionEvent",
    "SDL_PenTouchEvent",
    "SDL_PenButtonEvent",
    "SDL_PenAxisEvent",
    "SDL_DropEvent",
    "SDL_ClipboardEvent",
    "SDL_SensorEvent",
    "SDL_QuitEvent",
    "SDL_UserEvent",
)

# Other defined names which sometimes cause issues when parsed.
IGNORE_DEFINES = frozenset(
    (
        "SDL_DEPRECATED",
        "SDL_INLINE",
        "SDL_FORCE_INLINE",
        "SDL_FALLTHROUGH",
        "SDL_HAS_FALLTHROUGH",
        "SDL_NO_THREAD_SAFETY_ANALYSIS",
        "SDL_SCOPED_CAPABILITY",
        "SDL_NODISCARD",
        "SDL_NOLONGLONG",
        "SDL_WINAPI_FAMILY_PHONE",
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
    """Check the local SDL3 version on Linux distributions."""
    if not sys.platform.startswith("linux"):
        return
    needed_version = f"{SDL_MIN_VERSION[0]}.{SDL_MIN_VERSION[1]}.{SDL_MIN_VERSION[2]}"
    try:
        sdl_version_str = subprocess.check_output(
            ["pkg-config", "sdl3", "--modversion"], universal_newlines=True
        ).strip()
    except FileNotFoundError:
        try:
            sdl_version_str = subprocess.check_output(["sdl3-config", "--version"], universal_newlines=True).strip()
        except FileNotFoundError as exc:
            msg = (
                f"libsdl3-dev or equivalent must be installed on your system and must be at least version {needed_version}."
                "\nsdl3-config must be on PATH."
            )
            raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        if sys.version_info >= (3, 11):
            exc.add_note(f"Note: {os.environ.get('PKG_CONFIG_PATH')=}")
        raise
    logger.info(f"Found SDL {sdl_version_str}.")
    sdl_version = tuple(int(s) for s in sdl_version_str.split("."))
    if sdl_version < SDL_MIN_VERSION:
        msg = f"SDL version must be at least {needed_version}, (found {sdl_version_str})"
        raise RuntimeError(msg)


def get_sdl_file(version: str) -> Path:
    """Return a path to an SDL3 archive for the current platform.  The archive is downloaded if missing."""
    if sys.platform == "win32":
        sdl_archive = f"SDL3-devel-{version}-VC.zip"
    else:
        assert sys.platform == "darwin"
        sdl_archive = f"SDL3-{version}.dmg"
    sdl_local_file = Path("dependencies", sdl_archive)
    sdl_remote_url = f"https://www.libsdl.org/release/{sdl_archive}"
    if not sdl_local_file.exists():
        logger.info(f"Downloading {sdl_remote_url}")
        Path("dependencies/").mkdir(parents=True, exist_ok=True)
        with requests.get(sdl_remote_url) as response:  # noqa: S113
            response.raise_for_status()
            sdl_local_file.write_bytes(response.content)
    return sdl_local_file


def unpack_sdl(version: str) -> Path:
    """Return the path to an extracted SDL distribution.  Creates it if missing."""
    sdl_path = Path(f"dependencies/SDL3-{version}")
    if sys.platform == "darwin":
        sdl_dir = sdl_path
        sdl_path /= "SDL3.framework"
    if sdl_path.exists():
        return sdl_path
    sdl_archive = get_sdl_file(version)
    logger.info(f"Extracting {sdl_archive}")
    if sdl_archive.suffix == ".zip":
        with zipfile.ZipFile(sdl_archive) as zf:
            zf.extractall("dependencies/")
    elif sys.platform == "darwin":
        assert sdl_archive.suffix == ".dmg"
        subprocess.check_call(["hdiutil", "mount", sdl_archive])
        subprocess.check_call(["mkdir", "-p", sdl_dir])
        subprocess.check_call(["cp", "-r", "/Volumes/SDL3/SDL3.xcframework/macos-arm64_x86_64/SDL3.framework", sdl_dir])
        subprocess.check_call(["hdiutil", "unmount", "/Volumes/SDL3"])
    return sdl_path


class SDLParser(pcpp.Preprocessor):  # type: ignore[misc]
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

    def on_file_open(self, is_system_include: bool, includepath: str) -> Any:  # noqa: ANN401, FBT001
        """Ignore includes other than SDL headers."""
        if not Path(includepath).parent.name == "SDL3":
            raise FileNotFoundError
        return super().on_file_open(is_system_include, includepath)

    def on_include_not_found(self, is_malformed: bool, is_system_include: bool, curdir: str, includepath: str) -> None:  # noqa: ARG002, FBT001
        """Remove bad includes such as stddef.h and stdarg.h."""
        assert "SDL3/SDL" not in includepath, (includepath, curdir)
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
        if tokens[0].value.startswith("SDL_PLATFORM_"):
            return False  # Ignore platform definitions
        return bool(str(tokens[0].value).startswith(("SDL_", "SDLK_")))

    def on_directive_handle(
        self,
        directive: Any,  # noqa: ANN401
        tokens: list[Any],
        if_passthru: bool,  # noqa: FBT001
        preceding_tokens: list[Any],
    ) -> Any:  # noqa: ANN401
        """Catch and store definitions."""
        if directive.value == "define" and self._should_track_define(tokens):
            if tokens[2].type == "CPP_STRING":
                self.known_string_defines[tokens[0].value] = tokens[2].value
            elif tokens[2].value in self.known_string_defines:
                self.known_string_defines[tokens[0].value] = "..."
            else:
                self.known_defines.add(tokens[0].value)
        return super().on_directive_handle(directive, tokens, if_passthru, preceding_tokens)


def get_emscripten_include_dir() -> Path:
    """Find and return the Emscripten include dir."""
    # None of the EMSDK environment variables exist! Search PATH for Emscripten as a workaround
    for path in os.environ["PATH"].split(os.pathsep)[::-1]:
        if Path(path).match("upstream/emscripten"):
            return Path(path, "system/include").resolve(strict=True)
    raise AssertionError(os.environ["PATH"])


check_sdl_version()

SDL_PARSE_PATH: Path | None = None
SDL_BUNDLE_PATH: Path | None = None
if (sys.platform == "win32" or sys.platform == "darwin") and "PYODIDE" not in os.environ:
    SDL_PARSE_PATH = unpack_sdl(SDL_PARSE_VERSION)
    SDL_BUNDLE_PATH = unpack_sdl(SDL_BUNDLE_VERSION)

SDL_INCLUDE: Path
if sys.platform == "win32" and SDL_PARSE_PATH is not None:
    SDL_INCLUDE = SDL_PARSE_PATH / "include"
elif sys.platform == "darwin" and SDL_PARSE_PATH is not None:
    SDL_INCLUDE = SDL_PARSE_PATH / "Versions/A/Headers"
else:  # Unix
    matches = re.findall(
        r"-I(\S+)",
        subprocess.check_output(["pkg-config", "sdl3", "--cflags"], universal_newlines=True),
    )
    if not matches:
        matches = ["/usr/include"]

    for match in matches:
        if Path(match, "SDL3/SDL.h").is_file():
            SDL_INCLUDE = Path(match)
            break
    else:
        raise AssertionError(matches)
    assert SDL_INCLUDE

logger.info(f"{SDL_INCLUDE=}")

EXTRA_CDEF = """
#define SDLK_SCANCODE_MASK ...

extern "Python" {
// SDL_AudioCallback callback.
void _sdl_audio_stream_callback(void* userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);
// SDL to Python log function.
void _sdl_log_output_function(void *userdata, int category, SDL_LogPriority priority, const char *message);
// Generic event watcher callback.
int _sdl_event_watcher(void* userdata, SDL_Event* event);
}
"""


def get_cdef() -> tuple[str, dict[str, str]]:
    """Return the parsed code of SDL for CFFI."""
    with TemporaryDirectory() as temp_dir:
        # Add a false SDL_oldnames.h to prevent old symbols from being collected
        fake_header_dir = Path(temp_dir, "SDL3")
        fake_header_dir.mkdir()
        (fake_header_dir / "SDL_oldnames.h").write_text("")

        parser = SDLParser()
        parser.add_path(temp_dir)
        parser.add_path(SDL_INCLUDE)
        if Path(SDL_INCLUDE, "../Headers/SDL.h").exists():  # Using MacOS dmg archive
            fake_sdl_dir = Path(SDL_INCLUDE, "SDL3")
            if not fake_sdl_dir.exists():
                fake_sdl_dir.mkdir(exist_ok=False)
                for file in SDL_INCLUDE.glob("SDL*.h"):
                    shutil.copyfile(file, fake_sdl_dir / file.name)
        else:  # Regular path
            assert Path(SDL_INCLUDE, "SDL3/SDL.h").exists(), SDL_INCLUDE
        parser.parse(
            """
        // Remove extern keyword.
        #define extern
        // Ignore some SDL assert statements.
        #define DOXYGEN_SHOULD_IGNORE_THIS
        #define SDL_COMPILE_TIME_ASSERT(x, y)

        #define _SIZE_T_DEFINED_
        typedef int... size_t;
        #define bool _Bool

        #define SDL_oldnames_h_

        #include <SDL3/SDL.h>
        """
        )
        sdl_cdef = parser.get_output()

    sdl_cdef = sdl_cdef.replace("_Bool", "bool")
    sdl_cdef = RE_VAFUNC.sub("", sdl_cdef)
    sdl_cdef = RE_INLINE.sub("", sdl_cdef)
    sdl_cdef = RE_PIXELFORMAT.sub(r"\g<name> = ...", sdl_cdef)
    sdl_cdef = RE_SDLK.sub(r"\g<name> = ...", sdl_cdef)
    sdl_cdef = RE_NEWLINES.sub("\n", sdl_cdef)
    sdl_cdef = RE_PADDING.sub("padding[...];", sdl_cdef)
    sdl_cdef = sdl_cdef.replace("typedef unsigned int uintptr_t;", "typedef int... uintptr_t;").replace(
        "typedef unsigned int size_t;", "typedef int... size_t;"
    )
    for name in FLEXIBLE_STRUCTS:
        sdl_cdef = sdl_cdef.replace(f"}} {name};", f"...;}} {name};")
    return sdl_cdef + EXTRA_CDEF, parser.known_string_defines


include_dirs: list[str] = []
extra_compile_args: list[str] = []
extra_link_args: list[str] = []

libraries: list[str] = []
library_dirs: list[str] = []

if "PYODIDE" in os.environ:
    pass
elif sys.platform == "darwin":
    extra_link_args += ["-framework", "SDL3"]
else:
    libraries += ["SDL3"]

# Bundle the Windows SDL DLL.
if sys.platform == "win32" and SDL_BUNDLE_PATH is not None:
    include_dirs.append(str(SDL_INCLUDE))
    ARCH_MAPPING = {"32bit": "x86", "64bit": "x64"}
    SDL_LIB_DIR = Path(SDL_BUNDLE_PATH, "lib/", ARCH_MAPPING[BIT_SIZE])
    library_dirs.append(str(SDL_LIB_DIR))
    SDL_LIB_DEST = Path("tcod", ARCH_MAPPING[BIT_SIZE])
    SDL_LIB_DEST.mkdir(exist_ok=True)
    SDL_LIB_DEST_FILE = SDL_LIB_DEST / "SDL3.dll"
    SDL_LIB_FILE = SDL_LIB_DIR / "SDL3.dll"
    if not SDL_LIB_DEST_FILE.exists() or SDL_LIB_FILE.read_bytes() != SDL_LIB_DEST_FILE.read_bytes():
        shutil.copy(SDL_LIB_FILE, SDL_LIB_DEST_FILE)

# Link to the SDL framework on MacOS.
# Delocate will bundle the binaries in a later step.
if sys.platform == "darwin" and SDL_BUNDLE_PATH is not None:
    include_dirs.append(SDL_INCLUDE)
    extra_link_args += [f"-F{SDL_BUNDLE_PATH}/.."]
    extra_link_args += ["-rpath", f"{SDL_BUNDLE_PATH}/.."]
    extra_link_args += ["-rpath", "/usr/local/opt/llvm/lib/"]

if "PYODIDE" in os.environ:
    extra_compile_args += ["--use-port=sdl3"]
elif sys.platform not in ["win32", "darwin"]:
    # Use sdl-config to link to SDL on Linux.
    extra_compile_args += (
        subprocess.check_output(["pkg-config", "sdl3", "--cflags"], universal_newlines=True).strip().split()
    )
    extra_link_args += (
        subprocess.check_output(["pkg-config", "sdl3", "--libs"], universal_newlines=True).strip().split()
    )
