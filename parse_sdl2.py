import os.path
import sys

import platform
import re
from typing import Iterator

import cffi  # type: ignore

# Various poorly made regular expressions, these will miss code which isn't
# supported by cffi.
RE_COMMENT = re.compile(r" */\*.*?\*/", re.DOTALL)
RE_REMOVALS = re.compile(
    r"#ifndef DOXYGEN_SHOULD_IGNORE_THIS.*"
    r"#endif /\* DOXYGEN_SHOULD_IGNORE_THIS \*/",
    re.DOTALL,
)
RE_DEFINE = re.compile(r"#define \w+(?!\() (?:.*?(?:\\\n)?)*$", re.MULTILINE)
RE_TYPEDEF = re.compile(r"^typedef[^{;#]*?(?:{[^}]*\n}[^;]*)?;", re.MULTILINE)
RE_ENUM = re.compile(r"^enum[^{;#]*?(?:{[^}]*\n}[^;]*)?;", re.MULTILINE)
RE_DECL = re.compile(r"^extern[^#\n]*\([^#]*?\);$", re.MULTILINE | re.DOTALL)
RE_ENDIAN = re.compile(
    r"#if SDL_BYTEORDER == SDL_LIL_ENDIAN(.*?)#else(.*?)#endif", re.DOTALL
)
RE_ENDIAN2 = re.compile(
    r"#if SDL_BYTEORDER == SDL_BIG_ENDIAN(.*?)#else(.*?)#endif", re.DOTALL
)
RE_DEFINE_TRUNCATE = re.compile(r"(#define\s+\w+\s+).+$", flags=re.DOTALL)
RE_TYPEDEF_TRUNCATE = re.compile(
    r"(typedef\s+\w+\s+\w+)\s*{.*\n}(?=.*;$)", flags=re.DOTALL | re.MULTILINE
)
RE_ENUM_TRUNCATE = re.compile(
    r"(\w+\s*=).+?(?=,$|})(?![^(']*\))", re.MULTILINE | re.DOTALL
)


def get_header(name: str) -> str:
    """Return the source of a header in a partially preprocessed state."""
    with open(name, "r") as f:
        header = f.read()
    # Remove Doxygen code.
    header = RE_REMOVALS.sub("", header)
    # Remove comments.
    header = RE_COMMENT.sub("", header)
    # Deal with endianness in "SDL_audio.h".
    header = RE_ENDIAN.sub(
        r"\1" if sys.byteorder == "little" else r"\2", header
    )
    header = RE_ENDIAN2.sub(
        r"\1" if sys.byteorder != "little" else r"\2", header
    )

    # Ignore bad ARM compiler typedef.
    header = header.replace("typedef int SDL_bool;", "")
    return header


# Remove non-integer definitions.
DEFINE_BLACKLIST = [
    "SDL_AUDIOCVT_PACKED",
    "SDL_BlitScaled",
    "SDL_BlitSurface",
    "SDL_Colour",
]


def parse(header: str, NEEDS_PACK4: bool) -> Iterator[str]:
    """Pull individual sections from a header, processing them as needed."""
    for define in RE_DEFINE.findall(header):
        if any(item in define for item in DEFINE_BLACKLIST):
            continue  # Remove non-integer definitions.
        if '"' in define:
            continue  # Ignore definitions with strings.
        # Replace various definitions with "..." since cffi is limited here.
        yield RE_DEFINE_TRUNCATE.sub(r"\1 ...", define)

    for typedef in RE_TYPEDEF.findall(header):
        # Special case for SDL window flags enum.
        if "SDL_WINDOW_FULLSCREEN_DESKTOP" in typedef:
            typedef = typedef.replace(
                "( SDL_WINDOW_FULLSCREEN | 0x00001000 )", "..."
            )
        # Detect array sizes at compile time.
        typedef = typedef.replace("SDL_TEXTINPUTEVENT_TEXT_SIZE", "...")
        typedef = typedef.replace("SDL_TEXTEDITINGEVENT_TEXT_SIZE", "...")
        typedef = typedef.replace("SDL_AUDIOCVT_MAX_FILTERS + 1", "...")

        typedef = typedef.replace("SDLCALL ", "")
        typedef = typedef.replace("SDL_AUDIOCVT_PACKED ", "")

        if NEEDS_PACK4 and "typedef struct SDL_AudioCVT" in typedef:
            typedef = RE_TYPEDEF_TRUNCATE.sub(r"\1 { ...; }", typedef)
        if NEEDS_PACK4 and "typedef struct SDL_TouchFingerEvent" in typedef:
            typedef = RE_TYPEDEF_TRUNCATE.sub(r"\1 { ...; }", typedef)
        if NEEDS_PACK4 and "typedef struct SDL_MultiGestureEvent" in typedef:
            typedef = RE_TYPEDEF_TRUNCATE.sub(r"\1 { ...; }", typedef)
        if NEEDS_PACK4 and "typedef struct SDL_DollarGestureEvent" in typedef:
            typedef = RE_TYPEDEF_TRUNCATE.sub(r"\1 { ...; }", typedef)
        yield typedef

    for enum in RE_ENUM.findall(header):
        yield RE_ENUM_TRUNCATE.sub(r"\1 ...", enum)

    for decl in RE_DECL.findall(header):
        if "SDL_RWops" in decl:
            continue  # Ignore SDL_RWops functions.
        if "va_list" in decl:
            continue
        decl = re.sub(r"SDL_PRINTF_VARARG_FUNC\(\w*\)", "", decl)
        decl = decl.replace("extern DECLSPEC ", "")
        decl = decl.replace("SDLCALL ", "")
        yield decl.replace("SDL_PRINTF_FORMAT_STRING ", "")


# Parsed headers excluding "SDL_stdinc.h"
HEADERS = [
    "SDL_rect.h",
    "SDL_pixels.h",
    "SDL_blendmode.h",
    "SDL_error.h",
    "SDL_surface.h",
    "SDL_video.h",
    "SDL_render.h",
    "SDL_audio.h",
    "SDL_clipboard.h",
    "SDL_touch.h",
    "SDL_gesture.h",
    "SDL_hints.h",
    "SDL_joystick.h",
    "SDL_haptic.h",
    "SDL_gamecontroller.h",
    "SDL_power.h",
    "SDL_log.h",
    "SDL_messagebox.h",
    "SDL_mouse.h",
    "SDL_timer.h",
    "SDL_keycode.h",
    "SDL_scancode.h",
    "SDL_keyboard.h",
    "SDL_events.h",
    "SDL.h",
    "SDL_version.h",
]


def add_to_ffi(ffi: cffi.FFI, path: str) -> None:
    BITS, _ = platform.architecture()
    cdef_args = {}
    NEEDS_PACK4 = False
    if sys.platform == "win32" and BITS == "32bit":
        NEEDS_PACK4 = True
        # The following line is required but cffi does not currently support
        # it for ABI mode.
        # cdef_args["pack"] = 4

    ffi.cdef(
        "\n".join(
            RE_TYPEDEF.findall(get_header(os.path.join(path, "SDL_stdinc.h")))
        ).replace("SDLCALL ", ""),
        **cdef_args
    )
    for header in HEADERS:
        try:
            for code in parse(
                get_header(os.path.join(path, header)), NEEDS_PACK4
            ):
                if (
                    "typedef struct SDL_AudioCVT" in code
                    and sys.platform != "win32"
                    and not NEEDS_PACK4
                ):
                    # This specific struct needs to be packed.
                    ffi.cdef(code, packed=1)
                    continue
                ffi.cdef(code, **cdef_args)
        except:
            print("Error parsing %r code:\n%s" % (header, code))
            raise


def get_ffi(path: str) -> cffi.FFI:
    """Return an ffi for SDL2, needs to be compiled."""
    ffi = cffi.FFI()
    add_to_ffi(ffi, path)
    return ffi
