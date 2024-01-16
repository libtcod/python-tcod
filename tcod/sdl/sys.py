from __future__ import annotations

import enum
import warnings

from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _get_error


class Subsystem(enum.IntFlag):
    TIMER = 0x00000001
    AUDIO = 0x00000010
    VIDEO = 0x00000020
    JOYSTICK = 0x00000200
    HAPTIC = 0x00001000
    GAMECONTROLLER = 0x00002000
    EVENTS = 0x00004000
    SENSOR = 0x00008000
    EVERYTHING = int(lib.SDL_INIT_EVERYTHING)


def init(flags: int = Subsystem.EVERYTHING) -> None:
    _check(lib.SDL_InitSubSystem(flags))


def quit(flags: int = Subsystem.EVERYTHING) -> None:
    lib.SDL_QuitSubSystem(flags)


class _ScopeInit:
    def __init__(self, flags: int) -> None:
        init(flags)
        self.flags = flags

    def close(self) -> None:
        if self.flags:
            quit(self.flags)
            self.flags = 0

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> _ScopeInit:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _PowerState(enum.IntEnum):
    UNKNOWN = 0
    ON_BATTERY = enum.auto()
    NO_BATTERY = enum.auto()
    CHARGING = enum.auto()
    CHARGED = enum.auto()


def _get_power_info() -> tuple[_PowerState, int, int]:
    buffer = ffi.new("int[2]")
    power_state = _PowerState(lib.SDL_GetPowerInfo(buffer, buffer + 1))
    seconds_of_power = buffer[0]
    percentage = buffer[1]
    return power_state, seconds_of_power, percentage


def _get_clipboard() -> str:
    """Return the text of the clipboard."""
    text = str(ffi.string(lib.SDL_GetClipboardText()), encoding="utf-8")
    if not text:  # Show the reason for an empty return, this should probably be logged instead.
        warnings.warn(f"Return string is empty because: {_get_error()}", RuntimeWarning, stacklevel=2)
    return text


def _set_clipboard(text: str) -> None:
    """Replace the clipboard with text."""
    _check(lib.SDL_SetClipboardText(text.encode("utf-8")))
