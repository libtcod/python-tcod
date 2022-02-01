from __future__ import annotations

import enum
from typing import Any, Tuple

from tcod.loader import ffi, lib
from tcod.sdl import _check


class Subsystem(enum.IntFlag):
    TIMER = lib.SDL_INIT_TIMER or 0x00000001
    AUDIO = lib.SDL_INIT_AUDIO or 0x00000010
    VIDEO = lib.SDL_INIT_VIDEO or 0x00000020
    JOYSTICK = lib.SDL_INIT_JOYSTICK or 0x00000200
    HAPTIC = lib.SDL_INIT_HAPTIC or 0x00001000
    GAMECONTROLLER = lib.SDL_INIT_GAMECONTROLLER or 0x00002000
    EVENTS = lib.SDL_INIT_EVENTS or 0x00004000
    SENSOR = getattr(lib, "SDL_INIT_SENSOR", None) or 0x00008000  # SDL >= 2.0.9
    EVERYTHING = lib.SDL_INIT_EVERYTHING or 0


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

    def __exit__(self, *args: Any) -> None:
        self.close()


class _PowerState(enum.IntEnum):
    UNKNOWN = getattr(lib, "SDL_POWERSTATE_UNKNOWN", 0)
    ON_BATTERY = getattr(lib, "SDL_POWERSTATE_ON_BATTERY", 0)
    NO_BATTERY = getattr(lib, "SDL_POWERSTATE_NO_BATTERY", 0)
    CHARGING = getattr(lib, "SDL_POWERSTATE_CHARGING", 0)
    CHARGED = getattr(lib, "SDL_POWERSTATE_CHARGED", 0)


def _get_power_info() -> Tuple[_PowerState, int, int]:
    buffer = ffi.new("int[2]")
    power_state = _PowerState(lib.SDL_GetPowerInfo(buffer, buffer + 1))
    seconds_of_power = buffer[0]
    percenage = buffer[1]
    return power_state, seconds_of_power, percenage
