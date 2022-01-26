from __future__ import annotations

import enum
from typing import Any, Tuple

from tcod.loader import ffi, lib


class Subsystem(enum.IntFlag):
    TIMER = lib.SDL_INIT_TIMER
    AUDIO = lib.SDL_INIT_AUDIO
    VIDEO = lib.SDL_INIT_VIDEO
    JOYSTICK = lib.SDL_INIT_JOYSTICK
    HAPTIC = lib.SDL_INIT_HAPTIC
    GAMECONTROLLER = lib.SDL_INIT_GAMECONTROLLER
    EVENTS = lib.SDL_INIT_EVENTS
    SENSOR = getattr(lib, "SDL_INIT_SENSOR", 0)
    EVERYTHING = lib.SDL_INIT_EVERYTHING


def _check(result: int) -> int:
    if result < 0:
        raise RuntimeError(_get_error())
    return result


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


def _get_error() -> str:
    return str(ffi.string(lib.SDL_GetError()), encoding="utf-8")


class _PowerState(enum.IntEnum):
    UNKNOWN = lib.SDL_POWERSTATE_UNKNOWN
    ON_BATTERY = lib.SDL_POWERSTATE_ON_BATTERY
    NO_BATTERY = lib.SDL_POWERSTATE_NO_BATTERY
    CHARGING = lib.SDL_POWERSTATE_CHARGING
    CHARGED = lib.SDL_POWERSTATE_CHARGED


def _get_power_info() -> Tuple[_PowerState, int, int]:
    buffer = ffi.new("int[2]")
    power_state = _PowerState(lib.SDL_GetPowerInfo(buffer, buffer + 1))
    seconds_of_power = buffer[0]
    percenage = buffer[1]
    return power_state, seconds_of_power, percenage
