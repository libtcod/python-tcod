"""SDL Joystick Support

.. versionadded:: Unreleased
"""
from __future__ import annotations

import enum
from typing import Dict, List, Optional, Tuple

from typing_extensions import Final, Literal

import tcod.sdl.sys
from tcod.loader import ffi, lib
from tcod.sdl import _check, _check_p

_HAT_DIRECTIONS: Dict[int, Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]] = {
    lib.SDL_HAT_CENTERED or 0: (0, 0),
    lib.SDL_HAT_UP or 0: (0, -1),
    lib.SDL_HAT_RIGHT or 0: (1, 0),
    lib.SDL_HAT_DOWN or 0: (0, 1),
    lib.SDL_HAT_LEFT or 0: (-1, 0),
    lib.SDL_HAT_RIGHTUP or 0: (1, -1),
    lib.SDL_HAT_RIGHTDOWN or 0: (1, 1),
    lib.SDL_HAT_LEFTUP or 0: (-1, -1),
    lib.SDL_HAT_LEFTDOWN or 0: (-1, 1),
}


class Power(enum.IntEnum):
    """The possible power states of a controller.

    .. seealso::
        :any:`Joystick.get_current_power`
    """

    UNKNOWN = lib.SDL_JOYSTICK_POWER_UNKNOWN or -1
    """Power state is unknown."""
    EMPTY = lib.SDL_JOYSTICK_POWER_EMPTY or 0
    """<= 5% power."""
    LOW = lib.SDL_JOYSTICK_POWER_LOW or 1
    """<= 20% power."""
    MEDIUM = lib.SDL_JOYSTICK_POWER_MEDIUM or 2
    """<= 70% power."""
    FULL = lib.SDL_JOYSTICK_POWER_FULL or 3
    """<= 100% power."""
    WIRED = lib.SDL_JOYSTICK_POWER_WIRED or 4
    """"""
    MAX = lib.SDL_JOYSTICK_POWER_MAX or 5
    """"""


class Joystick:
    """An SDL joystick.

    .. seealso::
        https://wiki.libsdl.org/CategoryJoystick
    """

    def __init__(self, device_index: int):
        tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.JOYSTICK)
        self.sdl_joystick_p: Final = _check_p(ffi.gc(lib.SDL_JoystickOpen(device_index), lib.SDL_JoystickClose))
        self.axes: Final = _check(lib.SDL_JoystickNumAxes(self.sdl_joystick_p))
        self.balls: Final = _check(lib.SDL_JoystickNumBalls(self.sdl_joystick_p))
        self.buttons: Final = _check(lib.SDL_JoystickNumButtons(self.sdl_joystick_p))
        self.hats: Final = _check(lib.SDL_JoystickNumHats(self.sdl_joystick_p))
        self.name: Final = str(ffi.string(lib.SDL_JoystickName(self.sdl_joystick_p)), encoding="utf-8")
        """The name of this joystick."""
        self.guid: Final = self._get_guid()
        """The GUID of this joystick."""
        self.id: Final = _check(lib.SDL_JoystickInstanceID(self.sdl_joystick_p))
        """The instance ID of this joystick.  This is not the same as the device ID."""

    def _get_guid(self) -> str:
        guid_str = ffi.new("char[33]")
        lib.SDL_JoystickGetGUIDString(lib.SDL_JoystickGetGUID(self.sdl_joystick_p), guid_str, len(guid_str))
        return str(tcod.ffi.string(guid_str), encoding="utf-8")

    def get_current_power(self) -> Power:
        """Return the power level/state of this joystick.  See :any:`Power`."""
        return Power(lib.SDL_JoystickCurrentPowerLevel(self.sdl_joystick_p))

    def get_axis(self, axis: int) -> int:
        """Return the raw value of `axis` in the range -32768 to 32767."""
        return int(lib.SDL_JoystickGetAxis(self.sdl_joystick_p, axis))

    def get_ball(self, ball: int) -> Tuple[int, int]:
        """Return the values (delta_x, delta_y) of `ball` since the last poll."""
        xy = ffi.new("int[2]")
        _check(lib.SDL_JoystickGetBall(ball, xy, xy + 1))
        return int(xy[0]), int(xy[1])

    def get_button(self, button: int) -> bool:
        """Return True if `button` is pressed."""
        return bool(lib.SDL_JoystickGetButton(self.sdl_joystick_p, button))

    def get_hat(self, hat: int) -> Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]:
        """Return the direction of `hat` as (x, y).  With (-1, -1) being in the upper-left."""
        return _HAT_DIRECTIONS[lib.SDL_JoystickGetHat(self.sdl_joystick_p, hat)]


def get_number() -> int:
    """Return the number of attached joysticks."""
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.JOYSTICK)
    return _check(lib.SDL_NumJoysticks())


def get_joysticks() -> List[Joystick]:
    """Return a list of all connected joystick devices."""
    return [Joystick(i) for i in range(get_number())]


def event_state(new_state: Optional[bool] = None) -> bool:
    """Check or set joystick event polling.

    .. seealso::
        https://wiki.libsdl.org/SDL_JoystickEventState
    """
    _OPTIONS = {None: lib.SDL_QUERY, False: lib.SDL_IGNORE, True: lib.SDL_ENABLE}
    return bool(_check(lib.SDL_JoystickEventState(_OPTIONS[new_state])))
