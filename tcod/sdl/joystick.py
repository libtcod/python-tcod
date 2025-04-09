"""SDL Joystick Support.

.. versionadded:: 13.8
"""

from __future__ import annotations

import enum
from typing import Any, ClassVar, Final, Literal
from weakref import WeakValueDictionary

import tcod.sdl.sys
from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _check_int, _check_p

_HAT_DIRECTIONS: dict[int, tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]] = {
    int(lib.SDL_HAT_CENTERED): (0, 0),
    int(lib.SDL_HAT_UP): (0, -1),
    int(lib.SDL_HAT_RIGHT): (1, 0),
    int(lib.SDL_HAT_DOWN): (0, 1),
    int(lib.SDL_HAT_LEFT): (-1, 0),
    int(lib.SDL_HAT_RIGHTUP): (1, -1),
    int(lib.SDL_HAT_RIGHTDOWN): (1, 1),
    int(lib.SDL_HAT_LEFTUP): (-1, -1),
    int(lib.SDL_HAT_LEFTDOWN): (-1, 1),
}


class ControllerAxis(enum.IntEnum):
    """The standard axes for a game controller."""

    INVALID = int(lib.SDL_GAMEPAD_AXIS_INVALID)
    LEFTX = int(lib.SDL_GAMEPAD_AXIS_LEFTX)
    """"""
    LEFTY = int(lib.SDL_GAMEPAD_AXIS_LEFTY)
    """"""
    RIGHTX = int(lib.SDL_GAMEPAD_AXIS_RIGHTX)
    """"""
    RIGHTY = int(lib.SDL_GAMEPAD_AXIS_RIGHTY)
    """"""
    TRIGGERLEFT = int(lib.SDL_GAMEPAD_AXIS_LEFT_TRIGGER)
    """"""
    TRIGGERRIGHT = int(lib.SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
    """"""


class ControllerButton(enum.IntEnum):
    """The standard buttons for a game controller."""

    INVALID = int(lib.SDL_GAMEPAD_BUTTON_INVALID)
    A = int(lib.SDL_GAMEPAD_BUTTON_SOUTH)
    """"""
    B = int(lib.SDL_GAMEPAD_BUTTON_EAST)
    """"""
    X = int(lib.SDL_GAMEPAD_BUTTON_WEST)
    """"""
    Y = int(lib.SDL_GAMEPAD_BUTTON_NORTH)
    """"""
    BACK = int(lib.SDL_GAMEPAD_BUTTON_BACK)
    """"""
    GUIDE = int(lib.SDL_GAMEPAD_BUTTON_GUIDE)
    """"""
    START = int(lib.SDL_GAMEPAD_BUTTON_START)
    """"""
    LEFTSTICK = int(lib.SDL_GAMEPAD_BUTTON_LEFT_STICK)
    """"""
    RIGHTSTICK = int(lib.SDL_GAMEPAD_BUTTON_RIGHT_STICK)
    """"""
    LEFTSHOULDER = int(lib.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER)
    """"""
    RIGHTSHOULDER = int(lib.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)
    """"""
    DPAD_UP = int(lib.SDL_GAMEPAD_BUTTON_DPAD_UP)
    """"""
    DPAD_DOWN = int(lib.SDL_GAMEPAD_BUTTON_DPAD_DOWN)
    """"""
    DPAD_LEFT = int(lib.SDL_GAMEPAD_BUTTON_DPAD_LEFT)
    """"""
    DPAD_RIGHT = int(lib.SDL_GAMEPAD_BUTTON_DPAD_RIGHT)
    """"""
    MISC1 = 15
    """"""
    PADDLE1 = 16
    """"""
    PADDLE2 = 17
    """"""
    PADDLE3 = 18
    """"""
    PADDLE4 = 19
    """"""
    TOUCHPAD = 20
    """"""


class Joystick:
    """A low-level SDL joystick.

    .. seealso::
        https://wiki.libsdl.org/CategoryJoystick
    """

    _by_instance_id: ClassVar[WeakValueDictionary[int, Joystick]] = WeakValueDictionary()
    """Currently opened joysticks."""

    def __init__(self, sdl_joystick_p: Any) -> None:  # noqa: ANN401
        self.sdl_joystick_p: Final = sdl_joystick_p
        """The CFFI pointer to an SDL_Joystick struct."""
        self.axes: Final[int] = _check_int(lib.SDL_GetNumJoystickAxes(self.sdl_joystick_p), failure=-1)
        """The total number of axes."""
        self.balls: Final[int] = _check_int(lib.SDL_GetNumJoystickBalls(self.sdl_joystick_p), failure=-1)
        """The total number of trackballs."""
        self.buttons: Final[int] = _check_int(lib.SDL_GetNumJoystickButtons(self.sdl_joystick_p), failure=-1)
        """The total number of buttons."""
        self.hats: Final[int] = _check_int(lib.SDL_GetNumJoystickHats(self.sdl_joystick_p), failure=-1)
        """The total number of hats."""
        self.name: Final[str] = str(ffi.string(lib.SDL_GetJoystickName(self.sdl_joystick_p)), encoding="utf-8")
        """The name of this joystick."""
        self.guid: Final[str] = self._get_guid()
        """The GUID of this joystick."""
        self.id: Final[int] = _check(lib.SDL_GetJoystickID(self.sdl_joystick_p))
        """The instance ID of this joystick.  This is not the same as the device ID."""
        self._keep_alive: Any = None
        """The owner of this objects memory if this object does not own itself."""

        self._by_instance_id[self.id] = self

    @classmethod
    def _open(cls, device_index: int) -> Joystick:
        tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.JOYSTICK)
        p = _check_p(ffi.gc(lib.SDL_OpenJoystick(device_index), lib.SDL_CloseJoystick))
        return cls(p)

    @classmethod
    def _from_instance_id(cls, instance_id: int) -> Joystick:
        return cls._by_instance_id[instance_id]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Joystick):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)

    def _get_guid(self) -> str:
        guid_str = ffi.new("char[33]")
        lib.SDL_GUIDToString(lib.SDL_GetJoystickGUID(self.sdl_joystick_p), guid_str, len(guid_str))
        return str(ffi.string(guid_str), encoding="ascii")

    def get_axis(self, axis: int) -> int:
        """Return the raw value of `axis` in the range -32768 to 32767."""
        return int(lib.SDL_GetJoystickAxis(self.sdl_joystick_p, axis))

    def get_ball(self, ball: int) -> tuple[int, int]:
        """Return the values (delta_x, delta_y) of `ball` since the last poll."""
        xy = ffi.new("int[2]")
        _check(lib.SDL_GetJoystickBall(self.sdl_joystick_p, ball, xy, xy + 1))
        return int(xy[0]), int(xy[1])

    def get_button(self, button: int) -> bool:
        """Return True if `button` is currently held."""
        return bool(lib.SDL_GetJoystickButton(self.sdl_joystick_p, button))

    def get_hat(self, hat: int) -> tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]:
        """Return the direction of `hat` as (x, y).  With (-1, -1) being in the upper-left."""
        return _HAT_DIRECTIONS[lib.SDL_GetJoystickHat(self.sdl_joystick_p, hat)]


class GameController:
    """A standard interface for an Xbox 360 style game controller."""

    _by_instance_id: ClassVar[WeakValueDictionary[int, GameController]] = WeakValueDictionary()
    """Currently opened controllers."""

    def __init__(self, sdl_controller_p: Any) -> None:  # noqa: ANN401
        self.sdl_controller_p: Final = sdl_controller_p
        self.joystick: Final = Joystick(lib.SDL_GetGamepadJoystick(self.sdl_controller_p))
        """The :any:`Joystick` associated with this controller."""
        self.joystick._keep_alive = self.sdl_controller_p  # This objects real owner needs to be kept alive.
        self._by_instance_id[self.joystick.id] = self

    @classmethod
    def _open(cls, joystick_index: int) -> GameController:
        return cls(_check_p(ffi.gc(lib.SDL_OpenGamepad(joystick_index), lib.SDL_CloseGamepad)))

    @classmethod
    def _from_instance_id(cls, instance_id: int) -> GameController:
        return cls._by_instance_id[instance_id]

    def get_button(self, button: ControllerButton) -> bool:
        """Return True if `button` is currently held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, button))

    def get_axis(self, axis: ControllerAxis) -> int:
        """Return the state of the given `axis`.

        The state is usually a value from -32768 to 32767, with positive values towards the lower-right direction.
        Triggers have the range of 0 to 32767 instead.
        """
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, axis))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GameController):
            return self.joystick.id == other.joystick.id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.joystick.id)

    # These could exist as convenience functions, but the get_X functions are probably better.
    @property
    def _left_x(self) -> int:
        """Return the position of this axis (-32768 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_LEFTX))

    @property
    def _left_y(self) -> int:
        """Return the position of this axis (-32768 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_LEFTY))

    @property
    def _right_x(self) -> int:
        """Return the position of this axis (-32768 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_RIGHTX))

    @property
    def _right_y(self) -> int:
        """Return the position of this axis (-32768 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_RIGHTY))

    @property
    def _trigger_left(self) -> int:
        """Return the position of this trigger (0 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_LEFT_TRIGGER))

    @property
    def _trigger_right(self) -> int:
        """Return the position of this trigger (0 to 32767)."""
        return int(lib.SDL_GetGamepadAxis(self.sdl_controller_p, lib.SDL_GAMEPAD_AXIS_RIGHT_TRIGGER))

    @property
    def _a(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_SOUTH))

    @property
    def _b(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_EAST))

    @property
    def _x(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_WEST))

    @property
    def _y(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_NORTH))

    @property
    def _back(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_BACK))

    @property
    def _guide(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_GUIDE))

    @property
    def _start(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_START))

    @property
    def _left_stick(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_LEFT_STICK))

    @property
    def _right_stick(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_RIGHT_STICK))

    @property
    def _left_shoulder(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER))

    @property
    def _right_shoulder(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER))

    @property
    def _dpad(self) -> tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]:
        return (
            lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_DPAD_RIGHT)
            - lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_DPAD_LEFT),
            lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_DPAD_DOWN)
            - lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_DPAD_UP),
        )  # type: ignore[return-value] # Boolean math has predictable values

    @property
    def _misc1(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_MISC1))

    @property
    def _paddle1(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1))

    @property
    def _paddle2(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_LEFT_PADDLE1))

    @property
    def _paddle3(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2))

    @property
    def _paddle4(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_LEFT_PADDLE2))

    @property
    def _touchpad(self) -> bool:
        """Return True if this button is held."""
        return bool(lib.SDL_GetGamepadButton(self.sdl_controller_p, lib.SDL_GAMEPAD_BUTTON_TOUCHPAD))


def init() -> None:
    """Initialize SDL's joystick and game controller subsystems."""
    CONTROLLER_SYSTEMS = tcod.sdl.sys.Subsystem.JOYSTICK | tcod.sdl.sys.Subsystem.GAMECONTROLLER
    if tcod.sdl.sys.Subsystem(lib.SDL_WasInit(CONTROLLER_SYSTEMS)) == CONTROLLER_SYSTEMS:
        return  # Already initialized
    tcod.sdl.sys.init(CONTROLLER_SYSTEMS)


def _get_number() -> int:
    """Return the number of attached joysticks."""
    init()
    count = ffi.new("int*")
    lib.SDL_GetJoysticks(count)
    return int(count[0])


def get_joysticks() -> list[Joystick]:
    """Return a list of all connected joystick devices."""
    return [Joystick._open(i) for i in range(_get_number())]


def get_controllers() -> list[GameController]:
    """Return a list of all connected game controllers.

    This ignores joysticks without a game controller mapping.
    """
    return [GameController._open(i) for i in range(_get_number()) if lib.SDL_IsGamepad(i)]


def _get_all() -> list[Joystick | GameController]:
    """Return a list of all connected joystick or controller devices.

    If the joystick has a controller mapping then it is returned as a :any:`GameController`.
    Otherwise it is returned as a :any:`Joystick`.
    """
    return [GameController._open(i) if lib.SDL_IsGamepad(i) else Joystick._open(i) for i in range(_get_number())]


def joystick_event_state(new_state: bool | None = None) -> bool:
    """Check or set joystick event polling.

    .. seealso::
        https://wiki.libsdl.org/SDL3/SDL_SetJoystickEventsEnabled
    """
    if new_state is True:
        lib.SDL_SetJoystickEventsEnabled(True)  # noqa: FBT003
    elif new_state is False:
        lib.SDL_SetJoystickEventsEnabled(False)  # noqa: FBT003
    return lib.SDL_JoystickEventsEnabled()


def controller_event_state(new_state: bool | None = None) -> bool:
    """Check or set game controller event polling.

    .. seealso::
        https://wiki.libsdl.org/SDL3/SDL_SetGamepadEventsEnabled
    """
    if new_state is True:
        lib.SDL_SetGamepadEventsEnabled(True)  # noqa: FBT003
    elif new_state is False:
        lib.SDL_SetGamepadEventsEnabled(False)  # noqa: FBT003
    return lib.SDL_GamepadEventsEnabled()
