"""A light-weight implementation of event handling built on calls to SDL.

Many event constants are derived directly from SDL.
For example: ``tcod.event.KeySym.UP`` and ``tcod.event.Scancode.A`` refer to
SDL's ``SDLK_UP`` and ``SDL_SCANCODE_A`` respectfully.
`See this table for all of SDL's keyboard constants.
<https://wiki.libsdl.org/SDL_Keycode>`_

Printing any event will tell you its attributes in a human readable format.
An events type attribute if omitted is just the classes name with all letters upper-case.

As a general guideline, you should use :any:`KeyboardEvent.sym` for command inputs,
and :any:`TextInput.text` for name entry fields.

Example::

    import tcod

    KEY_COMMANDS = {
        tcod.event.KeySym.UP: "move N",
        tcod.event.KeySym.DOWN: "move S",
        tcod.event.KeySym.LEFT: "move W",
        tcod.event.KeySym.RIGHT: "move E",
    }

    context = tcod.context.new()
    while True:
        console = context.new_console()
        context.present(console, integer_scaling=True)
        for event in tcod.event.wait():
            context.convert_event(event)  # Adds tile coordinates to mouse events.
            if isinstance(event, tcod.event.Quit):
                print(event)
                raise SystemExit()
            elif isinstance(event, tcod.event.KeyDown):
                print(event)  # Prints the Scancode and KeySym enums for this event.
                if event.sym in KEY_COMMANDS:
                    print(f"Command: {KEY_COMMANDS[event.sym]}")
            elif isinstance(event, tcod.event.MouseButtonDown):
                print(event)  # Prints the mouse button constant names for this event.
            elif isinstance(event, tcod.event.MouseMotion):
                print(event)  # Prints the mouse button mask bits in a readable format.
            else:
                print(event)  # Print any unhandled events.

Python 3.10 introduced `match statements <https://docs.python.org/3/tutorial/controlflow.html#match-statements>`_
which can be used to dispatch events more gracefully:

Example::

    import tcod

    KEY_COMMANDS = {
        tcod.event.KeySym.UP: "move N",
        tcod.event.KeySym.DOWN: "move S",
        tcod.event.KeySym.LEFT: "move W",
        tcod.event.KeySym.RIGHT: "move E",
    }

    context = tcod.context.new()
    while True:
        console = context.new_console()
        context.present(console, integer_scaling=True)
        for event in tcod.event.wait():
            context.convert_event(event)  # Adds tile coordinates to mouse events.
            match event:
                case tcod.event.Quit():
                    raise SystemExit()
                case tcod.event.KeyDown(sym) if sym in KEY_COMMANDS:
                    print(f"Command: {KEY_COMMANDS[sym]}")
                case tcod.event.KeyDown(sym=sym, scancode=scancode, mod=mod, repeat=repeat):
                    print(f"KeyDown: {sym=}, {scancode=}, {mod=}, {repeat=}")
                case tcod.event.MouseButtonDown(button=button, pixel=pixel, tile=tile):
                    print(f"MouseButtonDown: {button=}, {pixel=}, {tile=}")
                case tcod.event.MouseMotion(pixel=pixel, pixel_motion=pixel_motion, tile=tile, tile_motion=tile_motion):
                    print(f"MouseMotion: {pixel=}, {pixel_motion=}, {tile=}, {tile_motion=}")
                case tcod.event.Event() as event:
                    print(event)  # Show any unhandled events.

.. versionadded:: 8.4
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, NamedTuple, TypeVar

import numpy as np
from typing_extensions import deprecated

import tcod.event
import tcod.event_constants
import tcod.sdl.joystick
import tcod.sdl.sys
from tcod.cffi import ffi, lib
from tcod.event_constants import *  # noqa: F403
from tcod.sdl.joystick import _HAT_DIRECTIONS

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class _ConstantsWithPrefix(Mapping[int, str]):
    def __init__(self, constants: Mapping[int, str]) -> None:
        self.constants = constants

    def __getitem__(self, key: int) -> str:
        return "tcod.event." + self.constants[key]

    def __len__(self) -> int:
        return len(self.constants)

    def __iter__(self) -> Iterator[int]:
        return iter(self.constants)


def _describe_bitmask(bits: int, table: Mapping[int, str], default: str = "0") -> str:
    """Return a bitmask in human readable form.

    This is a private function, used internally.

    `bits` is the bitmask to be represented.

    `table` is a reverse lookup table.

    `default` is returned when no other bits can be represented.
    """
    result = []
    for bit, name in table.items():
        if bit & bits:
            result.append(name)
    if not result:
        return default
    return "|".join(result)


def _pixel_to_tile(x: float, y: float) -> tuple[float, float] | None:
    """Convert pixel coordinates to tile coordinates."""
    if not lib.TCOD_ctx.engine:
        return None
    xy = ffi.new("double[2]", (x, y))
    lib.TCOD_sys_pixel_to_tile(xy, xy + 1)
    return xy[0], xy[1]


class Point(NamedTuple):
    """A 2D position used for events with mouse coordinates.

    .. seealso::
        :any:`MouseMotion` :any:`MouseButtonDown` :any:`MouseButtonUp`
    """

    x: float
    """A pixel or tile coordinate starting with zero as the left-most position."""
    y: float
    """A pixel or tile coordinate starting with zero as the top-most position."""


def _verify_tile_coordinates(xy: Point | None) -> Point:
    """Check if an events tile coordinate is initialized and warn if not.

    Always returns a valid Point object for backwards compatibility.
    """
    if xy is not None:
        return xy
    warnings.warn(
        "This events tile coordinates are uninitialized!"
        "\nYou MUST pass this event to `Context.convert_event` before you can"
        " read its tile attributes.",
        RuntimeWarning,
        stacklevel=3,  # Called within other functions, never directly.
    )
    return Point(0, 0)


def _init_sdl_video() -> None:
    """Keyboard layout stuff needs SDL to be initialized first."""
    if lib.SDL_WasInit(lib.SDL_INIT_VIDEO):
        return
    lib.SDL_InitSubSystem(lib.SDL_INIT_VIDEO)


class Modifier(enum.IntFlag):
    """Keyboard modifier flags, a bit-field of held modifier keys.

    Use `bitwise and` to check if a modifier key is held.

    The following example shows some common ways of checking modifiers.
    All non-zero return values are considered true.

    Example::

        >>> mod = tcod.event.Modifier(4098)
        >>> mod & tcod.event.Modifier.SHIFT  # Check if any shift key is held.
        <Modifier.RSHIFT: 2>
        >>> mod & tcod.event.Modifier.LSHIFT  # Check if left shift key is held.
        <Modifier.NONE: 0>
        >>> not mod & tcod.event.Modifier.LSHIFT  # Check if left shift key is NOT held.
        True
        >>> mod & tcod.event.Modifier.SHIFT and mod & tcod.event.Modifier.CTRL  # Check if Shift+Control is held.
        <Modifier.NONE: 0>

    .. versionadded:: 12.3
    """

    NONE = 0
    LSHIFT = 1
    """Left shift."""
    RSHIFT = 2
    """Right shift."""
    SHIFT = LSHIFT | RSHIFT
    """LSHIFT | RSHIFT"""
    LCTRL = 64
    """Left control."""
    RCTRL = 128
    """Right control."""
    CTRL = LCTRL | RCTRL
    """LCTRL | RCTRL"""
    LALT = 256
    """Left alt."""
    RALT = 512
    """Right alt."""
    ALT = LALT | RALT
    """LALT | RALT"""
    LGUI = 1024
    """Left meta key."""
    RGUI = 2048
    """Right meta key."""
    GUI = LGUI | RGUI
    """LGUI | RGUI"""
    NUM = 4096
    """Numpad lock."""
    CAPS = 8192
    """Caps lock."""
    MODE = 16384
    """Alt graph."""


class MouseButton(enum.IntEnum):
    """An enum for mouse buttons.

    .. versionadded:: 16.1
    """

    LEFT = 1
    """Left mouse button."""
    MIDDLE = 2
    """Middle mouse button."""
    RIGHT = 3
    """Right mouse button."""
    X1 = 4
    """Back mouse button."""
    X2 = 5
    """Forward mouse button."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


class MouseButtonMask(enum.IntFlag):
    """A mask enum for held mouse buttons.

    .. versionadded:: 16.1
    """

    LEFT = 0x1
    """Left mouse button is held."""
    MIDDLE = 0x2
    """Middle mouse button is held."""
    RIGHT = 0x4
    """Right mouse button is held."""
    X1 = 0x8
    """Back mouse button is held."""
    X2 = 0x10
    """Forward mouse button is held."""

    def __repr__(self) -> str:
        if self.value == 0:
            return f"{self.__class__.__name__}(0)"
        return "|".join(f"{self.__class__.__name__}.{self.__class__(bit).name}" for bit in self.__class__ if bit & self)


class Event:
    """The base event class.

    Attributes:
        type (str): This events type.
        sdl_event: When available, this holds a python-cffi 'SDL_Event*'
                   pointer.  All sub-classes have this attribute.
    """

    def __init__(self, type: str | None = None) -> None:
        if type is None:
            type = self.__class__.__name__.upper()
        self.type: Final = type
        self.sdl_event = None

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Event:
        """Return a class instance from a python-cffi 'SDL_Event*' pointer."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f"<type={self.type!r}>"


class Quit(Event):
    """An application quit request event.

    For more info on when this event is triggered see:
    https://wiki.libsdl.org/SDL_EventType#SDL_QUIT

    Attributes:
        type (str): Always "QUIT".
    """

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Quit:
        self = cls()
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}()"


class KeyboardEvent(Event):
    """Base keyboard event.

    Attributes:
        type (str): Will be "KEYDOWN" or "KEYUP", depending on the event.
        scancode (Scancode): The keyboard scan-code, this is the physical location
                        of the key on the keyboard rather than the keys symbol.
        sym (KeySym): The keyboard symbol.
        mod (Modifier): A bitmask of the currently held modifier keys.

            For example, if shift is held then
            ``event.mod & tcod.event.Modifier.SHIFT`` will evaluate to a true
            value.

        repeat (bool): True if this event exists because of key repeat.

    .. versionchanged:: 12.5
        `scancode`, `sym`, and `mod` now use their respective enums.
    """

    def __init__(self, scancode: int, sym: int, mod: int, repeat: bool = False) -> None:
        super().__init__()
        self.scancode = Scancode(scancode)
        self.sym = KeySym(sym)
        self.mod = Modifier(mod)
        self.repeat = repeat

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        keysym = sdl_event.key
        self = cls(keysym.scancode, keysym.key, keysym.mod, bool(sdl_event.key.repeat))
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.{}(scancode={!r}, sym={!r}, mod={!r}{})".format(
            self.__class__.__name__,
            self.scancode,
            self.sym,
            self.mod,
            ", repeat=True" if self.repeat else "",
        )

    def __str__(self) -> str:
        return self.__repr__().replace("tcod.event.", "")


class KeyDown(KeyboardEvent):
    pass


class KeyUp(KeyboardEvent):
    pass


class MouseState(Event):
    """Mouse state.

    Attributes:
        type (str): Always "MOUSESTATE".
        position (Point): The position coordinates of the mouse.
        tile (Point): The integer tile coordinates of the mouse on the screen.
        state (int): A bitmask of which mouse buttons are currently held.

            Will be a combination of the following names:

            * tcod.event.BUTTON_LMASK
            * tcod.event.BUTTON_MMASK
            * tcod.event.BUTTON_RMASK
            * tcod.event.BUTTON_X1MASK
            * tcod.event.BUTTON_X2MASK

    .. versionadded:: 9.3

    .. versionchanged:: 15.0
        Renamed `pixel` attribute to `position`.
    """

    def __init__(
        self,
        position: tuple[float, float] = (0, 0),
        tile: tuple[float, float] | None = (0, 0),
        state: int = 0,
    ) -> None:
        super().__init__()
        self.position = Point(*position)
        self._tile = Point(*tile) if tile is not None else None
        self.state = state

    @property
    def pixel(self) -> Point:
        warnings.warn(
            "The mouse.pixel attribute is deprecated.  Use mouse.position instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.position

    @pixel.setter
    def pixel(self, value: Point) -> None:
        self.position = value

    @property
    def tile(self) -> Point:
        warnings.warn(
            "The mouse.tile attribute is deprecated.  Use mouse.position of the event returned by context.convert_event instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _verify_tile_coordinates(self._tile)

    @tile.setter
    def tile(self, xy: tuple[float, float]) -> None:
        self._tile = Point(*xy)

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(position={tuple(self.position)!r}, tile={tuple(self.tile)!r}, state={MouseButtonMask(self.state)})"

    def __str__(self) -> str:
        return ("<%s, position=(x=%i, y=%i), tile=(x=%i, y=%i), state=%s>") % (
            super().__str__().strip("<>"),
            *self.position,
            *self.tile,
            MouseButtonMask(self.state),
        )


class MouseMotion(MouseState):
    """Mouse motion event.

    Attributes:
        type (str): Always "MOUSEMOTION".
        position (Point): The pixel coordinates of the mouse.
        motion (Point): The pixel delta.
        tile (Point): The integer tile coordinates of the mouse on the screen.
        tile_motion (Point): The integer tile delta.
        state (int): A bitmask of which mouse buttons are currently held.

            Will be a combination of the following names:

            * tcod.event.BUTTON_LMASK
            * tcod.event.BUTTON_MMASK
            * tcod.event.BUTTON_RMASK
            * tcod.event.BUTTON_X1MASK
            * tcod.event.BUTTON_X2MASK

    .. versionchanged:: 15.0
        Renamed `pixel` attribute to `position`.
        Renamed `pixel_motion` attribute to `motion`.
    """

    def __init__(
        self,
        position: tuple[float, float] = (0, 0),
        motion: tuple[float, float] = (0, 0),
        tile: tuple[float, float] | None = (0, 0),
        tile_motion: tuple[float, float] | None = (0, 0),
        state: int = 0,
    ) -> None:
        super().__init__(position, tile, state)
        self.motion = Point(*motion)
        self._tile_motion = Point(*tile_motion) if tile_motion is not None else None

    @property
    def pixel_motion(self) -> Point:
        warnings.warn(
            "The mouse.pixel_motion attribute is deprecated.  Use mouse.motion instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.motion

    @pixel_motion.setter
    def pixel_motion(self, value: Point) -> None:
        warnings.warn(
            "The mouse.pixel_motion attribute is deprecated.  Use mouse.motion instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.motion = value

    @property
    def tile_motion(self) -> Point:
        warnings.warn(
            "The mouse.tile_motion attribute is deprecated."
            "  Use mouse.motion of the event returned by context.convert_event instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _verify_tile_coordinates(self._tile_motion)

    @tile_motion.setter
    def tile_motion(self, xy: tuple[float, float]) -> None:
        warnings.warn(
            "The mouse.tile_motion attribute is deprecated."
            "  Use mouse.motion of the event returned by context.convert_event instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._tile_motion = Point(*xy)

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> MouseMotion:
        motion = sdl_event.motion

        pixel = motion.x, motion.y
        pixel_motion = motion.xrel, motion.yrel
        subtile = _pixel_to_tile(*pixel)
        if subtile is None:
            self = cls(pixel, pixel_motion, None, None, motion.state)
        else:
            tile = int(subtile[0]), int(subtile[1])
            prev_pixel = pixel[0] - pixel_motion[0], pixel[1] - pixel_motion[1]
            prev_subtile = _pixel_to_tile(*prev_pixel) or (0, 0)
            prev_tile = int(prev_subtile[0]), int(prev_subtile[1])
            tile_motion = tile[0] - prev_tile[0], tile[1] - prev_tile[1]
            self = cls(pixel, pixel_motion, tile, tile_motion, motion.state)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(position={tuple(self.position)!r}, motion={tuple(self.motion)!r}, tile={tuple(self.tile)!r}, tile_motion={tuple(self.tile_motion)!r}, state={MouseButtonMask(self.state)!r})"

    def __str__(self) -> str:
        return ("<%s, motion=(x=%i, y=%i), tile_motion=(x=%i, y=%i)>") % (
            super().__str__().strip("<>"),
            *self.motion,
            *self.tile_motion,
        )


class MouseButtonEvent(MouseState):
    """Mouse button event.

    Attributes:
        type (str): Will be "MOUSEBUTTONDOWN" or "MOUSEBUTTONUP",
                    depending on the event.
        position (Point): The pixel coordinates of the mouse.
        tile (Point): The integer tile coordinates of the mouse on the screen.
        button (int): Which mouse button.

            This will be one of the following names:

            * tcod.event.BUTTON_LEFT
            * tcod.event.BUTTON_MIDDLE
            * tcod.event.BUTTON_RIGHT
            * tcod.event.BUTTON_X1
            * tcod.event.BUTTON_X2

    """

    def __init__(
        self,
        pixel: tuple[float, float] = (0, 0),
        tile: tuple[float, float] | None = (0, 0),
        button: int = 0,
    ) -> None:
        super().__init__(pixel, tile, button)

    @property
    def button(self) -> int:
        return self.state

    @button.setter
    def button(self, value: int) -> None:
        self.state = value

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        button = sdl_event.button
        pixel = button.x, button.y
        subtile = _pixel_to_tile(*pixel)
        if subtile is None:
            tile: tuple[float, float] | None = None
        else:
            tile = float(subtile[0]), float(subtile[1])
        self = cls(pixel, tile, button.button)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(position={tuple(self.position)!r}, tile={tuple(self.tile)!r}, button={MouseButton(self.button)!r})"

    def __str__(self) -> str:
        return "<type=%r, position=(x=%i, y=%i), tile=(x=%i, y=%i), button=%r)" % (
            self.type,
            *self.position,
            *self.tile,
            MouseButton(self.button),
        )


class MouseButtonDown(MouseButtonEvent):
    """Same as MouseButtonEvent but with ``type="MouseButtonDown"``."""


class MouseButtonUp(MouseButtonEvent):
    """Same as MouseButtonEvent but with ``type="MouseButtonUp"``."""


class MouseWheel(Event):
    """Mouse wheel event.

    Attributes:
        type (str): Always "MOUSEWHEEL".
        x (int): Horizontal scrolling. A positive value means scrolling right.
        y (int): Vertical scrolling. A positive value means scrolling away from
                 the user.
        flipped (bool): If True then the values of `x` and `y` are the opposite
                        of their usual values.  This depends on the settings of
                        the Operating System.
    """

    def __init__(self, x: int, y: int, flipped: bool = False) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.flipped = flipped

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> MouseWheel:
        wheel = sdl_event.wheel
        self = cls(wheel.x, wheel.y, bool(wheel.direction))
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s(x=%i, y=%i%s)" % (
            self.__class__.__name__,
            self.x,
            self.y,
            ", flipped=True" if self.flipped else "",
        )

    def __str__(self) -> str:
        return "<%s, x=%i, y=%i, flipped=%r)" % (
            super().__str__().strip("<>"),
            self.x,
            self.y,
            self.flipped,
        )


class TextInput(Event):
    """SDL text input event.

    Attributes:
        type (str): Always "TEXTINPUT".
        text (str): A Unicode string with the input.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> TextInput:
        self = cls(ffi.string(sdl_event.text.text, 32).decode("utf8"))
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(text={self.text!r})"

    def __str__(self) -> str:
        return "<{}, text={!r})".format(super().__str__().strip("<>"), self.text)


class WindowEvent(Event):
    """A window event."""

    type: Final[  # type: ignore[misc]  # Narrowing final type.
        Literal[
            "WindowShown",
            "WindowHidden",
            "WindowExposed",
            "WindowMoved",
            "WindowResized",
            "WindowSizeChanged",
            "WindowMinimized",
            "WindowMaximized",
            "WindowRestored",
            "WindowEnter",
            "WindowLeave",
            "WindowFocusGained",
            "WindowFocusLost",
            "WindowClose",
            "WindowTakeFocus",
            "WindowHitTest",
        ]
    ]
    """The current window event. This can be one of various options."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> WindowEvent | Undefined:
        if sdl_event.window.event not in cls.__WINDOW_TYPES:
            return Undefined.from_sdl_event(sdl_event)
        event_type: Final = cls.__WINDOW_TYPES[sdl_event.window.event]
        self: WindowEvent
        if sdl_event.window.event == lib.SDL_EVENT_WINDOW_MOVED:
            self = WindowMoved(sdl_event.window.data1, sdl_event.window.data2)
        elif sdl_event.window.event in (
            lib.SDL_EVENT_WINDOW_RESIZED,
            lib.SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED,
        ):
            self = WindowResized(event_type, sdl_event.window.data1, sdl_event.window.data2)
        else:
            self = cls(event_type)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r})"

    __WINDOW_TYPES: Final = {
        lib.SDL_EVENT_WINDOW_SHOWN: "WindowShown",
        lib.SDL_EVENT_WINDOW_HIDDEN: "WindowHidden",
        lib.SDL_EVENT_WINDOW_EXPOSED: "WindowExposed",
        lib.SDL_EVENT_WINDOW_MOVED: "WindowMoved",
        lib.SDL_EVENT_WINDOW_RESIZED: "WindowResized",
        lib.SDL_EVENT_WINDOW_MINIMIZED: "WindowMinimized",
        lib.SDL_EVENT_WINDOW_MAXIMIZED: "WindowMaximized",
        lib.SDL_EVENT_WINDOW_RESTORED: "WindowRestored",
        lib.SDL_EVENT_WINDOW_MOUSE_ENTER: "WindowEnter",
        lib.SDL_EVENT_WINDOW_MOUSE_LEAVE: "WindowLeave",
        lib.SDL_EVENT_WINDOW_FOCUS_GAINED: "WindowFocusGained",
        lib.SDL_EVENT_WINDOW_FOCUS_LOST: "WindowFocusLost",
        lib.SDL_EVENT_WINDOW_CLOSE_REQUESTED: "WindowClose",
        lib.SDL_EVENT_WINDOW_HIT_TEST: "WindowHitTest",
    }


class WindowMoved(WindowEvent):
    """Window moved event.

    Attributes:
        x (int): Movement on the x-axis.
        y (int): Movement on the y-axis.
    """

    type: Final[Literal["WINDOWMOVED"]]  # type: ignore[assignment,misc]
    """Always "WINDOWMOVED"."""

    def __init__(self, x: int, y: int) -> None:
        super().__init__(None)
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, x={self.x!r}, y={self.y!r})"

    def __str__(self) -> str:
        return "<{}, x={!r}, y={!r})".format(
            super().__str__().strip("<>"),
            self.x,
            self.y,
        )


class WindowResized(WindowEvent):
    """Window resized event.

    Attributes:
        width (int): The current width of the window.
        height (int): The current height of the window.
    """

    type: Final[Literal["WindowResized", "WindowSizeChanged"]]  # type: ignore[misc]
    """WindowResized" or "WindowSizeChanged"""

    def __init__(self, type: str, width: int, height: int) -> None:
        super().__init__(type)
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, width={self.width!r}, height={self.height!r})"

    def __str__(self) -> str:
        return "<{}, width={!r}, height={!r})".format(
            super().__str__().strip("<>"),
            self.width,
            self.height,
        )


class JoystickEvent(Event):
    """A base class for joystick events.

    .. versionadded:: 13.8
    """

    def __init__(self, type: str, which: int) -> None:
        super().__init__(type)
        self.which = which
        """The ID of the joystick this event is for."""

    @property
    def joystick(self) -> tcod.sdl.joystick.Joystick:
        if self.type == "JOYDEVICEADDED":
            return tcod.sdl.joystick.Joystick._open(self.which)
        return tcod.sdl.joystick.Joystick._from_instance_id(self.which)

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, which={self.which})"

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, which={self.which}>"


class JoystickAxis(JoystickEvent):
    """When a joystick axis changes in value.

    .. versionadded:: 13.8

    .. seealso::
        :any:`tcod.sdl.joystick`
    """

    which: int
    """The ID of the joystick this event is for."""

    def __init__(self, type: str, which: int, axis: int, value: int) -> None:
        super().__init__(type, which)
        self.axis = axis
        """The index of the changed axis."""
        self.value = value
        """The raw value of the axis in the range -32768 to 32767."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> JoystickAxis:
        return cls("JOYAXISMOTION", sdl_event.jaxis.which, sdl_event.jaxis.axis, sdl_event.jaxis.value)

    def __repr__(self) -> str:
        return (
            f"tcod.event.{self.__class__.__name__}"
            f"(type={self.type!r}, which={self.which}, axis={self.axis}, value={self.value})"
        )

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, axis={self.axis}, value={self.value}>"


class JoystickBall(JoystickEvent):
    """When a joystick ball is moved.

    .. versionadded:: 13.8

    .. seealso::
        :any:`tcod.sdl.joystick`
    """

    which: int
    """The ID of the joystick this event is for."""

    def __init__(self, type: str, which: int, ball: int, dx: int, dy: int) -> None:
        super().__init__(type, which)
        self.ball = ball
        """The index of the moved ball."""
        self.dx = dx
        """The X motion of the ball."""
        self.dy = dy
        """The Y motion of the ball."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> JoystickBall:
        return cls(
            "JOYBALLMOTION", sdl_event.jball.which, sdl_event.jball.ball, sdl_event.jball.xrel, sdl_event.jball.yrel
        )

    def __repr__(self) -> str:
        return (
            f"tcod.event.{self.__class__.__name__}"
            f"(type={self.type!r}, which={self.which}, ball={self.ball}, dx={self.dx}, dy={self.dy})"
        )

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, ball={self.ball}, dx={self.dx}, dy={self.dy}>"


class JoystickHat(JoystickEvent):
    """When a joystick hat changes direction.

    .. versionadded:: 13.8

    .. seealso::
        :any:`tcod.sdl.joystick`
    """

    which: int
    """The ID of the joystick this event is for."""

    def __init__(self, type: str, which: int, x: Literal[-1, 0, 1], y: Literal[-1, 0, 1]) -> None:
        super().__init__(type, which)
        self.x = x
        """The new X direction of the hat."""
        self.y = y
        """The new Y direction of the hat."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> JoystickHat:
        return cls("JOYHATMOTION", sdl_event.jhat.which, *_HAT_DIRECTIONS[sdl_event.jhat.hat])

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, which={self.which}, x={self.x}, y={self.y})"

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, x={self.x}, y={self.y}>"


class JoystickButton(JoystickEvent):
    """When a joystick button is pressed or released.

    .. versionadded:: 13.8

    Example::

        for event in tcod.event.get():
            match event:
                case JoystickButton(which=which, button=button, pressed=True):
                    print(f"Pressed {button=} on controller {which}.")
                case JoystickButton(which=which, button=button, pressed=False):
                    print(f"Released {button=} on controller {which}.")
    """

    which: int
    """The ID of the joystick this event is for."""

    def __init__(self, type: str, which: int, button: int) -> None:
        super().__init__(type, which)
        self.button = button
        """The index of the button this event is for."""

    @property
    def pressed(self) -> bool:
        """True if the joystick button has been pressed, False when the button was released."""
        return self.type == "JOYBUTTONDOWN"

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> JoystickButton:
        type = {
            lib.SDL_EVENT_JOYSTICK_BUTTON_DOWN: "JOYBUTTONDOWN",
            lib.SDL_EVENT_JOYSTICK_BUTTON_UP: "JOYBUTTONUP",
        }[sdl_event.type]
        return cls(type, sdl_event.jbutton.which, sdl_event.jbutton.button)

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, which={self.which}, button={self.button})"

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, button={self.button}>"


class JoystickDevice(JoystickEvent):
    """An event for when a joystick is added or removed.

    .. versionadded:: 13.8

    Example::

        joysticks: set[tcod.sdl.joystick.Joystick] = {}
        for event in tcod.event.get():
            match event:
                case tcod.event.JoystickDevice(type="JOYDEVICEADDED", joystick=new_joystick):
                    joysticks.add(new_joystick)
                case tcod.event.JoystickDevice(type="JOYDEVICEREMOVED", joystick=joystick):
                    joysticks.remove(joystick)
    """

    type: Final[Literal["JOYDEVICEADDED", "JOYDEVICEREMOVED"]]  # type: ignore[misc]

    which: int
    """When type="JOYDEVICEADDED" this is the device ID.
    When type="JOYDEVICEREMOVED" this is the instance ID.
    """

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> JoystickDevice:
        type = {
            lib.SDL_EVENT_JOYSTICK_ADDED: "JOYDEVICEADDED",
            lib.SDL_EVENT_JOYSTICK_REMOVED: "JOYDEVICEREMOVED",
        }[sdl_event.type]
        return cls(type, sdl_event.jdevice.which)


class ControllerEvent(Event):
    """Base class for controller events.

    .. versionadded:: 13.8
    """

    def __init__(self, type: str, which: int) -> None:
        super().__init__(type)
        self.which = which
        """The ID of the joystick this event is for."""

    @property
    def controller(self) -> tcod.sdl.joystick.GameController:
        """The :any:`GameController` for this event."""
        if self.type == "CONTROLLERDEVICEADDED":
            return tcod.sdl.joystick.GameController._open(self.which)
        return tcod.sdl.joystick.GameController._from_instance_id(self.which)

    def __repr__(self) -> str:
        return f"tcod.event.{self.__class__.__name__}(type={self.type!r}, which={self.which})"

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, which={self.which}>"


class ControllerAxis(ControllerEvent):
    """When a controller axis is moved.

    .. versionadded:: 13.8
    """

    type: Final[Literal["CONTROLLERAXISMOTION"]]  # type: ignore[misc]

    def __init__(self, type: str, which: int, axis: tcod.sdl.joystick.ControllerAxis, value: int) -> None:
        super().__init__(type, which)
        self.axis = axis
        """Which axis is being moved.  One of :any:`ControllerAxis`."""
        self.value = value
        """The new value of this events axis.

        This will be -32768 to 32767 for all axes except for triggers which are 0 to 32767 instead."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> ControllerAxis:
        return cls(
            "CONTROLLERAXISMOTION",
            sdl_event.caxis.which,
            tcod.sdl.joystick.ControllerAxis(sdl_event.caxis.axis),
            sdl_event.caxis.value,
        )

    def __repr__(self) -> str:
        return (
            f"tcod.event.{self.__class__.__name__}"
            f"(type={self.type!r}, which={self.which}, axis={self.axis}, value={self.value})"
        )

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, axis={self.axis}, value={self.value}>"


class ControllerButton(ControllerEvent):
    """When a controller button is pressed or released.

    .. versionadded:: 13.8
    """

    type: Final[Literal["CONTROLLERBUTTONDOWN", "CONTROLLERBUTTONUP"]]  # type: ignore[misc]

    def __init__(self, type: str, which: int, button: tcod.sdl.joystick.ControllerButton, pressed: bool) -> None:
        super().__init__(type, which)
        self.button = button
        """The button for this event.  One of :any:`ControllerButton`."""
        self.pressed = pressed
        """True if the button was pressed, False if it was released."""

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> ControllerButton:
        type = {
            lib.SDL_EVENT_GAMEPAD_BUTTON_DOWN: "CONTROLLERBUTTONDOWN",
            lib.SDL_EVENT_GAMEPAD_BUTTON_UP: "CONTROLLERBUTTONUP",
        }[sdl_event.type]
        return cls(
            type,
            sdl_event.cbutton.which,
            tcod.sdl.joystick.ControllerButton(sdl_event.cbutton.button),
            bool(sdl_event.cbutton.down),
        )

    def __repr__(self) -> str:
        return (
            f"tcod.event.{self.__class__.__name__}"
            f"(type={self.type!r}, which={self.which}, button={self.button}, pressed={self.pressed})"
        )

    def __str__(self) -> str:
        prefix = super().__str__().strip("<>")
        return f"<{prefix}, button={self.button}, pressed={self.pressed}>"


class ControllerDevice(ControllerEvent):
    """When a controller is added, removed, or remapped.

    .. versionadded:: 13.8
    """

    type: Final[Literal["CONTROLLERDEVICEADDED", "CONTROLLERDEVICEREMOVED", "CONTROLLERDEVICEREMAPPED"]]  # type: ignore[misc]

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> ControllerDevice:
        type = {
            lib.SDL_EVENT_GAMEPAD_ADDED: "CONTROLLERDEVICEADDED",
            lib.SDL_EVENT_GAMEPAD_REMOVED: "CONTROLLERDEVICEREMOVED",
            lib.SDL_EVENT_GAMEPAD_REMAPPED: "CONTROLLERDEVICEREMAPPED",
        }[sdl_event.type]
        return cls(type, sdl_event.cdevice.which)


class Undefined(Event):
    """This class is a place holder for SDL events without their own tcod.event class."""

    def __init__(self) -> None:
        super().__init__("")

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Undefined:
        self = cls()
        self.sdl_event = sdl_event
        return self

    def __str__(self) -> str:
        if self.sdl_event:
            return "<Undefined sdl_event.type=%i>" % self.sdl_event.type
        return "<Undefined>"


_SDL_TO_CLASS_TABLE: dict[int, type[Event]] = {
    lib.SDL_EVENT_QUIT: Quit,
    lib.SDL_EVENT_KEY_DOWN: KeyDown,
    lib.SDL_EVENT_KEY_UP: KeyUp,
    lib.SDL_EVENT_MOUSE_MOTION: MouseMotion,
    lib.SDL_EVENT_MOUSE_BUTTON_DOWN: MouseButtonDown,
    lib.SDL_EVENT_MOUSE_BUTTON_UP: MouseButtonUp,
    lib.SDL_EVENT_MOUSE_WHEEL: MouseWheel,
    lib.SDL_EVENT_TEXT_INPUT: TextInput,
    # lib.SDL_EVENT_WINDOW_EVENT: WindowEvent,
    lib.SDL_EVENT_JOYSTICK_AXIS_MOTION: JoystickAxis,
    lib.SDL_EVENT_JOYSTICK_BALL_MOTION: JoystickBall,
    lib.SDL_EVENT_JOYSTICK_HAT_MOTION: JoystickHat,
    lib.SDL_EVENT_JOYSTICK_BUTTON_DOWN: JoystickButton,
    lib.SDL_EVENT_JOYSTICK_BUTTON_UP: JoystickButton,
    lib.SDL_EVENT_JOYSTICK_ADDED: JoystickDevice,
    lib.SDL_EVENT_JOYSTICK_REMOVED: JoystickDevice,
    lib.SDL_EVENT_GAMEPAD_AXIS_MOTION: ControllerAxis,
    lib.SDL_EVENT_GAMEPAD_BUTTON_DOWN: ControllerButton,
    lib.SDL_EVENT_GAMEPAD_BUTTON_UP: ControllerButton,
    lib.SDL_EVENT_GAMEPAD_ADDED: ControllerDevice,
    lib.SDL_EVENT_GAMEPAD_REMOVED: ControllerDevice,
    lib.SDL_EVENT_GAMEPAD_REMAPPED: ControllerDevice,
}


def _parse_event(sdl_event: Any) -> Event:
    """Convert a C SDL_Event* type into a tcod Event sub-class."""
    if sdl_event.type not in _SDL_TO_CLASS_TABLE:
        return Undefined.from_sdl_event(sdl_event)
    return _SDL_TO_CLASS_TABLE[sdl_event.type].from_sdl_event(sdl_event)


def get() -> Iterator[Any]:
    """Return an iterator for all pending events.

    Events are processed as the iterator is consumed.
    Breaking out of, or discarding the iterator will leave the remaining events on the event queue.
    It is also safe to call this function inside of a loop that is already handling events
    (the event iterator is reentrant.)
    """
    if not lib.SDL_WasInit(tcod.sdl.sys.Subsystem.EVENTS):
        warnings.warn(
            "Events polled before SDL was initialized.",
            RuntimeWarning,
            stacklevel=1,
        )
        return
    sdl_event = ffi.new("SDL_Event*")
    while lib.SDL_PollEvent(sdl_event):
        if sdl_event.type in _SDL_TO_CLASS_TABLE:
            yield _SDL_TO_CLASS_TABLE[sdl_event.type].from_sdl_event(sdl_event)
        else:
            yield Undefined.from_sdl_event(sdl_event)


def wait(timeout: float | None = None) -> Iterator[Any]:
    """Block until events exist, then return an event iterator.

    `timeout` is the maximum number of seconds to wait as a floating point
    number with millisecond precision, or it can be None to wait forever.

    Returns the same iterator as a call to :any:`tcod.event.get`.

    This function is useful for simple games with little to no animations.
    The following example sleeps whenever no events are queued:

    Example::

        context: tcod.context.Context  # Context object initialized earlier.
        while True:  # Main game-loop.
            console: tcod.console.Console  # Console used for rendering.
            ...  # Render the frame to `console` and then:
            context.present(console)  # Show the console to the display.
            # The ordering to draw first before waiting for events is important.
            for event in tcod.event.wait():  # Sleeps until the next events exist.
                ...  # All events are handled at once before the next frame.

    See :any:`tcod.event.get` examples for how different events are handled.
    """
    if timeout is not None:
        lib.SDL_WaitEventTimeout(ffi.NULL, int(timeout * 1000))
    else:
        lib.SDL_WaitEvent(ffi.NULL)
    return get()


@deprecated(
    "Event dispatch should be handled via a single custom method in a Protocol instead of this class.",
    category=DeprecationWarning,
)
class EventDispatch(Generic[T]):
    '''Dispatches events to methods depending on the events type attribute.

    To use this class, make a sub-class and override the relevant `ev_*` methods.
    Then send events to the dispatch method.

    .. versionchanged:: 11.12
        This is now a generic class.
        The type hints at the return value of :any:`dispatch` and the `ev_*` methods.

    .. deprecated:: 18.0
        Event dispatch should be handled via a single custom method in a Protocol instead of this class.
        Note that events can and should be handled using Python's `match` statement.

    Example::

        import tcod

        MOVE_KEYS = {  # key_symbol: (x, y)
            # Arrow keys.
            tcod.event.KeySym.LEFT: (-1, 0),
            tcod.event.KeySym.RIGHT: (1, 0),
            tcod.event.KeySym.UP: (0, -1),
            tcod.event.KeySym.DOWN: (0, 1),
            tcod.event.KeySym.HOME: (-1, -1),
            tcod.event.KeySym.END: (-1, 1),
            tcod.event.KeySym.PAGEUP: (1, -1),
            tcod.event.KeySym.PAGEDOWN: (1, 1),
            tcod.event.KeySym.PERIOD: (0, 0),
            # Numpad keys.
            tcod.event.KeySym.KP_1: (-1, 1),
            tcod.event.KeySym.KP_2: (0, 1),
            tcod.event.KeySym.KP_3: (1, 1),
            tcod.event.KeySym.KP_4: (-1, 0),
            tcod.event.KeySym.KP_5: (0, 0),
            tcod.event.KeySym.KP_6: (1, 0),
            tcod.event.KeySym.KP_7: (-1, -1),
            tcod.event.KeySym.KP_8: (0, -1),
            tcod.event.KeySym.KP_9: (1, -1),
            tcod.event.KeySym.CLEAR: (0, 0),  # Numpad `clear` key.
            # Vi Keys.
            tcod.event.KeySym.h: (-1, 0),
            tcod.event.KeySym.j: (0, 1),
            tcod.event.KeySym.k: (0, -1),
            tcod.event.KeySym.l: (1, 0),
            tcod.event.KeySym.y: (-1, -1),
            tcod.event.KeySym.u: (1, -1),
            tcod.event.KeySym.b: (-1, 1),
            tcod.event.KeySym.n: (1, 1),
        }


        class State(tcod.event.EventDispatch[None]):
            """A state-based superclass that converts `events` into `commands`.

            The configuration used to convert events to commands are hard-coded
            in this example, but could be modified to be user controlled.

            Subclasses will override the `cmd_*` methods with their own
            functionality.  There could be a subclass for every individual state
            of your game.
            """

            def ev_quit(self, event: tcod.event.Quit) -> None:
                """The window close button was clicked or Alt+F$ was pressed."""
                print(event)
                self.cmd_quit()

            def ev_keydown(self, event: tcod.event.KeyDown) -> None:
                """A key was pressed."""
                print(event)
                if event.sym in MOVE_KEYS:
                    # Send movement keys to the cmd_move method with parameters.
                    self.cmd_move(*MOVE_KEYS[event.sym])
                elif event.sym == tcod.event.KeySym.ESCAPE:
                    self.cmd_escape()

            def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown) -> None:
                """The window was clicked."""
                print(event)

            def ev_mousemotion(self, event: tcod.event.MouseMotion) -> None:
                """The mouse has moved within the window."""
                print(event)

            def cmd_move(self, x: int, y: int) -> None:
                """Intent to move: `x` and `y` is the direction, both may be 0."""
                print("Command move: " + str((x, y)))

            def cmd_escape(self) -> None:
                """Intent to exit this state."""
                print("Command escape.")
                self.cmd_quit()

            def cmd_quit(self) -> None:
                """Intent to exit the game."""
                print("Command quit.")
                raise SystemExit()


        root_console = libtcodpy.console_init_root(80, 60)
        state = State()
        while True:
            libtcodpy.console_flush()
            for event in tcod.event.wait():
                state.dispatch(event)
    '''

    __slots__ = ()

    def dispatch(self, event: Any) -> T | None:
        """Send an event to an `ev_*` method.

        `*` will be the `event.type` attribute converted to lower-case.

        Values returned by `ev_*` calls will be returned by this function.
        This value always defaults to None for any non-overridden method.

        .. versionchanged:: 11.12
            Now returns the return value of `ev_*` methods.
            `event.type` values of None are deprecated.
        """
        if event.type is None:
            warnings.warn(
                "`event.type` attribute should not be None.",
                DeprecationWarning,
                stacklevel=2,
            )
            return None
        func_name = f"ev_{event.type.lower()}"
        func: Callable[[Any], T | None] | None = getattr(self, func_name, None)
        if func is None:
            warnings.warn(f"{func_name} is missing from this EventDispatch object.", RuntimeWarning, stacklevel=2)
            return None
        return func(event)

    def event_get(self) -> None:
        for event in get():
            self.dispatch(event)

    def event_wait(self, timeout: float | None) -> None:
        wait(timeout)
        self.event_get()

    def ev_quit(self, event: tcod.event.Quit, /) -> T | None:
        """Called when the termination of the program is requested."""

    def ev_keydown(self, event: tcod.event.KeyDown, /) -> T | None:
        """Called when a keyboard key is pressed or repeated."""

    def ev_keyup(self, event: tcod.event.KeyUp, /) -> T | None:
        """Called when a keyboard key is released."""

    def ev_mousemotion(self, event: tcod.event.MouseMotion, /) -> T | None:
        """Called when the mouse is moved."""

    def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown, /) -> T | None:
        """Called when a mouse button is pressed."""

    def ev_mousebuttonup(self, event: tcod.event.MouseButtonUp, /) -> T | None:
        """Called when a mouse button is released."""

    def ev_mousewheel(self, event: tcod.event.MouseWheel, /) -> T | None:
        """Called when the mouse wheel is scrolled."""

    def ev_textinput(self, event: tcod.event.TextInput, /) -> T | None:
        """Called to handle Unicode input."""

    def ev_windowshown(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window is shown."""

    def ev_windowhidden(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window is hidden."""

    def ev_windowexposed(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when a window is exposed, and needs to be refreshed.

        This usually means a call to :any:`libtcodpy.console_flush` is necessary.
        """

    def ev_windowmoved(self, event: tcod.event.WindowMoved, /) -> T | None:
        """Called when the window is moved."""

    def ev_windowresized(self, event: tcod.event.WindowResized, /) -> T | None:
        """Called when the window is resized."""

    def ev_windowsizechanged(self, event: tcod.event.WindowResized, /) -> T | None:
        """Called when the system or user changes the size of the window."""

    def ev_windowminimized(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window is minimized."""

    def ev_windowmaximized(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window is maximized."""

    def ev_windowrestored(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window is restored."""

    def ev_windowenter(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window gains mouse focus."""

    def ev_windowleave(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window loses mouse focus."""

    def ev_windowfocusgained(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window gains keyboard focus."""

    def ev_windowfocuslost(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window loses keyboard focus."""

    def ev_windowclose(self, event: tcod.event.WindowEvent, /) -> T | None:
        """Called when the window manager requests the window to be closed."""

    def ev_windowtakefocus(self, event: tcod.event.WindowEvent, /) -> T | None:
        pass

    def ev_windowhittest(self, event: tcod.event.WindowEvent, /) -> T | None:
        pass

    def ev_joyaxismotion(self, event: tcod.event.JoystickAxis, /) -> T | None:
        """Called when a joystick analog is moved.

        .. versionadded:: 13.8
        """

    def ev_joyballmotion(self, event: tcod.event.JoystickBall, /) -> T | None:
        """Called when a joystick ball is moved.

        .. versionadded:: 13.8
        """

    def ev_joyhatmotion(self, event: tcod.event.JoystickHat, /) -> T | None:
        """Called when a joystick hat is moved.

        .. versionadded:: 13.8
        """

    def ev_joybuttondown(self, event: tcod.event.JoystickButton, /) -> T | None:
        """Called when a joystick button is pressed.

        .. versionadded:: 13.8
        """

    def ev_joybuttonup(self, event: tcod.event.JoystickButton, /) -> T | None:
        """Called when a joystick button is released.

        .. versionadded:: 13.8
        """

    def ev_joydeviceadded(self, event: tcod.event.JoystickDevice, /) -> T | None:
        """Called when a joystick is added.

        .. versionadded:: 13.8
        """

    def ev_joydeviceremoved(self, event: tcod.event.JoystickDevice, /) -> T | None:
        """Called when a joystick is removed.

        .. versionadded:: 13.8
        """

    def ev_controlleraxismotion(self, event: tcod.event.ControllerAxis, /) -> T | None:
        """Called when a controller analog is moved.

        .. versionadded:: 13.8
        """

    def ev_controllerbuttondown(self, event: tcod.event.ControllerButton, /) -> T | None:
        """Called when a controller button is pressed.

        .. versionadded:: 13.8
        """

    def ev_controllerbuttonup(self, event: tcod.event.ControllerButton, /) -> T | None:
        """Called when a controller button is released.

        .. versionadded:: 13.8
        """

    def ev_controllerdeviceadded(self, event: tcod.event.ControllerDevice, /) -> T | None:
        """Called when a standard controller is added.

        .. versionadded:: 13.8
        """

    def ev_controllerdeviceremoved(self, event: tcod.event.ControllerDevice, /) -> T | None:
        """Called when a standard controller is removed.

        .. versionadded:: 13.8
        """

    def ev_controllerdeviceremapped(self, event: tcod.event.ControllerDevice, /) -> T | None:
        """Called when a standard controller is remapped.

        .. versionadded:: 13.8
        """

    def ev_(self, event: Any, /) -> T | None:
        pass


def get_mouse_state() -> MouseState:
    """Return the current state of the mouse.

    .. versionadded:: 9.3
    """
    xy = ffi.new("int[2]")
    buttons = lib.SDL_GetMouseState(xy, xy + 1)
    tile = _pixel_to_tile(*xy)
    if tile is None:
        return MouseState((xy[0], xy[1]), None, buttons)
    return MouseState((xy[0], xy[1]), (int(tile[0]), int(tile[1])), buttons)


@ffi.def_extern()  # type: ignore[misc]
def _sdl_event_watcher(userdata: Any, sdl_event: Any) -> int:
    callback: Callable[[Event], None] = ffi.from_handle(userdata)
    callback(_parse_event(sdl_event))
    return 0


_EventCallback = TypeVar("_EventCallback", bound=Callable[[Event], None])
_event_watch_handles: dict[Callable[[Event], None], Any] = {}  # Callbacks and their FFI handles.


def add_watch(callback: _EventCallback) -> _EventCallback:
    """Add a callback for watching events.

    This function can be called with the callback to register, or be used as a decorator.

    Callbacks added as event watchers can later be removed with :any:`tcod.event.remove_watch`.

    .. warning::
        How uncaught exceptions in a callback are handled is not currently defined by tcod.
        They will likely be handled by :any:`sys.unraisablehook`.
        This may be later changed to pass the exception to a :any:`tcod.event.get` or :any:`tcod.event.wait` call.

    Args:
        callback (Callable[[Event], None]):
            A function which accepts :any:`Event` parameters.

    Example::

        import tcod.event

        @tcod.event.add_watch
        def handle_events(event: tcod.event.Event) -> None:
            if isinstance(event, tcod.event.KeyDown):
                print(event)

    .. versionadded:: 13.4
    """
    if callback in _event_watch_handles:
        warnings.warn(
            f"{callback} is already an active event watcher, nothing was added.", RuntimeWarning, stacklevel=2
        )
        return callback
    handle = _event_watch_handles[callback] = ffi.new_handle(callback)
    lib.SDL_AddEventWatch(lib._sdl_event_watcher, handle)
    return callback


def remove_watch(callback: Callable[[Event], None]) -> None:
    """Remove a callback as an event watcher.

    Args:
        callback (Callable[[Event], None]):
            A function which has been previously registered with :any:`tcod.event.add_watch`.

    .. versionadded:: 13.4
    """
    if callback not in _event_watch_handles:
        warnings.warn(f"{callback} is not an active event watcher, nothing was removed.", RuntimeWarning, stacklevel=2)
        return
    handle = _event_watch_handles[callback]
    lib.SDL_RemoveEventWatch(lib._sdl_event_watcher, handle)
    del _event_watch_handles[callback]


def get_keyboard_state() -> NDArray[np.bool_]:
    """Return a boolean array with the current keyboard state.

    Index this array with a scancode.  The value will be True if the key is
    currently held.

    Example::

        state = tcod.event.get_keyboard_state()

        # Get a WASD movement vector:
        x = int(state[tcod.event.Scancode.D]) - int(state[tcod.event.Scancode.A])
        y = int(state[tcod.event.Scancode.S]) - int(state[tcod.event.Scancode.W])

        # Key with 'z' glyph is held:
        is_z_held = state[tcod.event.KeySym.z.scancode]


    .. versionadded:: 12.3
    """
    num_keys = ffi.new("int[1]")
    keyboard_state = lib.SDL_GetKeyboardState(num_keys)
    out: NDArray[np.bool_] = np.frombuffer(ffi.buffer(keyboard_state[0 : num_keys[0]]), dtype=np.bool_)
    out.flags["WRITEABLE"] = False  # This buffer is supposed to be const.
    return out


def get_modifier_state() -> Modifier:
    """Return a bitmask of the active keyboard modifiers.

    .. versionadded:: 12.3
    """
    return Modifier(lib.SDL_GetModState())


class Scancode(enum.IntEnum):
    """A Scancode represents the physical location of a key.

    For example the scan codes for WASD remain in the same physical location
    regardless of the actual keyboard layout.

    These names are derived from SDL except for the numbers which are prefixed
    with ``N`` (since raw numbers can not be a Python name.)

    .. versionadded:: 12.3

    ==================  ===
    UNKNOWN               0
    A                     4
    B                     5
    C                     6
    D                     7
    E                     8
    F                     9
    G                    10
    H                    11
    I                    12
    J                    13
    K                    14
    L                    15
    M                    16
    N                    17
    O                    18
    P                    19
    Q                    20
    R                    21
    S                    22
    T                    23
    U                    24
    V                    25
    W                    26
    X                    27
    Y                    28
    Z                    29
    N1                   30
    N2                   31
    N3                   32
    N4                   33
    N5                   34
    N6                   35
    N7                   36
    N8                   37
    N9                   38
    N0                   39
    RETURN               40
    ESCAPE               41
    BACKSPACE            42
    TAB                  43
    SPACE                44
    MINUS                45
    EQUALS               46
    LEFTBRACKET          47
    RIGHTBRACKET         48
    BACKSLASH            49
    NONUSHASH            50
    SEMICOLON            51
    APOSTROPHE           52
    GRAVE                53
    COMMA                54
    PERIOD               55
    SLASH                56
    CAPSLOCK             57
    F1                   58
    F2                   59
    F3                   60
    F4                   61
    F5                   62
    F6                   63
    F7                   64
    F8                   65
    F9                   66
    F10                  67
    F11                  68
    F12                  69
    PRINTSCREEN          70
    SCROLLLOCK           71
    PAUSE                72
    INSERT               73
    HOME                 74
    PAGEUP               75
    DELETE               76
    END                  77
    PAGEDOWN             78
    RIGHT                79
    LEFT                 80
    DOWN                 81
    UP                   82
    NUMLOCKCLEAR         83
    KP_DIVIDE            84
    KP_MULTIPLY          85
    KP_MINUS             86
    KP_PLUS              87
    KP_ENTER             88
    KP_1                 89
    KP_2                 90
    KP_3                 91
    KP_4                 92
    KP_5                 93
    KP_6                 94
    KP_7                 95
    KP_8                 96
    KP_9                 97
    KP_0                 98
    KP_PERIOD            99
    NONUSBACKSLASH      100
    APPLICATION         101
    POWER               102
    KP_EQUALS           103
    F13                 104
    F14                 105
    F15                 106
    F16                 107
    F17                 108
    F18                 109
    F19                 110
    F20                 111
    F21                 112
    F22                 113
    F23                 114
    F24                 115
    EXECUTE             116
    HELP                117
    MENU                118
    SELECT              119
    STOP                120
    AGAIN               121
    UNDO                122
    CUT                 123
    COPY                124
    PASTE               125
    FIND                126
    MUTE                127
    VOLUMEUP            128
    VOLUMEDOWN          129
    KP_COMMA            133
    KP_EQUALSAS400      134
    INTERNATIONAL1      135
    INTERNATIONAL2      136
    INTERNATIONAL3      137
    INTERNATIONAL4      138
    INTERNATIONAL5      139
    INTERNATIONAL6      140
    INTERNATIONAL7      141
    INTERNATIONAL8      142
    INTERNATIONAL9      143
    LANG1               144
    LANG2               145
    LANG3               146
    LANG4               147
    LANG5               148
    LANG6               149
    LANG7               150
    LANG8               151
    LANG9               152
    ALTERASE            153
    SYSREQ              154
    CANCEL              155
    CLEAR               156
    PRIOR               157
    RETURN2             158
    SEPARATOR           159
    OUT                 160
    OPER                161
    CLEARAGAIN          162
    CRSEL               163
    EXSEL               164
    KP_00               176
    KP_000              177
    THOUSANDSSEPARATOR  178
    DECIMALSEPARATOR    179
    CURRENCYUNIT        180
    CURRENCYSUBUNIT     181
    KP_LEFTPAREN        182
    KP_RIGHTPAREN       183
    KP_LEFTBRACE        184
    KP_RIGHTBRACE       185
    KP_TAB              186
    KP_BACKSPACE        187
    KP_A                188
    KP_B                189
    KP_C                190
    KP_D                191
    KP_E                192
    KP_F                193
    KP_XOR              194
    KP_POWER            195
    KP_PERCENT          196
    KP_LESS             197
    KP_GREATER          198
    KP_AMPERSAND        199
    KP_DBLAMPERSAND     200
    KP_VERTICALBAR      201
    KP_DBLVERTICALBAR   202
    KP_COLON            203
    KP_HASH             204
    KP_SPACE            205
    KP_AT               206
    KP_EXCLAM           207
    KP_MEMSTORE         208
    KP_MEMRECALL        209
    KP_MEMCLEAR         210
    KP_MEMADD           211
    KP_MEMSUBTRACT      212
    KP_MEMMULTIPLY      213
    KP_MEMDIVIDE        214
    KP_PLUSMINUS        215
    KP_CLEAR            216
    KP_CLEARENTRY       217
    KP_BINARY           218
    KP_OCTAL            219
    KP_DECIMAL          220
    KP_HEXADECIMAL      221
    LCTRL               224
    LSHIFT              225
    LALT                226
    LGUI                227
    RCTRL               228
    RSHIFT              229
    RALT                230
    RGUI                231
    MODE                257
    AUDIONEXT           258
    AUDIOPREV           259
    AUDIOSTOP           260
    AUDIOPLAY           261
    AUDIOMUTE           262
    MEDIASELECT         263
    WWW                 264
    MAIL                265
    CALCULATOR          266
    COMPUTER            267
    AC_SEARCH           268
    AC_HOME             269
    AC_BACK             270
    AC_FORWARD          271
    AC_STOP             272
    AC_REFRESH          273
    AC_BOOKMARKS        274
    BRIGHTNESSDOWN      275
    BRIGHTNESSUP        276
    DISPLAYSWITCH       277
    KBDILLUMTOGGLE      278
    KBDILLUMDOWN        279
    KBDILLUMUP          280
    EJECT               281
    SLEEP               282
    APP1                283
    APP2                284
    ==================  ===

    """

    # --- SDL scancodes ---
    UNKNOWN = 0
    A = 4
    B = 5
    C = 6
    D = 7
    E = 8
    F = 9
    G = 10
    H = 11
    I = 12  # noqa: E741
    J = 13
    K = 14
    L = 15
    M = 16
    N = 17
    O = 18  # noqa: E741
    P = 19
    Q = 20
    R = 21
    S = 22
    T = 23
    U = 24
    V = 25
    W = 26
    X = 27
    Y = 28
    Z = 29
    N1 = 30
    N2 = 31
    N3 = 32
    N4 = 33
    N5 = 34
    N6 = 35
    N7 = 36
    N8 = 37
    N9 = 38
    N0 = 39
    RETURN = 40
    ESCAPE = 41
    BACKSPACE = 42
    TAB = 43
    SPACE = 44
    MINUS = 45
    EQUALS = 46
    LEFTBRACKET = 47
    RIGHTBRACKET = 48
    BACKSLASH = 49
    NONUSHASH = 50
    SEMICOLON = 51
    APOSTROPHE = 52
    GRAVE = 53
    COMMA = 54
    PERIOD = 55
    SLASH = 56
    CAPSLOCK = 57
    F1 = 58
    F2 = 59
    F3 = 60
    F4 = 61
    F5 = 62
    F6 = 63
    F7 = 64
    F8 = 65
    F9 = 66
    F10 = 67
    F11 = 68
    F12 = 69
    PRINTSCREEN = 70
    SCROLLLOCK = 71
    PAUSE = 72
    INSERT = 73
    HOME = 74
    PAGEUP = 75
    DELETE = 76
    END = 77
    PAGEDOWN = 78
    RIGHT = 79
    LEFT = 80
    DOWN = 81
    UP = 82
    NUMLOCKCLEAR = 83
    KP_DIVIDE = 84
    KP_MULTIPLY = 85
    KP_MINUS = 86
    KP_PLUS = 87
    KP_ENTER = 88
    KP_1 = 89
    KP_2 = 90
    KP_3 = 91
    KP_4 = 92
    KP_5 = 93
    KP_6 = 94
    KP_7 = 95
    KP_8 = 96
    KP_9 = 97
    KP_0 = 98
    KP_PERIOD = 99
    NONUSBACKSLASH = 100
    APPLICATION = 101
    POWER = 102
    KP_EQUALS = 103
    F13 = 104
    F14 = 105
    F15 = 106
    F16 = 107
    F17 = 108
    F18 = 109
    F19 = 110
    F20 = 111
    F21 = 112
    F22 = 113
    F23 = 114
    F24 = 115
    EXECUTE = 116
    HELP = 117
    MENU = 118
    SELECT = 119
    STOP = 120
    AGAIN = 121
    UNDO = 122
    CUT = 123
    COPY = 124
    PASTE = 125
    FIND = 126
    MUTE = 127
    VOLUMEUP = 128
    VOLUMEDOWN = 129
    KP_COMMA = 133
    KP_EQUALSAS400 = 134
    INTERNATIONAL1 = 135
    INTERNATIONAL2 = 136
    INTERNATIONAL3 = 137
    INTERNATIONAL4 = 138
    INTERNATIONAL5 = 139
    INTERNATIONAL6 = 140
    INTERNATIONAL7 = 141
    INTERNATIONAL8 = 142
    INTERNATIONAL9 = 143
    LANG1 = 144
    LANG2 = 145
    LANG3 = 146
    LANG4 = 147
    LANG5 = 148
    LANG6 = 149
    LANG7 = 150
    LANG8 = 151
    LANG9 = 152
    ALTERASE = 153
    SYSREQ = 154
    CANCEL = 155
    CLEAR = 156
    PRIOR = 157
    RETURN2 = 158
    SEPARATOR = 159
    OUT = 160
    OPER = 161
    CLEARAGAIN = 162
    CRSEL = 163
    EXSEL = 164
    KP_00 = 176
    KP_000 = 177
    THOUSANDSSEPARATOR = 178
    DECIMALSEPARATOR = 179
    CURRENCYUNIT = 180
    CURRENCYSUBUNIT = 181
    KP_LEFTPAREN = 182
    KP_RIGHTPAREN = 183
    KP_LEFTBRACE = 184
    KP_RIGHTBRACE = 185
    KP_TAB = 186
    KP_BACKSPACE = 187
    KP_A = 188
    KP_B = 189
    KP_C = 190
    KP_D = 191
    KP_E = 192
    KP_F = 193
    KP_XOR = 194
    KP_POWER = 195
    KP_PERCENT = 196
    KP_LESS = 197
    KP_GREATER = 198
    KP_AMPERSAND = 199
    KP_DBLAMPERSAND = 200
    KP_VERTICALBAR = 201
    KP_DBLVERTICALBAR = 202
    KP_COLON = 203
    KP_HASH = 204
    KP_SPACE = 205
    KP_AT = 206
    KP_EXCLAM = 207
    KP_MEMSTORE = 208
    KP_MEMRECALL = 209
    KP_MEMCLEAR = 210
    KP_MEMADD = 211
    KP_MEMSUBTRACT = 212
    KP_MEMMULTIPLY = 213
    KP_MEMDIVIDE = 214
    KP_PLUSMINUS = 215
    KP_CLEAR = 216
    KP_CLEARENTRY = 217
    KP_BINARY = 218
    KP_OCTAL = 219
    KP_DECIMAL = 220
    KP_HEXADECIMAL = 221
    LCTRL = 224
    LSHIFT = 225
    LALT = 226
    LGUI = 227
    RCTRL = 228
    RSHIFT = 229
    RALT = 230
    RGUI = 231
    MODE = 257
    SLEEP = 258
    WAKE = 259
    CHANNEL_INCREMENT = 260
    CHANNEL_DECREMENT = 261
    MEDIA_PLAY = 262
    MEDIA_PAUSE = 263
    MEDIA_RECORD = 264
    MEDIA_FAST_FORWARD = 265
    MEDIA_REWIND = 266
    MEDIA_NEXT_TRACK = 267
    MEDIA_PREVIOUS_TRACK = 268
    MEDIA_STOP = 269
    MEDIA_EJECT = 270
    MEDIA_PLAY_PAUSE = 271
    MEDIA_SELECT = 272
    AC_NEW = 273
    AC_OPEN = 274
    AC_CLOSE = 275
    AC_EXIT = 276
    AC_SAVE = 277
    AC_PRINT = 278
    AC_PROPERTIES = 279
    AC_SEARCH = 280
    AC_HOME = 281
    AC_BACK = 282
    AC_FORWARD = 283
    AC_STOP = 284
    AC_REFRESH = 285
    AC_BOOKMARKS = 286
    SOFTLEFT = 287
    SOFTRIGHT = 288
    CALL = 289
    ENDCALL = 290
    RESERVED = 400
    COUNT = 512
    # --- end ---

    @property
    def label(self) -> str:
        """Return a human-readable name of a key based on its scancode.

        Be sure not to confuse this with ``.name``, which will return the enum
        name rather than the human-readable name.

        .. seealso::
            :any:`KeySym.label`
        """
        return self.keysym.label

    @property
    def keysym(self) -> KeySym:
        """Return a :class:`KeySym` from a scancode.

        Based on the current keyboard layout.
        """
        _init_sdl_video()
        return KeySym(lib.SDL_GetKeyFromScancode(self.value, 0, False))  # noqa: FBT003

    @property
    def scancode(self) -> Scancode:
        """Return a scancode from a keycode.

        Returns itself since it is already a :class:`Scancode`.

        .. seealso::
            :any:`KeySym.scancode`
        """
        return self

    @classmethod
    def _missing_(cls, value: object) -> Scancode | None:
        if not isinstance(value, int):
            return None
        result = cls(0)
        result._value_ = value
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KeySym):
            msg = "Scancode and KeySym enums can not be compared directly. Convert one or the other to the same type."
            raise TypeError(msg)
        return super().__eq__(other)

    def __hash__(self) -> int:
        # __eq__ was defined, so __hash__ must be defined.
        return super().__hash__()

    def __repr__(self) -> str:
        """Return the fully qualified name of this enum."""
        return f"tcod.event.{self.__class__.__name__}.{self.name}"


class KeySym(enum.IntEnum):
    """Keyboard constants based on their symbol.

    These names are derived from SDL except for the numbers which are prefixed
    with ``N`` (since raw numbers can not be a Python name.)

    .. versionadded:: 12.3

    ==================  ==========
    UNKNOWN                      0
    BACKSPACE                    8
    TAB                          9
    RETURN                      13
    ESCAPE                      27
    SPACE                       32
    EXCLAIM                     33
    QUOTEDBL                    34
    HASH                        35
    DOLLAR                      36
    PERCENT                     37
    AMPERSAND                   38
    QUOTE                       39
    LEFTPAREN                   40
    RIGHTPAREN                  41
    ASTERISK                    42
    PLUS                        43
    COMMA                       44
    MINUS                       45
    PERIOD                      46
    SLASH                       47
    N0                          48
    N1                          49
    N2                          50
    N3                          51
    N4                          52
    N5                          53
    N6                          54
    N7                          55
    N8                          56
    N9                          57
    COLON                       58
    SEMICOLON                   59
    LESS                        60
    EQUALS                      61
    GREATER                     62
    QUESTION                    63
    AT                          64
    LEFTBRACKET                 91
    BACKSLASH                   92
    RIGHTBRACKET                93
    CARET                       94
    UNDERSCORE                  95
    BACKQUOTE                   96
    a                           97
    b                           98
    c                           99
    d                          100
    e                          101
    f                          102
    g                          103
    h                          104
    i                          105
    j                          106
    k                          107
    l                          108
    m                          109
    n                          110
    o                          111
    p                          112
    q                          113
    r                          114
    s                          115
    t                          116
    u                          117
    v                          118
    w                          119
    x                          120
    y                          121
    z                          122
    DELETE                     127
    SCANCODE_MASK       1073741824
    CAPSLOCK            1073741881
    F1                  1073741882
    F2                  1073741883
    F3                  1073741884
    F4                  1073741885
    F5                  1073741886
    F6                  1073741887
    F7                  1073741888
    F8                  1073741889
    F9                  1073741890
    F10                 1073741891
    F11                 1073741892
    F12                 1073741893
    PRINTSCREEN         1073741894
    SCROLLLOCK          1073741895
    PAUSE               1073741896
    INSERT              1073741897
    HOME                1073741898
    PAGEUP              1073741899
    END                 1073741901
    PAGEDOWN            1073741902
    RIGHT               1073741903
    LEFT                1073741904
    DOWN                1073741905
    UP                  1073741906
    NUMLOCKCLEAR        1073741907
    KP_DIVIDE           1073741908
    KP_MULTIPLY         1073741909
    KP_MINUS            1073741910
    KP_PLUS             1073741911
    KP_ENTER            1073741912
    KP_1                1073741913
    KP_2                1073741914
    KP_3                1073741915
    KP_4                1073741916
    KP_5                1073741917
    KP_6                1073741918
    KP_7                1073741919
    KP_8                1073741920
    KP_9                1073741921
    KP_0                1073741922
    KP_PERIOD           1073741923
    APPLICATION         1073741925
    POWER               1073741926
    KP_EQUALS           1073741927
    F13                 1073741928
    F14                 1073741929
    F15                 1073741930
    F16                 1073741931
    F17                 1073741932
    F18                 1073741933
    F19                 1073741934
    F20                 1073741935
    F21                 1073741936
    F22                 1073741937
    F23                 1073741938
    F24                 1073741939
    EXECUTE             1073741940
    HELP                1073741941
    MENU                1073741942
    SELECT              1073741943
    STOP                1073741944
    AGAIN               1073741945
    UNDO                1073741946
    CUT                 1073741947
    COPY                1073741948
    PASTE               1073741949
    FIND                1073741950
    MUTE                1073741951
    VOLUMEUP            1073741952
    VOLUMEDOWN          1073741953
    KP_COMMA            1073741957
    KP_EQUALSAS400      1073741958
    ALTERASE            1073741977
    SYSREQ              1073741978
    CANCEL              1073741979
    CLEAR               1073741980
    PRIOR               1073741981
    RETURN2             1073741982
    SEPARATOR           1073741983
    OUT                 1073741984
    OPER                1073741985
    CLEARAGAIN          1073741986
    CRSEL               1073741987
    EXSEL               1073741988
    KP_00               1073742000
    KP_000              1073742001
    THOUSANDSSEPARATOR  1073742002
    DECIMALSEPARATOR    1073742003
    CURRENCYUNIT        1073742004
    CURRENCYSUBUNIT     1073742005
    KP_LEFTPAREN        1073742006
    KP_RIGHTPAREN       1073742007
    KP_LEFTBRACE        1073742008
    KP_RIGHTBRACE       1073742009
    KP_TAB              1073742010
    KP_BACKSPACE        1073742011
    KP_A                1073742012
    KP_B                1073742013
    KP_C                1073742014
    KP_D                1073742015
    KP_E                1073742016
    KP_F                1073742017
    KP_XOR              1073742018
    KP_POWER            1073742019
    KP_PERCENT          1073742020
    KP_LESS             1073742021
    KP_GREATER          1073742022
    KP_AMPERSAND        1073742023
    KP_DBLAMPERSAND     1073742024
    KP_VERTICALBAR      1073742025
    KP_DBLVERTICALBAR   1073742026
    KP_COLON            1073742027
    KP_HASH             1073742028
    KP_SPACE            1073742029
    KP_AT               1073742030
    KP_EXCLAM           1073742031
    KP_MEMSTORE         1073742032
    KP_MEMRECALL        1073742033
    KP_MEMCLEAR         1073742034
    KP_MEMADD           1073742035
    KP_MEMSUBTRACT      1073742036
    KP_MEMMULTIPLY      1073742037
    KP_MEMDIVIDE        1073742038
    KP_PLUSMINUS        1073742039
    KP_CLEAR            1073742040
    KP_CLEARENTRY       1073742041
    KP_BINARY           1073742042
    KP_OCTAL            1073742043
    KP_DECIMAL          1073742044
    KP_HEXADECIMAL      1073742045
    LCTRL               1073742048
    LSHIFT              1073742049
    LALT                1073742050
    LGUI                1073742051
    RCTRL               1073742052
    RSHIFT              1073742053
    RALT                1073742054
    RGUI                1073742055
    MODE                1073742081
    AUDIONEXT           1073742082
    AUDIOPREV           1073742083
    AUDIOSTOP           1073742084
    AUDIOPLAY           1073742085
    AUDIOMUTE           1073742086
    MEDIASELECT         1073742087
    WWW                 1073742088
    MAIL                1073742089
    CALCULATOR          1073742090
    COMPUTER            1073742091
    AC_SEARCH           1073742092
    AC_HOME             1073742093
    AC_BACK             1073742094
    AC_FORWARD          1073742095
    AC_STOP             1073742096
    AC_REFRESH          1073742097
    AC_BOOKMARKS        1073742098
    BRIGHTNESSDOWN      1073742099
    BRIGHTNESSUP        1073742100
    DISPLAYSWITCH       1073742101
    KBDILLUMTOGGLE      1073742102
    KBDILLUMDOWN        1073742103
    KBDILLUMUP          1073742104
    EJECT               1073742105
    SLEEP               1073742106
    ==================  ==========
    """

    # --- SDL keyboard symbols ---
    UNKNOWN = 0
    BACKSPACE = 8
    TAB = 9
    RETURN = 13
    ESCAPE = 27
    SPACE = 32
    EXCLAIM = 33
    DBLAPOSTROPHE = 34
    HASH = 35
    DOLLAR = 36
    PERCENT = 37
    AMPERSAND = 38
    APOSTROPHE = 39
    LEFTPAREN = 40
    RIGHTPAREN = 41
    ASTERISK = 42
    PLUS = 43
    COMMA = 44
    MINUS = 45
    PERIOD = 46
    SLASH = 47
    N0 = 48
    N1 = 49
    N2 = 50
    N3 = 51
    N4 = 52
    N5 = 53
    N6 = 54
    N7 = 55
    N8 = 56
    N9 = 57
    COLON = 58
    SEMICOLON = 59
    LESS = 60
    EQUALS = 61
    GREATER = 62
    QUESTION = 63
    AT = 64
    LEFTBRACKET = 91
    BACKSLASH = 92
    RIGHTBRACKET = 93
    CARET = 94
    UNDERSCORE = 95
    GRAVE = 96
    A = 97
    B = 98
    C = 99
    D = 100
    E = 101
    F = 102
    G = 103
    H = 104
    I = 105  # noqa: E741
    J = 106
    K = 107
    L = 108
    M = 109
    N = 110
    O = 111  # noqa: E741
    P = 112
    Q = 113
    R = 114
    S = 115
    T = 116
    U = 117
    V = 118
    W = 119
    X = 120
    Y = 121
    Z = 122
    LEFTBRACE = 123
    PIPE = 124
    RIGHTBRACE = 125
    TILDE = 126
    DELETE = 127
    PLUSMINUS = 177
    EXTENDED_MASK = 536870912
    LEFT_TAB = 536870913
    LEVEL5_SHIFT = 536870914
    MULTI_KEY_COMPOSE = 536870915
    LMETA = 536870916
    RMETA = 536870917
    LHYPER = 536870918
    RHYPER = 536870919
    SCANCODE_MASK = 1073741824
    CAPSLOCK = 1073741881
    F1 = 1073741882
    F2 = 1073741883
    F3 = 1073741884
    F4 = 1073741885
    F5 = 1073741886
    F6 = 1073741887
    F7 = 1073741888
    F8 = 1073741889
    F9 = 1073741890
    F10 = 1073741891
    F11 = 1073741892
    F12 = 1073741893
    PRINTSCREEN = 1073741894
    SCROLLLOCK = 1073741895
    PAUSE = 1073741896
    INSERT = 1073741897
    HOME = 1073741898
    PAGEUP = 1073741899
    END = 1073741901
    PAGEDOWN = 1073741902
    RIGHT = 1073741903
    LEFT = 1073741904
    DOWN = 1073741905
    UP = 1073741906
    NUMLOCKCLEAR = 1073741907
    KP_DIVIDE = 1073741908
    KP_MULTIPLY = 1073741909
    KP_MINUS = 1073741910
    KP_PLUS = 1073741911
    KP_ENTER = 1073741912
    KP_1 = 1073741913
    KP_2 = 1073741914
    KP_3 = 1073741915
    KP_4 = 1073741916
    KP_5 = 1073741917
    KP_6 = 1073741918
    KP_7 = 1073741919
    KP_8 = 1073741920
    KP_9 = 1073741921
    KP_0 = 1073741922
    KP_PERIOD = 1073741923
    APPLICATION = 1073741925
    POWER = 1073741926
    KP_EQUALS = 1073741927
    F13 = 1073741928
    F14 = 1073741929
    F15 = 1073741930
    F16 = 1073741931
    F17 = 1073741932
    F18 = 1073741933
    F19 = 1073741934
    F20 = 1073741935
    F21 = 1073741936
    F22 = 1073741937
    F23 = 1073741938
    F24 = 1073741939
    EXECUTE = 1073741940
    HELP = 1073741941
    MENU = 1073741942
    SELECT = 1073741943
    STOP = 1073741944
    AGAIN = 1073741945
    UNDO = 1073741946
    CUT = 1073741947
    COPY = 1073741948
    PASTE = 1073741949
    FIND = 1073741950
    MUTE = 1073741951
    VOLUMEUP = 1073741952
    VOLUMEDOWN = 1073741953
    KP_COMMA = 1073741957
    KP_EQUALSAS400 = 1073741958
    ALTERASE = 1073741977
    SYSREQ = 1073741978
    CANCEL = 1073741979
    CLEAR = 1073741980
    PRIOR = 1073741981
    RETURN2 = 1073741982
    SEPARATOR = 1073741983
    OUT = 1073741984
    OPER = 1073741985
    CLEARAGAIN = 1073741986
    CRSEL = 1073741987
    EXSEL = 1073741988
    KP_00 = 1073742000
    KP_000 = 1073742001
    THOUSANDSSEPARATOR = 1073742002
    DECIMALSEPARATOR = 1073742003
    CURRENCYUNIT = 1073742004
    CURRENCYSUBUNIT = 1073742005
    KP_LEFTPAREN = 1073742006
    KP_RIGHTPAREN = 1073742007
    KP_LEFTBRACE = 1073742008
    KP_RIGHTBRACE = 1073742009
    KP_TAB = 1073742010
    KP_BACKSPACE = 1073742011
    KP_A = 1073742012
    KP_B = 1073742013
    KP_C = 1073742014
    KP_D = 1073742015
    KP_E = 1073742016
    KP_F = 1073742017
    KP_XOR = 1073742018
    KP_POWER = 1073742019
    KP_PERCENT = 1073742020
    KP_LESS = 1073742021
    KP_GREATER = 1073742022
    KP_AMPERSAND = 1073742023
    KP_DBLAMPERSAND = 1073742024
    KP_VERTICALBAR = 1073742025
    KP_DBLVERTICALBAR = 1073742026
    KP_COLON = 1073742027
    KP_HASH = 1073742028
    KP_SPACE = 1073742029
    KP_AT = 1073742030
    KP_EXCLAM = 1073742031
    KP_MEMSTORE = 1073742032
    KP_MEMRECALL = 1073742033
    KP_MEMCLEAR = 1073742034
    KP_MEMADD = 1073742035
    KP_MEMSUBTRACT = 1073742036
    KP_MEMMULTIPLY = 1073742037
    KP_MEMDIVIDE = 1073742038
    KP_PLUSMINUS = 1073742039
    KP_CLEAR = 1073742040
    KP_CLEARENTRY = 1073742041
    KP_BINARY = 1073742042
    KP_OCTAL = 1073742043
    KP_DECIMAL = 1073742044
    KP_HEXADECIMAL = 1073742045
    LCTRL = 1073742048
    LSHIFT = 1073742049
    LALT = 1073742050
    LGUI = 1073742051
    RCTRL = 1073742052
    RSHIFT = 1073742053
    RALT = 1073742054
    RGUI = 1073742055
    MODE = 1073742081
    SLEEP = 1073742082
    WAKE = 1073742083
    CHANNEL_INCREMENT = 1073742084
    CHANNEL_DECREMENT = 1073742085
    MEDIA_PLAY = 1073742086
    MEDIA_PAUSE = 1073742087
    MEDIA_RECORD = 1073742088
    MEDIA_FAST_FORWARD = 1073742089
    MEDIA_REWIND = 1073742090
    MEDIA_NEXT_TRACK = 1073742091
    MEDIA_PREVIOUS_TRACK = 1073742092
    MEDIA_STOP = 1073742093
    MEDIA_EJECT = 1073742094
    MEDIA_PLAY_PAUSE = 1073742095
    MEDIA_SELECT = 1073742096
    AC_NEW = 1073742097
    AC_OPEN = 1073742098
    AC_CLOSE = 1073742099
    AC_EXIT = 1073742100
    AC_SAVE = 1073742101
    AC_PRINT = 1073742102
    AC_PROPERTIES = 1073742103
    AC_SEARCH = 1073742104
    AC_HOME = 1073742105
    AC_BACK = 1073742106
    AC_FORWARD = 1073742107
    AC_STOP = 1073742108
    AC_REFRESH = 1073742109
    AC_BOOKMARKS = 1073742110
    SOFTLEFT = 1073742111
    SOFTRIGHT = 1073742112
    CALL = 1073742113
    ENDCALL = 1073742114
    # --- end ---

    @property
    def label(self) -> str:
        """A human-readable name of a keycode.

        Returns "" if the keycode doesn't have a name.

        Be sure not to confuse this with ``.name``, which will return the enum
        name rather than the human-readable name.

        Example::

            >>> tcod.event.KeySym.F1.label
            'F1'
            >>> tcod.event.KeySym.BACKSPACE.label
            'Backspace'
        """
        return str(ffi.string(lib.SDL_GetKeyName(self.value)), encoding="utf-8")

    @property
    def keysym(self) -> KeySym:
        """Return a keycode from a scancode.

        Returns itself since it is already a :class:`KeySym`.

        .. seealso::
            :any:`Scancode.keysym`
        """
        return self

    @property
    def scancode(self) -> Scancode:
        """Return a scancode from a keycode.

        Based on the current keyboard layout.
        """
        _init_sdl_video()
        return Scancode(lib.SDL_GetScancodeFromKey(self.value, ffi.NULL))

    @classmethod
    def _missing_(cls, value: object) -> KeySym | None:
        if not isinstance(value, int):
            return None
        result = cls(0)
        result._value_ = value
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scancode):
            msg = "Scancode and KeySym enums can not be compared directly. Convert one or the other to the same type."
            raise TypeError(msg)
        return super().__eq__(other)

    def __hash__(self) -> int:
        # __eq__ was defined, so __hash__ must be defined.
        return super().__hash__()

    def __repr__(self) -> str:
        """Return the fully qualified name of this enum."""
        return f"tcod.event.{self.__class__.__name__}.{self.name}"


def __getattr__(name: str) -> int:
    """Migrate deprecated access of event constants."""
    if name.startswith("BUTTON_"):
        replacement = {
            "BUTTON_LEFT": MouseButton.LEFT,
            "BUTTON_MIDDLE": MouseButton.MIDDLE,
            "BUTTON_RIGHT": MouseButton.RIGHT,
            "BUTTON_X1": MouseButton.X1,
            "BUTTON_X2": MouseButton.X2,
            "BUTTON_LMASK": MouseButtonMask.LEFT,
            "BUTTON_MMASK": MouseButtonMask.MIDDLE,
            "BUTTON_RMASK": MouseButtonMask.RIGHT,
            "BUTTON_X1MASK": MouseButtonMask.X1,
            "BUTTON_X2MASK": MouseButtonMask.X2,
        }[name]
        warnings.warn(
            "Key constants have been replaced with enums.\n"
            f"'tcod.event.{name}' should be replaced with 'tcod.event.{replacement!r}'",
            FutureWarning,
            stacklevel=2,
        )
        return replacement

    value: int | None = getattr(tcod.event_constants, name, None)
    if not value:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    if name.startswith("SCANCODE_"):
        scancode = name[9:]
        if scancode.isdigit():
            scancode = f"N{scancode}"
        warnings.warn(
            "Key constants have been replaced with enums.\n"
            f"`tcod.event.{name}` should be replaced with `tcod.event.Scancode.{scancode}`",
            FutureWarning,
            stacklevel=2,
        )
    elif name.startswith("K_"):
        sym = name[2:]
        if sym.isdigit():
            sym = f"N{sym}"
        warnings.warn(
            "Key constants have been replaced with enums.\n"
            f"`tcod.event.{name}` should be replaced with `tcod.event.KeySym.{sym}`",
            FutureWarning,
            stacklevel=2,
        )
    elif name.startswith("KMOD_"):
        modifier = name[5:]
        warnings.warn(
            "Key modifiers have been replaced with the Modifier IntFlag.\n"
            f"`tcod.event.{modifier}` should be replaced with `tcod.event.Modifier.{modifier}`",
            FutureWarning,
            stacklevel=2,
        )
    return value


__all__ = [  # noqa: F405 RUF022
    "Modifier",
    "Point",
    "BUTTON_LEFT",
    "BUTTON_MIDDLE",
    "BUTTON_RIGHT",
    "BUTTON_X1",
    "BUTTON_X2",
    "BUTTON_LMASK",
    "BUTTON_MMASK",
    "BUTTON_RMASK",
    "BUTTON_X1MASK",
    "BUTTON_X2MASK",
    "Event",
    "Quit",
    "KeyboardEvent",
    "KeyDown",
    "KeyUp",
    "MouseMotion",
    "MouseButtonEvent",
    "MouseButtonDown",
    "MouseButtonUp",
    "MouseWheel",
    "TextInput",
    "WindowEvent",
    "WindowMoved",
    "WindowResized",
    "JoystickEvent",
    "JoystickAxis",
    "JoystickBall",
    "JoystickHat",
    "JoystickButton",
    "JoystickDevice",
    "ControllerEvent",
    "ControllerAxis",
    "ControllerButton",
    "ControllerDevice",
    "Undefined",
    "get",
    "wait",
    "get_mouse_state",
    "add_watch",
    "remove_watch",
    "EventDispatch",
    "get_keyboard_state",
    "get_modifier_state",
    "Scancode",
    "KeySym",
    # --- From event_constants.py ---
    "MOUSEWHEEL_NORMAL",
    "MOUSEWHEEL_FLIPPED",
    "MOUSEWHEEL",
]
