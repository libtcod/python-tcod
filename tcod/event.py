"""
A light-weight implementation of event handling built on calls to SDL.

Many event constants are derived directly from SDL.
For example: ``tcod.event.K_UP`` and ``tcod.event.SCANCODE_A`` refer to
SDL's ``SDLK_UP`` and ``SDL_SCANCODE_A`` respectfully.
`See this table for all of SDL's keyboard constants.
<https://wiki.libsdl.org/SDL_Keycode>`_

Printing any event will tell you its attributes in a human readable format.
An events type attribute if omitted is just the classes name with all letters
upper-case.

As a general guideline, you should use :any:`KeyboardEvent.sym` for command
inputs, and :any:`TextInput.text` for name entry fields.

Remember to add the line ``import tcod.event``, as importing this module is not
implied by ``import tcod``.

.. versionadded:: 8.4
"""
import enum
import warnings
from typing import Any, Callable, Dict, Generic, Iterator, Mapping, NamedTuple, Optional, Tuple, TypeVar

import numpy as np

import tcod.event_constants
from tcod.event_constants import *  # noqa: F4
from tcod.event_constants import KMOD_ALT, KMOD_CTRL, KMOD_GUI, KMOD_SHIFT
from tcod.loader import ffi, lib

T = TypeVar("T")


class _ConstantsWithPrefix(Mapping[int, str]):
    def __init__(self, constants: Mapping[int, str]):
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


def _pixel_to_tile(x: float, y: float) -> Optional[Tuple[float, float]]:
    """Convert pixel coordinates to tile coordinates."""
    if not lib.TCOD_ctx.engine:
        return None
    xy = ffi.new("double[2]", (x, y))
    lib.TCOD_sys_pixel_to_tile(xy, xy + 1)
    return xy[0], xy[1]


Point = NamedTuple("Point", [("x", int), ("y", int)])


def _verify_tile_coordinates(xy: Optional[Point]) -> Point:
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


_is_sdl_video_initialized = False


def _init_sdl_video() -> None:
    """Keyboard layout stuff needs SDL to be initialized first."""
    global _is_sdl_video_initialized
    if _is_sdl_video_initialized:
        return
    lib.SDL_InitSubSystem(lib.SDL_INIT_VIDEO)
    _is_sdl_video_initialized = True


class Modifier(enum.IntFlag):
    """Keyboard modifier flags, a bitfield of held modifier keys.

    Use `bitwise and` to check if a modifier key is held.

    The following example shows some common ways of checking modifiers.
    All non-zero return values are considered true.

    Example::

        >>> mod = tcod.event.Modifier(4098)
        >>> mod
        <Modifier.NUM|RSHIFT: 4098>
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


# manually define names for SDL macros
BUTTON_LEFT = 1
BUTTON_MIDDLE = 2
BUTTON_RIGHT = 3
BUTTON_X1 = 4
BUTTON_X2 = 5
BUTTON_LMASK = 0x1
BUTTON_MMASK = 0x2
BUTTON_RMASK = 0x4
BUTTON_X1MASK = 0x8
BUTTON_X2MASK = 0x10

# reverse tables are used to get the tcod.event name from the value.
_REVERSE_BUTTON_TABLE = {
    BUTTON_LEFT: "BUTTON_LEFT",
    BUTTON_MIDDLE: "BUTTON_MIDDLE",
    BUTTON_RIGHT: "BUTTON_RIGHT",
    BUTTON_X1: "BUTTON_X1",
    BUTTON_X2: "BUTTON_X2",
}

_REVERSE_BUTTON_MASK_TABLE = {
    BUTTON_LMASK: "BUTTON_LMASK",
    BUTTON_MMASK: "BUTTON_MMASK",
    BUTTON_RMASK: "BUTTON_RMASK",
    BUTTON_X1MASK: "BUTTON_X1MASK",
    BUTTON_X2MASK: "BUTTON_X2MASK",
}

_REVERSE_BUTTON_TABLE_PREFIX = _ConstantsWithPrefix(_REVERSE_BUTTON_TABLE)
_REVERSE_BUTTON_MASK_TABLE_PREFIX = _ConstantsWithPrefix(_REVERSE_BUTTON_MASK_TABLE)
_REVERSE_SCANCODE_TABLE_PREFIX = _ConstantsWithPrefix(tcod.event_constants._REVERSE_SCANCODE_TABLE)
_REVERSE_SYM_TABLE_PREFIX = _ConstantsWithPrefix(tcod.event_constants._REVERSE_SYM_TABLE)


_REVERSE_MOD_TABLE = tcod.event_constants._REVERSE_MOD_TABLE.copy()
del _REVERSE_MOD_TABLE[KMOD_SHIFT]
del _REVERSE_MOD_TABLE[KMOD_CTRL]
del _REVERSE_MOD_TABLE[KMOD_ALT]
del _REVERSE_MOD_TABLE[KMOD_GUI]

_REVERSE_MOD_TABLE_PREFIX = _ConstantsWithPrefix(_REVERSE_MOD_TABLE)


class Event:
    """The base event class.

    Attributes:
        type (str): This events type.
        sdl_event: When available, this holds a python-cffi 'SDL_Event*'
                   pointer.  All sub-classes have this attribute.
    """

    def __init__(self, type: Optional[str] = None):
        if type is None:
            type = self.__class__.__name__.upper()
        self.type = type
        self.sdl_event = None

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        """Return a class instance from a python-cffi 'SDL_Event*' pointer."""
        raise NotImplementedError()

    def __str__(self) -> str:
        return "<type=%r>" % (self.type,)


class Quit(Event):
    """An application quit request event.

    For more info on when this event is triggered see:
    https://wiki.libsdl.org/SDL_EventType#SDL_QUIT

    Attributes:
        type (str): Always "QUIT".
    """

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "Quit":
        self = cls()
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s()" % (self.__class__.__name__,)


class KeyboardEvent(Event):
    """
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

    def __init__(self, scancode: int, sym: int, mod: int, repeat: bool = False):
        super().__init__()
        self.scancode = Scancode(scancode)
        self.sym = KeySym(sym)
        self.mod = Modifier(mod)
        self.repeat = repeat

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        keysym = sdl_event.key.keysym
        self = cls(keysym.scancode, keysym.sym, keysym.mod, bool(sdl_event.key.repeat))
        self.sdl_event = sdl_event
        return self

    def _scancode_constant(self, table: Mapping[int, str]) -> str:
        """Return the constant name for this scan-code from a table."""
        try:
            return table[self.scancode]
        except KeyError:
            return str(self.scancode)

    def _sym_constant(self, table: Mapping[int, str]) -> str:
        """Return the constant name for this symbol from a table."""
        try:
            return table[self.sym]
        except KeyError:
            return str(self.sym)

    def __repr__(self) -> str:
        return "tcod.event.%s(scancode=%s, sym=%s, mod=%s%s)" % (
            self.__class__.__name__,
            self._scancode_constant(_REVERSE_SCANCODE_TABLE_PREFIX),
            self._sym_constant(_REVERSE_SYM_TABLE_PREFIX),
            _describe_bitmask(self.mod, _REVERSE_MOD_TABLE_PREFIX),
            ", repeat=True" if self.repeat else "",
        )

    def __str__(self) -> str:
        return "<%s, scancode=%s, sym=%s, mod=%s, repeat=%r>" % (
            super().__str__().strip("<>"),
            self._scancode_constant(tcod.event_constants._REVERSE_SCANCODE_TABLE),
            self._sym_constant(tcod.event_constants._REVERSE_SYM_TABLE),
            _describe_bitmask(self.mod, _REVERSE_MOD_TABLE),
            self.repeat,
        )


class KeyDown(KeyboardEvent):
    pass


class KeyUp(KeyboardEvent):
    pass


class MouseState(Event):
    """
    Attributes:
        type (str): Always "MOUSESTATE".
        pixel (Point): The pixel coordinates of the mouse.
        tile (Point): The integer tile coordinates of the mouse on the screen.
        state (int): A bitmask of which mouse buttons are currently held.

            Will be a combination of the following names:

            * tcod.event.BUTTON_LMASK
            * tcod.event.BUTTON_MMASK
            * tcod.event.BUTTON_RMASK
            * tcod.event.BUTTON_X1MASK
            * tcod.event.BUTTON_X2MASK

    .. versionadded:: 9.3
    """

    def __init__(
        self,
        pixel: Tuple[int, int] = (0, 0),
        tile: Optional[Tuple[int, int]] = (0, 0),
        state: int = 0,
    ):
        super().__init__()
        self.pixel = Point(*pixel)
        self.__tile = Point(*tile) if tile is not None else None
        self.state = state

    @property
    def tile(self) -> Point:
        return _verify_tile_coordinates(self.__tile)

    @tile.setter
    def tile(self, xy: Tuple[int, int]) -> None:
        self.__tile = Point(*xy)

    def __repr__(self) -> str:
        return ("tcod.event.%s(pixel=%r, tile=%r, state=%s)") % (
            self.__class__.__name__,
            tuple(self.pixel),
            tuple(self.tile),
            _describe_bitmask(self.state, _REVERSE_BUTTON_MASK_TABLE_PREFIX),
        )

    def __str__(self) -> str:
        return ("<%s, pixel=(x=%i, y=%i), tile=(x=%i, y=%i), state=%s>") % (
            super().__str__().strip("<>"),
            *self.pixel,
            *self.tile,
            _describe_bitmask(self.state, _REVERSE_BUTTON_MASK_TABLE),
        )


class MouseMotion(MouseState):
    """
    Attributes:
        type (str): Always "MOUSEMOTION".
        pixel (Point): The pixel coordinates of the mouse.
        pixel_motion (Point): The pixel delta.
        tile (Point): The integer tile coordinates of the mouse on the screen.
        tile_motion (Point): The integer tile delta.
        state (int): A bitmask of which mouse buttons are currently held.

            Will be a combination of the following names:

            * tcod.event.BUTTON_LMASK
            * tcod.event.BUTTON_MMASK
            * tcod.event.BUTTON_RMASK
            * tcod.event.BUTTON_X1MASK
            * tcod.event.BUTTON_X2MASK
    """

    def __init__(
        self,
        pixel: Tuple[int, int] = (0, 0),
        pixel_motion: Tuple[int, int] = (0, 0),
        tile: Optional[Tuple[int, int]] = (0, 0),
        tile_motion: Optional[Tuple[int, int]] = (0, 0),
        state: int = 0,
    ):
        super().__init__(pixel, tile, state)
        self.pixel_motion = Point(*pixel_motion)
        self.__tile_motion = Point(*tile_motion) if tile_motion is not None else None

    @property
    def tile_motion(self) -> Point:
        return _verify_tile_coordinates(self.__tile_motion)

    @tile_motion.setter
    def tile_motion(self, xy: Tuple[int, int]) -> None:
        self.__tile_motion = Point(*xy)

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "MouseMotion":
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
        return ("tcod.event.%s(pixel=%r, pixel_motion=%r, " "tile=%r, tile_motion=%r, state=%s)") % (
            self.__class__.__name__,
            tuple(self.pixel),
            tuple(self.pixel_motion),
            tuple(self.tile),
            tuple(self.tile_motion),
            _describe_bitmask(self.state, _REVERSE_BUTTON_MASK_TABLE_PREFIX),
        )

    def __str__(self) -> str:
        return ("<%s, pixel_motion=(x=%i, y=%i), tile_motion=(x=%i, y=%i)>") % (
            super().__str__().strip("<>"),
            *self.pixel_motion,
            *self.tile_motion,
        )


class MouseButtonEvent(MouseState):
    """
    Attributes:
        type (str): Will be "MOUSEBUTTONDOWN" or "MOUSEBUTTONUP",
                    depending on the event.
        pixel (Point): The pixel coordinates of the mouse.
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
        pixel: Tuple[int, int] = (0, 0),
        tile: Optional[Tuple[int, int]] = (0, 0),
        button: int = 0,
    ):
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
            tile: Optional[Tuple[int, int]] = None
        else:
            tile = int(subtile[0]), int(subtile[1])
        self = cls(pixel, tile, button.button)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s(pixel=%r, tile=%r, button=%s)" % (
            self.__class__.__name__,
            tuple(self.pixel),
            tuple(self.tile),
            _REVERSE_BUTTON_TABLE_PREFIX[self.button],
        )

    def __str__(self) -> str:
        return "<type=%r, pixel=(x=%i, y=%i), tile=(x=%i, y=%i), button=%s)" % (
            self.type,
            *self.pixel,
            *self.tile,
            _REVERSE_BUTTON_TABLE[self.button],
        )


class MouseButtonDown(MouseButtonEvent):
    """Same as MouseButtonEvent but with ``type="MouseButtonDown"``."""


class MouseButtonUp(MouseButtonEvent):
    """Same as MouseButtonEvent but with ``type="MouseButtonUp"``."""


class MouseWheel(Event):
    """
    Attributes:
        type (str): Always "MOUSEWHEEL".
        x (int): Horizontal scrolling. A positive value means scrolling right.
        y (int): Vertical scrolling. A positive value means scrolling away from
                 the user.
        flipped (bool): If True then the values of `x` and `y` are the opposite
                        of their usual values.  This depends on the settings of
                        the Operating System.
    """

    def __init__(self, x: int, y: int, flipped: bool = False):
        super().__init__()
        self.x = x
        self.y = y
        self.flipped = flipped

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "MouseWheel":
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
    """
    Attributes:
        type (str): Always "TEXTINPUT".
        text (str): A Unicode string with the input.
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "TextInput":
        self = cls(ffi.string(sdl_event.text.text, 32).decode("utf8"))
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s(text=%r)" % (self.__class__.__name__, self.text)

    def __str__(self) -> str:
        return "<%s, text=%r)" % (super().__str__().strip("<>"), self.text)


class WindowEvent(Event):
    """
    Attributes:
        type (str): A window event could mean various event types.
    """

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        if sdl_event.window.event not in cls.__WINDOW_TYPES:
            return Undefined.from_sdl_event(sdl_event)
        event_type = cls.__WINDOW_TYPES[sdl_event.window.event].upper()
        self = None  # type: Any
        if sdl_event.window.event == lib.SDL_WINDOWEVENT_MOVED:
            self = WindowMoved(sdl_event.window.data1, sdl_event.window.data2)
        elif sdl_event.window.event in (
            lib.SDL_WINDOWEVENT_RESIZED,
            lib.SDL_WINDOWEVENT_SIZE_CHANGED,
        ):
            self = WindowResized(event_type, sdl_event.window.data1, sdl_event.window.data2)
        else:
            self = cls(event_type)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s(type=%r)" % (self.__class__.__name__, self.type)

    __WINDOW_TYPES = {
        lib.SDL_WINDOWEVENT_SHOWN: "WindowShown",
        lib.SDL_WINDOWEVENT_HIDDEN: "WindowHidden",
        lib.SDL_WINDOWEVENT_EXPOSED: "WindowExposed",
        lib.SDL_WINDOWEVENT_MOVED: "WindowMoved",
        lib.SDL_WINDOWEVENT_RESIZED: "WindowResized",
        lib.SDL_WINDOWEVENT_SIZE_CHANGED: "WindowSizeChanged",
        lib.SDL_WINDOWEVENT_MINIMIZED: "WindowMinimized",
        lib.SDL_WINDOWEVENT_MAXIMIZED: "WindowMaximized",
        lib.SDL_WINDOWEVENT_RESTORED: "WindowRestored",
        lib.SDL_WINDOWEVENT_ENTER: "WindowEnter",
        lib.SDL_WINDOWEVENT_LEAVE: "WindowLeave",
        lib.SDL_WINDOWEVENT_FOCUS_GAINED: "WindowFocusGained",
        lib.SDL_WINDOWEVENT_FOCUS_LOST: "WindowFocusLost",
        lib.SDL_WINDOWEVENT_CLOSE: "WindowClose",
        lib.SDL_WINDOWEVENT_TAKE_FOCUS: "WindowTakeFocus",
        lib.SDL_WINDOWEVENT_HIT_TEST: "WindowHitTest",
    }


class WindowMoved(WindowEvent):
    """
    Attributes:
        type (str): Always "WINDOWMOVED".
        x (int): Movement on the x-axis.
        y (int): Movement on the y-axis.
    """

    def __init__(self, x: int, y: int) -> None:
        super().__init__(None)
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "tcod.event.%s(type=%r, x=%r, y=%r)" % (
            self.__class__.__name__,
            self.type,
            self.x,
            self.y,
        )

    def __str__(self) -> str:
        return "<%s, x=%r, y=%r)" % (
            super().__str__().strip("<>"),
            self.x,
            self.y,
        )


class WindowResized(WindowEvent):
    """
    Attributes:
        type (str): "WINDOWRESIZED" or "WINDOWSIZECHANGED"
        width (int): The current width of the window.
        height (int): The current height of the window.
    """

    def __init__(self, type: str, width: int, height: int) -> None:
        super().__init__(type)
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return "tcod.event.%s(type=%r, width=%r, height=%r)" % (
            self.__class__.__name__,
            self.type,
            self.width,
            self.height,
        )

    def __str__(self) -> str:
        return "<%s, width=%r, height=%r)" % (
            super().__str__().strip("<>"),
            self.width,
            self.height,
        )


class Undefined(Event):
    """This class is a place holder for SDL events without their own tcod.event
    class.
    """

    def __init__(self) -> None:
        super().__init__("")

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "Undefined":
        self = cls()
        self.sdl_event = sdl_event
        return self

    def __str__(self) -> str:
        if self.sdl_event:
            return "<Undefined sdl_event.type=%i>" % self.sdl_event.type
        return "<Undefined>"


_SDL_TO_CLASS_TABLE = {
    lib.SDL_QUIT: Quit,
    lib.SDL_KEYDOWN: KeyDown,
    lib.SDL_KEYUP: KeyUp,
    lib.SDL_MOUSEMOTION: MouseMotion,
    lib.SDL_MOUSEBUTTONDOWN: MouseButtonDown,
    lib.SDL_MOUSEBUTTONUP: MouseButtonUp,
    lib.SDL_MOUSEWHEEL: MouseWheel,
    lib.SDL_TEXTINPUT: TextInput,
    lib.SDL_WINDOWEVENT: WindowEvent,
}  # type: Dict[int, Any]


def get() -> Iterator[Any]:
    """Return an iterator for all pending events.

    Events are processed as the iterator is consumed.  Breaking out of, or
    discarding the iterator will leave the remaining events on the event queue.
    It is also safe to call this function inside of a loop that is already
    handling events (the event iterator is reentrant.)

    Example::

        for event in tcod.event.get():
            if event.type == "QUIT":
                print(event)
                raise SystemExit()
            elif event.type == "KEYDOWN":
                print(event)
            elif event.type == "MOUSEBUTTONDOWN":
                print(event)
            elif event.type == "MOUSEMOTION":
                print(event)
            else:
                print(event)
        # For loop exits after all current events are processed.
    """
    sdl_event = ffi.new("SDL_Event*")
    while lib.SDL_PollEvent(sdl_event):
        if sdl_event.type in _SDL_TO_CLASS_TABLE:
            yield _SDL_TO_CLASS_TABLE[sdl_event.type].from_sdl_event(sdl_event)
        else:
            yield Undefined.from_sdl_event(sdl_event)


def wait(timeout: Optional[float] = None) -> Iterator[Any]:
    """Block until events exist, then return an event iterator.

    `timeout` is the maximum number of seconds to wait as a floating point
    number with millisecond precision, or it can be None to wait forever.

    Returns the same iterator as a call to :any:`tcod.event.get`.

    Example::

        for event in tcod.event.wait():
            if event.type == "QUIT":
                print(event)
                raise SystemExit()
            elif event.type == "KEYDOWN":
                print(event)
            elif event.type == "MOUSEBUTTONDOWN":
                print(event)
            elif event.type == "MOUSEMOTION":
                print(event)
            else:
                print(event)
        # For loop exits on timeout or after at least one event is processed.
    """
    if timeout is not None:
        lib.SDL_WaitEventTimeout(ffi.NULL, int(timeout * 1000))
    else:
        lib.SDL_WaitEvent(ffi.NULL)
    return get()


class EventDispatch(Generic[T]):
    '''This class dispatches events to methods depending on the events type
    attribute.

    To use this class, make a sub-class and override the relevant `ev_*`
    methods.  Then send events to the dispatch method.

    .. versionchanged:: 11.12
        This is now a generic class.  The type hists at the return value of
        :any:`dispatch` and the `ev_*` methods.

    Example::

        import tcod

        MOVE_KEYS = {  # key_symbol: (x, y)
            # Arrow keys.
            tcod.event.K_LEFT: (-1, 0),
            tcod.event.K_RIGHT: (1, 0),
            tcod.event.K_UP: (0, -1),
            tcod.event.K_DOWN: (0, 1),
            tcod.event.K_HOME: (-1, -1),
            tcod.event.K_END: (-1, 1),
            tcod.event.K_PAGEUP: (1, -1),
            tcod.event.K_PAGEDOWN: (1, 1),
            tcod.event.K_PERIOD: (0, 0),
            # Numpad keys.
            tcod.event.K_KP_1: (-1, 1),
            tcod.event.K_KP_2: (0, 1),
            tcod.event.K_KP_3: (1, 1),
            tcod.event.K_KP_4: (-1, 0),
            tcod.event.K_KP_5: (0, 0),
            tcod.event.K_KP_6: (1, 0),
            tcod.event.K_KP_7: (-1, -1),
            tcod.event.K_KP_8: (0, -1),
            tcod.event.K_KP_9: (1, -1),
            tcod.event.K_CLEAR: (0, 0),  # Numpad `clear` key.
            # Vi Keys.
            tcod.event.K_h: (-1, 0),
            tcod.event.K_j: (0, 1),
            tcod.event.K_k: (0, -1),
            tcod.event.K_l: (1, 0),
            tcod.event.K_y: (-1, -1),
            tcod.event.K_u: (1, -1),
            tcod.event.K_b: (-1, 1),
            tcod.event.K_n: (1, 1),
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
                elif event.sym == tcod.event.K_ESCAPE:
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


        root_console = tcod.console_init_root(80, 60)
        state = State()
        while True:
            tcod.console_flush()
            for event in tcod.event.wait():
                state.dispatch(event)
    '''  # noqa: E501

    def dispatch(self, event: Any) -> Optional[T]:
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
        func = getattr(self, "ev_%s" % (event.type.lower(),))  # type: Callable[[Any], Optional[T]]
        return func(event)

    def event_get(self) -> None:
        for event in get():
            self.dispatch(event)

    def event_wait(self, timeout: Optional[float]) -> None:
        wait(timeout)
        self.event_get()

    def ev_quit(self, event: "tcod.event.Quit") -> Optional[T]:
        """Called when the termination of the program is requested."""

    def ev_keydown(self, event: "tcod.event.KeyDown") -> Optional[T]:
        """Called when a keyboard key is pressed or repeated."""

    def ev_keyup(self, event: "tcod.event.KeyUp") -> Optional[T]:
        """Called when a keyboard key is released."""

    def ev_mousemotion(self, event: "tcod.event.MouseMotion") -> Optional[T]:
        """Called when the mouse is moved."""

    def ev_mousebuttondown(self, event: "tcod.event.MouseButtonDown") -> Optional[T]:
        """Called when a mouse button is pressed."""

    def ev_mousebuttonup(self, event: "tcod.event.MouseButtonUp") -> Optional[T]:
        """Called when a mouse button is released."""

    def ev_mousewheel(self, event: "tcod.event.MouseWheel") -> Optional[T]:
        """Called when the mouse wheel is scrolled."""

    def ev_textinput(self, event: "tcod.event.TextInput") -> Optional[T]:
        """Called to handle Unicode input."""

    def ev_windowshown(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window is shown."""

    def ev_windowhidden(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window is hidden."""

    def ev_windowexposed(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when a window is exposed, and needs to be refreshed.

        This usually means a call to :any:`tcod.console_flush` is necessary.
        """

    def ev_windowmoved(self, event: "tcod.event.WindowMoved") -> Optional[T]:
        """Called when the window is moved."""

    def ev_windowresized(self, event: "tcod.event.WindowResized") -> Optional[T]:
        """Called when the window is resized."""

    def ev_windowsizechanged(self, event: "tcod.event.WindowResized") -> Optional[T]:
        """Called when the system or user changes the size of the window."""

    def ev_windowminimized(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window is minimized."""

    def ev_windowmaximized(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window is maximized."""

    def ev_windowrestored(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window is restored."""

    def ev_windowenter(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window gains mouse focus."""

    def ev_windowleave(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window loses mouse focus."""

    def ev_windowfocusgained(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window gains keyboard focus."""

    def ev_windowfocuslost(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window loses keyboard focus."""

    def ev_windowclose(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window manager requests the window to be closed."""

    def ev_windowtakefocus(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        pass

    def ev_windowhittest(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        pass

    def ev_(self, event: Any) -> Optional[T]:
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


@ffi.def_extern()  # type: ignore
def _pycall_event_watch(userdata: Any, sdl_event: Any) -> int:
    return 0


def get_keyboard_state() -> "np.ndarray[Any, np.dtype[np.bool_]]":
    """Return a boolean array with the current keyboard state.

    Index this array with a scancode.  The value will be True if the key is
    currently held.

    Example::

        state = tcod.event.get_keyboard_state()

        # Get a WASD movement vector:
        x = int(state[tcod.event.Scancode.E]) - int(state[tcod.event.Scancode.A])
        y = int(state[tcod.event.Scancode.S]) - int(state[tcod.event.Scancode.W])

        # Key with 'z' glyph is held:
        is_z_held = state[tcod.event.KeySym.z.scancode]


    .. versionadded:: 12.3
    """
    numkeys = ffi.new("int[1]")
    keyboard_state = lib.SDL_GetKeyboardState(numkeys)
    out: np.ndarray[Any, Any] = np.frombuffer(ffi.buffer(keyboard_state[0 : numkeys[0]]), dtype=bool)  # type: ignore
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
    AUDIONEXT = 258
    AUDIOPREV = 259
    AUDIOSTOP = 260
    AUDIOPLAY = 261
    AUDIOMUTE = 262
    MEDIASELECT = 263
    WWW = 264
    MAIL = 265
    CALCULATOR = 266
    COMPUTER = 267
    AC_SEARCH = 268
    AC_HOME = 269
    AC_BACK = 270
    AC_FORWARD = 271
    AC_STOP = 272
    AC_REFRESH = 273
    AC_BOOKMARKS = 274
    BRIGHTNESSDOWN = 275
    BRIGHTNESSUP = 276
    DISPLAYSWITCH = 277
    KBDILLUMTOGGLE = 278
    KBDILLUMDOWN = 279
    KBDILLUMUP = 280
    EJECT = 281
    SLEEP = 282
    APP1 = 283
    APP2 = 284
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
    def keysym(self) -> "KeySym":
        """Return a :class:`KeySym` from a scancode.

        Based on the current keyboard layout.
        """
        _init_sdl_video()
        return KeySym(lib.SDL_GetKeyFromScancode(self.value))

    @property
    def scancode(self) -> "Scancode":
        """Return a scancode from a keycode.

        Returns itself since it is already a :class:`Scancode`.

        .. seealso::
            :any:`KeySym.scancode`
        """
        return self

    @classmethod
    def _missing_(cls, value: object) -> "Optional[Scancode]":
        if not isinstance(value, int):
            return None
        result = cls(0)
        result._value_ = value
        return result

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, KeySym):
            raise TypeError(
                "Scancode and KeySym enums can not be compared directly." " Convert one or the other to the same type."
            )
        return super().__eq__(other)

    def __hash__(self) -> int:
        # __eq__ was defined, so __hash__ must be defined.
        return super().__hash__()


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
    QUOTEDBL = 34
    HASH = 35
    DOLLAR = 36
    PERCENT = 37
    AMPERSAND = 38
    QUOTE = 39
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
    BACKQUOTE = 96
    a = 97
    b = 98
    c = 99
    d = 100
    e = 101
    f = 102
    g = 103
    h = 104
    i = 105
    j = 106
    k = 107
    l = 108  # noqa: E741
    m = 109
    n = 110
    o = 111
    p = 112
    q = 113
    r = 114
    s = 115
    t = 116
    u = 117
    v = 118
    w = 119
    x = 120
    y = 121
    z = 122
    DELETE = 127
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
    AUDIONEXT = 1073742082
    AUDIOPREV = 1073742083
    AUDIOSTOP = 1073742084
    AUDIOPLAY = 1073742085
    AUDIOMUTE = 1073742086
    MEDIASELECT = 1073742087
    WWW = 1073742088
    MAIL = 1073742089
    CALCULATOR = 1073742090
    COMPUTER = 1073742091
    AC_SEARCH = 1073742092
    AC_HOME = 1073742093
    AC_BACK = 1073742094
    AC_FORWARD = 1073742095
    AC_STOP = 1073742096
    AC_REFRESH = 1073742097
    AC_BOOKMARKS = 1073742098
    BRIGHTNESSDOWN = 1073742099
    BRIGHTNESSUP = 1073742100
    DISPLAYSWITCH = 1073742101
    KBDILLUMTOGGLE = 1073742102
    KBDILLUMDOWN = 1073742103
    KBDILLUMUP = 1073742104
    EJECT = 1073742105
    SLEEP = 1073742106
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
    def keysym(self) -> "KeySym":
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
        return Scancode(lib.SDL_GetScancodeFromKey(self.value))

    @classmethod
    def _missing_(cls, value: object) -> "Optional[KeySym]":
        if not isinstance(value, int):
            return None
        result = cls(0)
        result._value_ = value
        return result

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Scancode):
            raise TypeError(
                "Scancode and KeySym enums can not be compared directly." " Convert one or the other to the same type."
            )
        return super().__eq__(other)

    def __hash__(self) -> int:
        # __eq__ was defined, so __hash__ must be defined.
        return super().__hash__()


__all__ = [  # noqa: F405
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
    "Undefined",
    "get",
    "wait",
    "get_mouse_state",
    "EventDispatch",
    "get_keyboard_state",
    "get_modifier_state",
    "Scancode",
    "KeySym",
    # --- From event_constants.py ---
    "SCANCODE_UNKNOWN",
    "SCANCODE_A",
    "SCANCODE_B",
    "SCANCODE_C",
    "SCANCODE_D",
    "SCANCODE_E",
    "SCANCODE_F",
    "SCANCODE_G",
    "SCANCODE_H",
    "SCANCODE_I",
    "SCANCODE_J",
    "SCANCODE_K",
    "SCANCODE_L",
    "SCANCODE_M",
    "SCANCODE_N",
    "SCANCODE_O",
    "SCANCODE_P",
    "SCANCODE_Q",
    "SCANCODE_R",
    "SCANCODE_S",
    "SCANCODE_T",
    "SCANCODE_U",
    "SCANCODE_V",
    "SCANCODE_W",
    "SCANCODE_X",
    "SCANCODE_Y",
    "SCANCODE_Z",
    "SCANCODE_1",
    "SCANCODE_2",
    "SCANCODE_3",
    "SCANCODE_4",
    "SCANCODE_5",
    "SCANCODE_6",
    "SCANCODE_7",
    "SCANCODE_8",
    "SCANCODE_9",
    "SCANCODE_0",
    "SCANCODE_RETURN",
    "SCANCODE_ESCAPE",
    "SCANCODE_BACKSPACE",
    "SCANCODE_TAB",
    "SCANCODE_SPACE",
    "SCANCODE_MINUS",
    "SCANCODE_EQUALS",
    "SCANCODE_LEFTBRACKET",
    "SCANCODE_RIGHTBRACKET",
    "SCANCODE_BACKSLASH",
    "SCANCODE_NONUSHASH",
    "SCANCODE_SEMICOLON",
    "SCANCODE_APOSTROPHE",
    "SCANCODE_GRAVE",
    "SCANCODE_COMMA",
    "SCANCODE_PERIOD",
    "SCANCODE_SLASH",
    "SCANCODE_CAPSLOCK",
    "SCANCODE_F1",
    "SCANCODE_F2",
    "SCANCODE_F3",
    "SCANCODE_F4",
    "SCANCODE_F5",
    "SCANCODE_F6",
    "SCANCODE_F7",
    "SCANCODE_F8",
    "SCANCODE_F9",
    "SCANCODE_F10",
    "SCANCODE_F11",
    "SCANCODE_F12",
    "SCANCODE_PRINTSCREEN",
    "SCANCODE_SCROLLLOCK",
    "SCANCODE_PAUSE",
    "SCANCODE_INSERT",
    "SCANCODE_HOME",
    "SCANCODE_PAGEUP",
    "SCANCODE_DELETE",
    "SCANCODE_END",
    "SCANCODE_PAGEDOWN",
    "SCANCODE_RIGHT",
    "SCANCODE_LEFT",
    "SCANCODE_DOWN",
    "SCANCODE_UP",
    "SCANCODE_NUMLOCKCLEAR",
    "SCANCODE_KP_DIVIDE",
    "SCANCODE_KP_MULTIPLY",
    "SCANCODE_KP_MINUS",
    "SCANCODE_KP_PLUS",
    "SCANCODE_KP_ENTER",
    "SCANCODE_KP_1",
    "SCANCODE_KP_2",
    "SCANCODE_KP_3",
    "SCANCODE_KP_4",
    "SCANCODE_KP_5",
    "SCANCODE_KP_6",
    "SCANCODE_KP_7",
    "SCANCODE_KP_8",
    "SCANCODE_KP_9",
    "SCANCODE_KP_0",
    "SCANCODE_KP_PERIOD",
    "SCANCODE_NONUSBACKSLASH",
    "SCANCODE_APPLICATION",
    "SCANCODE_POWER",
    "SCANCODE_KP_EQUALS",
    "SCANCODE_F13",
    "SCANCODE_F14",
    "SCANCODE_F15",
    "SCANCODE_F16",
    "SCANCODE_F17",
    "SCANCODE_F18",
    "SCANCODE_F19",
    "SCANCODE_F20",
    "SCANCODE_F21",
    "SCANCODE_F22",
    "SCANCODE_F23",
    "SCANCODE_F24",
    "SCANCODE_EXECUTE",
    "SCANCODE_HELP",
    "SCANCODE_MENU",
    "SCANCODE_SELECT",
    "SCANCODE_STOP",
    "SCANCODE_AGAIN",
    "SCANCODE_UNDO",
    "SCANCODE_CUT",
    "SCANCODE_COPY",
    "SCANCODE_PASTE",
    "SCANCODE_FIND",
    "SCANCODE_MUTE",
    "SCANCODE_VOLUMEUP",
    "SCANCODE_VOLUMEDOWN",
    "SCANCODE_KP_COMMA",
    "SCANCODE_KP_EQUALSAS400",
    "SCANCODE_INTERNATIONAL1",
    "SCANCODE_INTERNATIONAL2",
    "SCANCODE_INTERNATIONAL3",
    "SCANCODE_INTERNATIONAL4",
    "SCANCODE_INTERNATIONAL5",
    "SCANCODE_INTERNATIONAL6",
    "SCANCODE_INTERNATIONAL7",
    "SCANCODE_INTERNATIONAL8",
    "SCANCODE_INTERNATIONAL9",
    "SCANCODE_LANG1",
    "SCANCODE_LANG2",
    "SCANCODE_LANG3",
    "SCANCODE_LANG4",
    "SCANCODE_LANG5",
    "SCANCODE_LANG6",
    "SCANCODE_LANG7",
    "SCANCODE_LANG8",
    "SCANCODE_LANG9",
    "SCANCODE_ALTERASE",
    "SCANCODE_SYSREQ",
    "SCANCODE_CANCEL",
    "SCANCODE_CLEAR",
    "SCANCODE_PRIOR",
    "SCANCODE_RETURN2",
    "SCANCODE_SEPARATOR",
    "SCANCODE_OUT",
    "SCANCODE_OPER",
    "SCANCODE_CLEARAGAIN",
    "SCANCODE_CRSEL",
    "SCANCODE_EXSEL",
    "SCANCODE_KP_00",
    "SCANCODE_KP_000",
    "SCANCODE_THOUSANDSSEPARATOR",
    "SCANCODE_DECIMALSEPARATOR",
    "SCANCODE_CURRENCYUNIT",
    "SCANCODE_CURRENCYSUBUNIT",
    "SCANCODE_KP_LEFTPAREN",
    "SCANCODE_KP_RIGHTPAREN",
    "SCANCODE_KP_LEFTBRACE",
    "SCANCODE_KP_RIGHTBRACE",
    "SCANCODE_KP_TAB",
    "SCANCODE_KP_BACKSPACE",
    "SCANCODE_KP_A",
    "SCANCODE_KP_B",
    "SCANCODE_KP_C",
    "SCANCODE_KP_D",
    "SCANCODE_KP_E",
    "SCANCODE_KP_F",
    "SCANCODE_KP_XOR",
    "SCANCODE_KP_POWER",
    "SCANCODE_KP_PERCENT",
    "SCANCODE_KP_LESS",
    "SCANCODE_KP_GREATER",
    "SCANCODE_KP_AMPERSAND",
    "SCANCODE_KP_DBLAMPERSAND",
    "SCANCODE_KP_VERTICALBAR",
    "SCANCODE_KP_DBLVERTICALBAR",
    "SCANCODE_KP_COLON",
    "SCANCODE_KP_HASH",
    "SCANCODE_KP_SPACE",
    "SCANCODE_KP_AT",
    "SCANCODE_KP_EXCLAM",
    "SCANCODE_KP_MEMSTORE",
    "SCANCODE_KP_MEMRECALL",
    "SCANCODE_KP_MEMCLEAR",
    "SCANCODE_KP_MEMADD",
    "SCANCODE_KP_MEMSUBTRACT",
    "SCANCODE_KP_MEMMULTIPLY",
    "SCANCODE_KP_MEMDIVIDE",
    "SCANCODE_KP_PLUSMINUS",
    "SCANCODE_KP_CLEAR",
    "SCANCODE_KP_CLEARENTRY",
    "SCANCODE_KP_BINARY",
    "SCANCODE_KP_OCTAL",
    "SCANCODE_KP_DECIMAL",
    "SCANCODE_KP_HEXADECIMAL",
    "SCANCODE_LCTRL",
    "SCANCODE_LSHIFT",
    "SCANCODE_LALT",
    "SCANCODE_LGUI",
    "SCANCODE_RCTRL",
    "SCANCODE_RSHIFT",
    "SCANCODE_RALT",
    "SCANCODE_RGUI",
    "SCANCODE_MODE",
    "SCANCODE_AUDIONEXT",
    "SCANCODE_AUDIOPREV",
    "SCANCODE_AUDIOSTOP",
    "SCANCODE_AUDIOPLAY",
    "SCANCODE_AUDIOMUTE",
    "SCANCODE_MEDIASELECT",
    "SCANCODE_WWW",
    "SCANCODE_MAIL",
    "SCANCODE_CALCULATOR",
    "SCANCODE_COMPUTER",
    "SCANCODE_AC_SEARCH",
    "SCANCODE_AC_HOME",
    "SCANCODE_AC_BACK",
    "SCANCODE_AC_FORWARD",
    "SCANCODE_AC_STOP",
    "SCANCODE_AC_REFRESH",
    "SCANCODE_AC_BOOKMARKS",
    "SCANCODE_BRIGHTNESSDOWN",
    "SCANCODE_BRIGHTNESSUP",
    "SCANCODE_DISPLAYSWITCH",
    "SCANCODE_KBDILLUMTOGGLE",
    "SCANCODE_KBDILLUMDOWN",
    "SCANCODE_KBDILLUMUP",
    "SCANCODE_EJECT",
    "SCANCODE_SLEEP",
    "SCANCODE_APP1",
    "SCANCODE_APP2",
    "K_UNKNOWN",
    "K_BACKSPACE",
    "K_TAB",
    "K_RETURN",
    "K_ESCAPE",
    "K_SPACE",
    "K_EXCLAIM",
    "K_QUOTEDBL",
    "K_HASH",
    "K_DOLLAR",
    "K_PERCENT",
    "K_AMPERSAND",
    "K_QUOTE",
    "K_LEFTPAREN",
    "K_RIGHTPAREN",
    "K_ASTERISK",
    "K_PLUS",
    "K_COMMA",
    "K_MINUS",
    "K_PERIOD",
    "K_SLASH",
    "K_0",
    "K_1",
    "K_2",
    "K_3",
    "K_4",
    "K_5",
    "K_6",
    "K_7",
    "K_8",
    "K_9",
    "K_COLON",
    "K_SEMICOLON",
    "K_LESS",
    "K_EQUALS",
    "K_GREATER",
    "K_QUESTION",
    "K_AT",
    "K_LEFTBRACKET",
    "K_BACKSLASH",
    "K_RIGHTBRACKET",
    "K_CARET",
    "K_UNDERSCORE",
    "K_BACKQUOTE",
    "K_a",
    "K_b",
    "K_c",
    "K_d",
    "K_e",
    "K_f",
    "K_g",
    "K_h",
    "K_i",
    "K_j",
    "K_k",
    "K_l",
    "K_m",
    "K_n",
    "K_o",
    "K_p",
    "K_q",
    "K_r",
    "K_s",
    "K_t",
    "K_u",
    "K_v",
    "K_w",
    "K_x",
    "K_y",
    "K_z",
    "K_DELETE",
    "K_SCANCODE_MASK",
    "K_CAPSLOCK",
    "K_F1",
    "K_F2",
    "K_F3",
    "K_F4",
    "K_F5",
    "K_F6",
    "K_F7",
    "K_F8",
    "K_F9",
    "K_F10",
    "K_F11",
    "K_F12",
    "K_PRINTSCREEN",
    "K_SCROLLLOCK",
    "K_PAUSE",
    "K_INSERT",
    "K_HOME",
    "K_PAGEUP",
    "K_END",
    "K_PAGEDOWN",
    "K_RIGHT",
    "K_LEFT",
    "K_DOWN",
    "K_UP",
    "K_NUMLOCKCLEAR",
    "K_KP_DIVIDE",
    "K_KP_MULTIPLY",
    "K_KP_MINUS",
    "K_KP_PLUS",
    "K_KP_ENTER",
    "K_KP_1",
    "K_KP_2",
    "K_KP_3",
    "K_KP_4",
    "K_KP_5",
    "K_KP_6",
    "K_KP_7",
    "K_KP_8",
    "K_KP_9",
    "K_KP_0",
    "K_KP_PERIOD",
    "K_APPLICATION",
    "K_POWER",
    "K_KP_EQUALS",
    "K_F13",
    "K_F14",
    "K_F15",
    "K_F16",
    "K_F17",
    "K_F18",
    "K_F19",
    "K_F20",
    "K_F21",
    "K_F22",
    "K_F23",
    "K_F24",
    "K_EXECUTE",
    "K_HELP",
    "K_MENU",
    "K_SELECT",
    "K_STOP",
    "K_AGAIN",
    "K_UNDO",
    "K_CUT",
    "K_COPY",
    "K_PASTE",
    "K_FIND",
    "K_MUTE",
    "K_VOLUMEUP",
    "K_VOLUMEDOWN",
    "K_KP_COMMA",
    "K_KP_EQUALSAS400",
    "K_ALTERASE",
    "K_SYSREQ",
    "K_CANCEL",
    "K_CLEAR",
    "K_PRIOR",
    "K_RETURN2",
    "K_SEPARATOR",
    "K_OUT",
    "K_OPER",
    "K_CLEARAGAIN",
    "K_CRSEL",
    "K_EXSEL",
    "K_KP_00",
    "K_KP_000",
    "K_THOUSANDSSEPARATOR",
    "K_DECIMALSEPARATOR",
    "K_CURRENCYUNIT",
    "K_CURRENCYSUBUNIT",
    "K_KP_LEFTPAREN",
    "K_KP_RIGHTPAREN",
    "K_KP_LEFTBRACE",
    "K_KP_RIGHTBRACE",
    "K_KP_TAB",
    "K_KP_BACKSPACE",
    "K_KP_A",
    "K_KP_B",
    "K_KP_C",
    "K_KP_D",
    "K_KP_E",
    "K_KP_F",
    "K_KP_XOR",
    "K_KP_POWER",
    "K_KP_PERCENT",
    "K_KP_LESS",
    "K_KP_GREATER",
    "K_KP_AMPERSAND",
    "K_KP_DBLAMPERSAND",
    "K_KP_VERTICALBAR",
    "K_KP_DBLVERTICALBAR",
    "K_KP_COLON",
    "K_KP_HASH",
    "K_KP_SPACE",
    "K_KP_AT",
    "K_KP_EXCLAM",
    "K_KP_MEMSTORE",
    "K_KP_MEMRECALL",
    "K_KP_MEMCLEAR",
    "K_KP_MEMADD",
    "K_KP_MEMSUBTRACT",
    "K_KP_MEMMULTIPLY",
    "K_KP_MEMDIVIDE",
    "K_KP_PLUSMINUS",
    "K_KP_CLEAR",
    "K_KP_CLEARENTRY",
    "K_KP_BINARY",
    "K_KP_OCTAL",
    "K_KP_DECIMAL",
    "K_KP_HEXADECIMAL",
    "K_LCTRL",
    "K_LSHIFT",
    "K_LALT",
    "K_LGUI",
    "K_RCTRL",
    "K_RSHIFT",
    "K_RALT",
    "K_RGUI",
    "K_MODE",
    "K_AUDIONEXT",
    "K_AUDIOPREV",
    "K_AUDIOSTOP",
    "K_AUDIOPLAY",
    "K_AUDIOMUTE",
    "K_MEDIASELECT",
    "K_WWW",
    "K_MAIL",
    "K_CALCULATOR",
    "K_COMPUTER",
    "K_AC_SEARCH",
    "K_AC_HOME",
    "K_AC_BACK",
    "K_AC_FORWARD",
    "K_AC_STOP",
    "K_AC_REFRESH",
    "K_AC_BOOKMARKS",
    "K_BRIGHTNESSDOWN",
    "K_BRIGHTNESSUP",
    "K_DISPLAYSWITCH",
    "K_KBDILLUMTOGGLE",
    "K_KBDILLUMDOWN",
    "K_KBDILLUMUP",
    "K_EJECT",
    "K_SLEEP",
    "KMOD_NONE",
    "KMOD_LSHIFT",
    "KMOD_RSHIFT",
    "KMOD_SHIFT",
    "KMOD_LCTRL",
    "KMOD_RCTRL",
    "KMOD_CTRL",
    "KMOD_LALT",
    "KMOD_RALT",
    "KMOD_ALT",
    "KMOD_LGUI",
    "KMOD_RGUI",
    "KMOD_GUI",
    "KMOD_NUM",
    "KMOD_CAPS",
    "KMOD_MODE",
    "KMOD_RESERVED",
    "MOUSEWHEEL_NORMAL",
    "MOUSEWHEEL_FLIPPED",
    "MOUSEWHEEL",
]
