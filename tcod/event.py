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
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
)

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


def _describe_bitmask(
    bits: int, table: Mapping[int, str], default: str = "0"
) -> str:
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


def _pixel_to_tile(x: float, y: float) -> Tuple[float, float]:
    """Convert pixel coordinates to tile coordinates."""
    if not lib.TCOD_ctx.engine:
        return 0, 0
    xy = ffi.new("double[2]", (x, y))
    lib.TCOD_sys_pixel_to_tile(xy, xy + 1)
    return xy[0], xy[1]


Point = NamedTuple("Point", [("x", int), ("y", int)])

# manually define names for SDL macros
BUTTON_LEFT = lib.SDL_BUTTON_LEFT
BUTTON_MIDDLE = lib.SDL_BUTTON_MIDDLE
BUTTON_RIGHT = lib.SDL_BUTTON_RIGHT
BUTTON_X1 = lib.SDL_BUTTON_X1
BUTTON_X2 = lib.SDL_BUTTON_X2
BUTTON_LMASK = lib.SDL_BUTTON_LMASK
BUTTON_MMASK = lib.SDL_BUTTON_MMASK
BUTTON_RMASK = lib.SDL_BUTTON_RMASK
BUTTON_X1MASK = lib.SDL_BUTTON_X1MASK
BUTTON_X2MASK = lib.SDL_BUTTON_X2MASK

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
_REVERSE_BUTTON_MASK_TABLE_PREFIX = _ConstantsWithPrefix(
    _REVERSE_BUTTON_MASK_TABLE
)
_REVERSE_SCANCODE_TABLE_PREFIX = _ConstantsWithPrefix(
    tcod.event_constants._REVERSE_SCANCODE_TABLE
)
_REVERSE_SYM_TABLE_PREFIX = _ConstantsWithPrefix(
    tcod.event_constants._REVERSE_SYM_TABLE
)


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
        scancode (int): The keyboard scan-code, this is the physical location
                        of the key on the keyboard rather than the keys symbol.
        sym (int): The keyboard symbol.
        mod (int): A bitmask of the currently held modifier keys.

            You can use the following to check if a modifier key is held:

            * `tcod.event.KMOD_LSHIFT`
                Left shift bit.
            * `tcod.event.KMOD_RSHIFT`
                Right shift bit.
            * `tcod.event.KMOD_LCTRL`
                Left control bit.
            * `tcod.event.KMOD_RCTRL`
                Right control bit.
            * `tcod.event.KMOD_LALT`
                Left alt bit.
            * `tcod.event.KMOD_RALT`
                Right alt bit.
            * `tcod.event.KMOD_LGUI`
                Left meta key bit.
            * `tcod.event.KMOD_RGUI`
                Right meta key bit.
            * `tcod.event.KMOD_SHIFT`
                ``tcod.event.KMOD_LSHIFT | tcod.event.KMOD_RSHIFT``
            * `tcod.event.KMOD_CTRL`
                ``tcod.event.KMOD_LCTRL | tcod.event.KMOD_RCTRL``
            * `tcod.event.KMOD_ALT`
                ``tcod.event.KMOD_LALT | tcod.event.KMOD_RALT``
            * `tcod.event.KMOD_GUI`
                ``tcod.event.KMOD_LGUI | tcod.event.KMOD_RGUI``
            * `tcod.event.KMOD_NUM`
                Num lock bit.
            * `tcod.event.KMOD_CAPS`
                Caps lock bit.
            * `tcod.event.KMOD_MODE`
                AltGr key bit.

            For example, if shift is held then
            ``event.mod & tcod.event.KMOD_SHIFT`` will evaluate to a true
            value.

        repeat (bool): True if this event exists because of key repeat.
    """

    def __init__(
        self, scancode: int, sym: int, mod: int, repeat: bool = False
    ):
        super().__init__()
        self.scancode = scancode
        self.sym = sym
        self.mod = mod
        self.repeat = repeat

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> Any:
        keysym = sdl_event.key.keysym
        self = cls(
            keysym.scancode, keysym.sym, keysym.mod, bool(sdl_event.key.repeat)
        )
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
            self._scancode_constant(
                tcod.event_constants._REVERSE_SCANCODE_TABLE
            ),
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
        tile: Tuple[int, int] = (0, 0),
        state: int = 0,
    ):
        super().__init__()
        self.pixel = Point(*pixel)
        self.tile = Point(*tile)
        self.state = state

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
        tile: Tuple[int, int] = (0, 0),
        tile_motion: Tuple[int, int] = (0, 0),
        state: int = 0,
    ):
        super().__init__(pixel, tile, state)
        self.pixel_motion = Point(*pixel_motion)
        self.tile_motion = Point(*tile_motion)

    @classmethod
    def from_sdl_event(cls, sdl_event: Any) -> "MouseMotion":
        motion = sdl_event.motion

        pixel = motion.x, motion.y
        pixel_motion = motion.xrel, motion.yrel
        subtile = _pixel_to_tile(*pixel)
        tile = int(subtile[0]), int(subtile[1])
        prev_pixel = pixel[0] - pixel_motion[0], pixel[1] - pixel_motion[1]
        prev_subtile = _pixel_to_tile(*prev_pixel)
        prev_tile = int(prev_subtile[0]), int(prev_subtile[1])
        tile_motion = tile[0] - prev_tile[0], tile[1] - prev_tile[1]
        self = cls(pixel, pixel_motion, tile, tile_motion, motion.state)
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return (
            "tcod.event.%s(pixel=%r, pixel_motion=%r, "
            "tile=%r, tile_motion=%r, state=%s)"
        ) % (
            self.__class__.__name__,
            tuple(self.pixel),
            tuple(self.pixel_motion),
            tuple(self.tile),
            tuple(self.tile_motion),
            _describe_bitmask(self.state, _REVERSE_BUTTON_MASK_TABLE_PREFIX),
        )

    def __str__(self) -> str:
        return (
            "<%s, pixel_motion=(x=%i, y=%i), tile_motion=(x=%i, y=%i)>"
        ) % (
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
        tile: Tuple[int, int] = (0, 0),
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
        return (
            "<type=%r, pixel=(x=%i, y=%i), tile=(x=%i, y=%i), button=%s)"
            % (
                self.type,
                *self.pixel,
                *self.tile,
                _REVERSE_BUTTON_TABLE[self.button],
            )
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
            self = WindowResized(
                event_type, sdl_event.window.data1, sdl_event.window.data2
            )
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
        x (int): Movement on the y-axis.
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
        func = getattr(
            self, "ev_%s" % (event.type.lower(),)
        )  # type: Callable[[Any], Optional[T]]
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

    def ev_mousebuttondown(
        self, event: "tcod.event.MouseButtonDown"
    ) -> Optional[T]:
        """Called when a mouse button is pressed."""

    def ev_mousebuttonup(
        self, event: "tcod.event.MouseButtonUp"
    ) -> Optional[T]:
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

    def ev_windowresized(
        self, event: "tcod.event.WindowResized"
    ) -> Optional[T]:
        """Called when the window is resized."""

    def ev_windowsizechanged(
        self, event: "tcod.event.WindowResized"
    ) -> Optional[T]:
        """Called when the system or user changes the size of the window."""

    def ev_windowminimized(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
        """Called when the window is minimized."""

    def ev_windowmaximized(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
        """Called when the window is maximized."""

    def ev_windowrestored(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
        """Called when the window is restored."""

    def ev_windowenter(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window gains mouse focus."""

    def ev_windowleave(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window loses mouse focus."""

    def ev_windowfocusgained(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
        """Called when the window gains keyboard focus."""

    def ev_windowfocuslost(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
        """Called when the window loses keyboard focus."""

    def ev_windowclose(self, event: "tcod.event.WindowEvent") -> Optional[T]:
        """Called when the window manager requests the window to be closed."""

    def ev_windowtakefocus(
        self, event: "tcod.event.WindowEvent"
    ) -> Optional[T]:
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
    x, y = _pixel_to_tile(*xy)
    return MouseState((xy[0], xy[1]), (int(x), int(y)), buttons)


@ffi.def_extern()  # type: ignore
def _pycall_event_watch(userdata: Any, sdl_event: Any) -> int:
    return 0


__all__ = [  # noqa: F405
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
