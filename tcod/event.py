"""
An alternative, more direct implementation of event handling based on using
cffi calls to SDL functions.  The current code is partially incomplete.

Printing any event will tell you its attributes in a human readable format.
An events type attribute if omitted is just the classes name with all letters
upper-case.  Do not use :any:`isinstance` to tell events apart as that method
won't be forward compatible.

As a general guideline, you should use :any:`KeyboardEvent.sym` for command
inputs, and :any:`TextInput.text` for name entry fields.

Remember to add the line ``import tcod.event``, as importing this module is not
implied by ``import tcod``.

.. versionadded:: 8.4
"""
from typing import Any, Dict, NamedTuple, Optional, Iterator, Tuple

import tcod
import tcod.event_constants
from tcod.event_constants import *  # noqa: F4


def _describe_bitmask(
    bits: int, table: Dict[Any, str], default: str = "0"
) -> str:
    """Returns a bitmask in human readable form.

    This is a private function, used internally.

    Args:
        bits (int): The bitmask to be represented.
        table (Dict[Any,str]): A reverse lookup table.
        default (Any): A default return value when bits is 0.

    Returns: str: A printable version of the bits variable.
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
    xy = tcod.ffi.new("double[2]", (x, y))
    tcod.lib.TCOD_sys_pixel_to_tile(xy, xy + 1)
    return xy[0], xy[1]


Point = NamedTuple("Point", [("x", float), ("y", float)])

# manually define names for SDL macros
BUTTON_LEFT = 1
BUTTON_MIDDLE = 2
BUTTON_RIGHT = 3
BUTTON_X1 = 4
BUTTON_X2 = 5
BUTTON_LMASK = 0x01
BUTTON_MMASK = 0x02
BUTTON_RMASK = 0x04
BUTTON_X1MASK = 0x08
BUTTON_X2MASK = 0x10

KMOD_SHIFT = (
    tcod.event_constants.KMOD_LSHIFT | tcod.event_constants.KMOD_RSHIFT
)
KMOD_CTRL = tcod.event_constants.KMOD_LCTRL | tcod.event_constants.KMOD_RCTRL
KMOD_ALT = tcod.event_constants.KMOD_LALT | tcod.event_constants.KMOD_RALT
KMOD_GUI = tcod.event_constants.KMOD_LGUI | tcod.event_constants.KMOD_RGUI

# reverse tables are used to get the tcod.event name from the value.
_REVERSE_BUTTON_TABLE = {
    BUTTON_LEFT: "tcod.event.BUTTON_LEFT",
    BUTTON_MIDDLE: "tcod.event.BUTTON_MIDDLE",
    BUTTON_RIGHT: "tcod.event.BUTTON_RIGHT",
    BUTTON_X1: "tcod.event.BUTTON_X1",
    BUTTON_X2: "tcod.event.BUTTON_X2",
}

_REVERSE_BUTTON_MASK_TABLE = {
    BUTTON_LMASK: "tcod.event.BUTTON_LMASK",
    BUTTON_MMASK: "tcod.event.BUTTON_MMASK",
    BUTTON_RMASK: "tcod.event.BUTTON_RMASK",
    BUTTON_X1MASK: "tcod.event.BUTTON_X1MASK",
    BUTTON_X2MASK: "tcod.event.BUTTON_X2MASK",
}


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
        return "tcod.event.%s()" % self.__class__.__name__


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

    def __repr__(self) -> str:
        return "tcod.event.%s(scancode=%s, sym=%s, mod=%s%s)" % (
            self.__class__.__name__,
            tcod.event_constants._REVERSE_SCANCODE_TABLE[self.scancode],
            tcod.event_constants._REVERSE_SYM_TABLE[self.sym],
            _describe_bitmask(
                self.mod, tcod.event_constants._REVERSE_MOD_TABLE
            ),
            ", repeat=True" if self.repeat else "",
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

    .. addedversion:: 9.3
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
            self.pixel,
            self.tile,
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
            self.pixel,
            self.pixel_motion,
            self.tile,
            self.tile_motion,
            _describe_bitmask(self.state, _REVERSE_BUTTON_MASK_TABLE),
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
            self.pixel,
            self.tile,
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
        self = cls(tcod.ffi.string(sdl_event.text.text, 32).decode("utf8"))
        self.sdl_event = sdl_event
        return self

    def __repr__(self) -> str:
        return "tcod.event.%s(text=%r)" % (self.__class__.__name__, self.text)


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
        if sdl_event.window.event == tcod.lib.SDL_WINDOWEVENT_MOVED:
            self = WindowMoved(sdl_event.window.data1, sdl_event.window.data2)
        elif sdl_event.window.event in (
            tcod.lib.SDL_WINDOWEVENT_RESIZED,
            tcod.lib.SDL_WINDOWEVENT_SIZE_CHANGED,
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
        tcod.lib.SDL_WINDOWEVENT_SHOWN: "WindowShown",
        tcod.lib.SDL_WINDOWEVENT_HIDDEN: "WindowHidden",
        tcod.lib.SDL_WINDOWEVENT_EXPOSED: "WindowExposed",
        tcod.lib.SDL_WINDOWEVENT_MOVED: "WindowMoved",
        tcod.lib.SDL_WINDOWEVENT_RESIZED: "WindowResized",
        tcod.lib.SDL_WINDOWEVENT_SIZE_CHANGED: "WindowSizeChanged",
        tcod.lib.SDL_WINDOWEVENT_MINIMIZED: "WindowMinimized",
        tcod.lib.SDL_WINDOWEVENT_MAXIMIZED: "WindowMaximized",
        tcod.lib.SDL_WINDOWEVENT_RESTORED: "WindowRestored",
        tcod.lib.SDL_WINDOWEVENT_ENTER: "WindowEnter",
        tcod.lib.SDL_WINDOWEVENT_LEAVE: "WindowLeave",
        tcod.lib.SDL_WINDOWEVENT_FOCUS_GAINED: "WindowFocusGained",
        tcod.lib.SDL_WINDOWEVENT_FOCUS_LOST: "WindowFocusLost",
        tcod.lib.SDL_WINDOWEVENT_CLOSE: "WindowClose",
        tcod.lib.SDL_WINDOWEVENT_TAKE_FOCUS: "WindowTakeFocus",
        tcod.lib.SDL_WINDOWEVENT_HIT_TEST: "WindowHitTest",
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
    tcod.lib.SDL_QUIT: Quit,
    tcod.lib.SDL_KEYDOWN: KeyDown,
    tcod.lib.SDL_KEYUP: KeyUp,
    tcod.lib.SDL_MOUSEMOTION: MouseMotion,
    tcod.lib.SDL_MOUSEBUTTONDOWN: MouseButtonDown,
    tcod.lib.SDL_MOUSEBUTTONUP: MouseButtonUp,
    tcod.lib.SDL_MOUSEWHEEL: MouseWheel,
    tcod.lib.SDL_TEXTINPUT: TextInput,
    tcod.lib.SDL_WINDOWEVENT: WindowEvent,
}  # type: Dict[int, Any]


def get() -> Iterator[Any]:
    """Return an iterator for all pending events.

    Events are processed as the iterator is consumed.  Breaking out of, or
    discarding the iterator will leave the remaining events on the event queue.

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
    """
    sdl_event = tcod.ffi.new("SDL_Event*")
    while tcod.lib.SDL_PollEvent(sdl_event):
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
    """
    if timeout is not None:
        tcod.lib.SDL_WaitEventTimeout(tcod.ffi.NULL, int(timeout * 1000))
    else:
        tcod.lib.SDL_WaitEvent(tcod.ffi.NULL)
    return get()


class EventDispatch:
    """This class dispatches events to methods depending on the events type
    attribute.

    To use this class, make a sub-class and override the relevant `ev_*`
    methods.  Then send events to the dispatch method.

    Example::

        import tcod
        import tcod.event

        class State(tcod.event.EventDispatch):
            def ev_quit(self, event):
                raise SystemExit()

            def ev_keydown(self, event):
                print(event)

            def ev_mousebuttondown(self, event):
                print(event)

            def ev_mousemotion(self, event):
                print(event)

        root_console = tcod.console_init_root(80, 60)
        state = State()
        while True:
            for event in tcod.event.wait()
                state.dispatch(event)
    """

    def dispatch(self, event: Any) -> None:
        """Send an event to an `ev_*` method.

        `*` will be the events type converted to lower-case.

        If `event.type` is an empty string or None then it will be ignored.
        """
        if event.type:
            getattr(self, "ev_%s" % (event.type.lower(),))(event)

    def event_get(self) -> None:
        for event in get():
            self.dispatch(event)

    def event_wait(self, timeout: Optional[float]) -> None:
        wait(timeout)
        self.event_get()

    def ev_quit(self, event: Quit) -> None:
        """Called when the termination of the program is requested."""

    def ev_keydown(self, event: KeyDown) -> None:
        """Called when a keyboard key is pressed or repeated."""

    def ev_keyup(self, event: KeyUp) -> None:
        """Called when a keyboard key is released."""

    def ev_mousemotion(self, event: MouseMotion) -> None:
        """Called when the mouse is moved."""

    def ev_mousebuttondown(self, event: MouseButtonDown) -> None:
        """Called when a mouse button is pressed."""

    def ev_mousebuttonup(self, event: MouseButtonUp) -> None:
        """Called when a mouse button is released."""

    def ev_mousewheel(self, event: MouseWheel) -> None:
        """Called when the mouse wheel is scrolled."""

    def ev_textinput(self, event: TextInput) -> None:
        """Called to handle Unicode input."""

    def ev_windowshown(self, event: WindowEvent) -> None:
        """Called when the window is shown."""

    def ev_windowhidden(self, event: WindowEvent) -> None:
        """Called when the window is hidden."""

    def ev_windowexposed(self, event: WindowEvent) -> None:
        """Called when a window is exposed, and needs to be refreshed.

        This usually means a call to :any:`tcod.console_flush` is necessary.
        """

    def ev_windowmoved(self, event: WindowMoved) -> None:
        """Called when the window is moved."""

    def ev_windowresized(self, event: WindowResized) -> None:
        """Called when the window is resized."""

    def ev_windowsizechanged(self, event: WindowResized) -> None:
        """Called when the system or user changes the size of the window."""

    def ev_windowminimized(self, event: WindowEvent) -> None:
        """Called when the window is minimized."""

    def ev_windowmaximized(self, event: WindowEvent) -> None:
        """Called when the window is maximized."""

    def ev_windowrestored(self, event: WindowEvent) -> None:
        """Called when the window is restored."""

    def ev_windowenter(self, event: WindowEvent) -> None:
        """Called when the window gains mouse focus."""

    def ev_windowleave(self, event: WindowEvent) -> None:
        """Called when the window loses mouse focus."""

    def ev_windowfocusgained(self, event: WindowEvent) -> None:
        """Called when the window gains keyboard focus."""

    def ev_windowfocuslost(self, event: WindowEvent) -> None:
        """Called when the window loses keyboard focus."""

    def ev_windowclose(self, event: WindowEvent) -> None:
        """Called when the window manager requests the window to be closed."""

    def ev_windowtakefocus(self, event: WindowEvent) -> None:
        pass

    def ev_windowhittest(self, event: WindowEvent) -> None:
        pass


def get_mouse_state() -> MouseState:
    """Return the current state of the mouse.

    .. addedversion:: 9.3
    """
    xy = tcod.ffi.new("int[2]")
    buttons = tcod.lib.SDL_GetMouseState(xy, xy + 1)
    x, y = _pixel_to_tile(*xy)
    return MouseState((xy[0], xy[1]), (int(x), int(y)), buttons)


__all__ = [
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
    "EventDispatch",
] + tcod.event_constants.__all__
