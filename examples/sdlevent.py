#!/usr/bin/env python
"""
    An alternative, more direct implementation of event handling using cffi
    calls to SDL functions.  The current code is incomplete, but can be
    extended easily by following the official SDL documentation.

    This module be run directly like any other event get example, but is meant
    to be copied into your code base.  Then you can use the sdlevent.get and
    sdlevent.wait functions in your code.

    Printing any event will tell you its attributes in a human readable format.
    An events type attr is just the classes name with all letters upper-case.

    Like in tdl, events use a type attribute to tell events apart.  Unlike tdl
    and tcod the names and values used are directly derived from SDL.

    As a general guideline for turn-based rouge-likes, you should use
    KeyDown.sym for commands, and TextInput.text for name entry.

    This module may be included as a tcod.event module once it is more mature.

    An absolute minimal example:

        import tcod
        import sdlevent

        with tcod.console_init_root(80, 60, 'title') as console:
            while True:
                for event in sdlevent.wait():
                    print(event)
                    if event.type == 'QUIT':
                        raise SystemExit()
                tcod.console_flush()
"""
# CC0 License: https://creativecommons.org/publicdomain/zero/1.0/
# To the extent possible under law, Kyle Stewart has waived all copyright and
# related or neighboring rights to this sdlevent.py module.

import tcod

def _copy_attrs(prefix):
    """Copy names and values from tcod.lib into this modules top-level scope.

    This is a private function, used internally.

    Args:
        prefix (str): Used to filter out which names to copy.

    Returns:
        Dict[Any,str]: A reverse lookup table used in Event repr functions.
    """
    g = globals() # dynamically add names to the global state
    # removes the SDL prefix, this module already has SDL in the name
    if prefix.startswith('SDL_'):
        name_starts_at = 4
    elif prefix.startswith('SDL'):
        name_starts_at = 3
    else:
        name_starts_at = 0
    revere_table = {}
    for attr in dir(tcod.lib):
        if attr.startswith(prefix):
            name = attr[name_starts_at:]
            value = getattr(tcod.lib, attr)
            revere_table[value] = 'sdlevent.' + name
            g[name] = value

    return revere_table

def _describe_bitmask(bits, table, default='0'):
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
    return '|'.join(result)


_REVSRSE_SCANCODE_TABLE = _copy_attrs('SDL_SCANCODE')
_REVSRSE_SYM_TABLE = _copy_attrs('SDLK')
_REVSRSE_MOD_TABLE = _copy_attrs('KMOD')
_REVSRSE_WHEEL_TABLE = _copy_attrs('SDL_MOUSEWHEEL')

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

_REVSRSE_BUTTON_TABLE = {
    BUTTON_LEFT: 'sdlevent.BUTTON_LEFT',
    BUTTON_MIDDLE: 'sdlevent.BUTTON_MIDDLE',
    BUTTON_RIGHT: 'sdlevent.BUTTON_RIGHT',
    BUTTON_X1: 'sdlevent.BUTTON_X1',
    BUTTON_X2: 'sdlevent.BUTTON_X2',
}

_REVSRSE_BUTTON_MASK_TABLE = {
    BUTTON_LMASK: 'sdlevent.BUTTON_LMASK',
    BUTTON_MMASK: 'sdlevent.BUTTON_MMASK',
    BUTTON_RMASK: 'sdlevent.BUTTON_RMASK',
    BUTTON_X1MASK: 'sdlevent.BUTTON_X1MASK',
    BUTTON_X2MASK: 'sdlevent.BUTTON_X2MASK',
}

class Event(object):
    """The base event class."""
    @classmethod
    def from_sdl_event(cls, sdl_event):
        """Return a class instance from a cffi SDL_Event pointer."""
        raise NotImplementedError()

    @property
    def type(self):
        """All event types are just the class name, but all upper-case."""
        return self.__class__.__name__.upper()


class Quit(Event):
    """An application quit request event.

    For more info on when this event is triggered see:
    https://wiki.libsdl.org/SDL_EventType#SDL_QUIT
    """
    @classmethod
    def from_sdl_event(cls, sdl_event):
        return cls()

    def __repr__(self):
        return 'sdlevent.%s()' % self.__class__.__name__


class KeyboardEvent(Event):

    def __init__(self, scancode, sym, mod):
        self.scancode = scancode
        self.sym = sym
        self.mod = mod

    @classmethod
    def from_sdl_event(cls, sdl_event):
        keysym = sdl_event.key.keysym
        return cls(keysym.scancode, keysym.sym, keysym.mod)

    def __repr__(self):
        return ('sdlevent.%s(scancode=%s, sym=%s, mod=%s)' %
                (self.__class__.__name__,
                 _REVSRSE_SCANCODE_TABLE[self.scancode],
                 _REVSRSE_SYM_TABLE[self.sym],
                 _describe_bitmask(self.mod, _REVSRSE_MOD_TABLE),
                 )
                )


class KeyDown(KeyboardEvent):
    pass


class KeyUp(KeyboardEvent):
    pass


class MouseMotion(Event):

    def __init__(self, x, y, xrel, yrel, state):
        self.x = x
        self.y = y
        self.xrel = xrel
        self.yrel = yrel
        self.state = state

    @classmethod
    def from_sdl_event(cls, sdl_event):
        motion = sdl_event.motion
        return cls(motion.x, motion.y, motion.xrel, motion.yrel, motion.state)

    def __repr__(self):
        return ('sdlevent.%s(x=%i, y=%i, xrel=%i, yrel=%i, state=%s)' %
                (self.__class__.__name__,
                 self.x, self.y,
                 self.xrel, self.yrel,
                 _describe_bitmask(self.state, _REVSRSE_BUTTON_MASK_TABLE),
                 )
                )


class MouseButtonEvent(Event):

    def __init__(self, x, y, button):
        self.x = x
        self.y = y
        self.button = button

    @classmethod
    def from_sdl_event(cls, sdl_event):
        button = sdl_event.button
        return cls(button.x, button.y, button.button)

    def __repr__(self):
        return ('sdlevent.%s(x=%i, y=%i, button=%s)' %
                (self.__class__.__name__,
                 self.x, self.y, _REVSRSE_BUTTON_TABLE[self.button],
                 )
                )


class MouseButtonDown(MouseButtonEvent):
    pass


class MouseButtonUp(MouseButtonEvent):
    pass


class MouseWheel(Event):

    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    @classmethod
    def from_sdl_event(cls, sdl_event):
        wheel = sdl_event.wheel
        return cls(wheel.x, wheel.y, wheel.direction)

    def __repr__(self):
        return ('sdlevent.%s(x=%i, y=%i, direction=%s)' %
                (self.__class__.__name__,
                 self.x, self.y,
                 _REVSRSE_WHEEL_TABLE[self.direction],
                 )
                )


class TextInput(Event):

    def __init__(self, text):
        self.text = text

    @classmethod
    def from_sdl_event(cls, sdl_event):
        return cls(tcod.ffi.string(sdl_event.text.text, 32).decode('utf8'))

    def __repr__(self):
        return ('sdlevent.%s(text=%r)' % (self.__class__.__name__, self.text))


_SDL_TO_CLASS_TABLE = {
    tcod.lib.SDL_QUIT: Quit,
    tcod.lib.SDL_KEYDOWN: KeyDown,
    tcod.lib.SDL_KEYUP: KeyUp,
    tcod.lib.SDL_MOUSEMOTION: MouseMotion,
    tcod.lib.SDL_MOUSEBUTTONDOWN: MouseButtonDown,
    tcod.lib.SDL_MOUSEBUTTONUP: MouseButtonUp,
    tcod.lib.SDL_MOUSEWHEEL: MouseWheel,
    tcod.lib.SDL_TEXTINPUT: TextInput,
}

def get():
    """Iterate over all pending events.

    Returns:
        Iterator[sdlevent.Event]:
            An iterator of Event subclasses.
    """
    sdl_event = tcod.ffi.new('SDL_Event*')
    while tcod.lib.SDL_PollEvent(sdl_event):
        if sdl_event.type in _SDL_TO_CLASS_TABLE:
            yield _SDL_TO_CLASS_TABLE[sdl_event.type].from_sdl_event(sdl_event)

def wait(timeout=None):
    """Block until an event exists, then iterate over all events.

    Keep in mind that this function will wake even for events not handled by
    this module.

    Args:
        timeout (Optional[int]):
            Maximum number of milliseconds to wait, or None to wait forever.

    Returns:
        Iterator[sdlevent.Event]: Same iterator as a call to sdlevent.get
    """
    if timeout is not None:
        tcod.lib.SDL_WaitEventTimeout(tcod.ffi.NULL, timeout)
    else:
        tcod.lib.SDL_WaitEvent(tcod.ffi.NULL)
    return get()

def _main():
    """An example program for when this module is run directly."""
    WIDTH, HEIGHT = 120, 60
    TITLE = 'sdlevent.py engine'

    with tcod.console_init_root(WIDTH, HEIGHT, TITLE) as console:
        tcod.sys_set_fps(24)
        while True:
            for event in wait():
                print(event)
                if event.type == 'QUIT':
                    raise SystemExit()
                elif event.type == 'MOUSEMOTION':
                    console.rect(0, HEIGHT - 1, WIDTH, 1, True)
                    console.print_(0, HEIGHT - 1, repr(event))
                else:
                    console.blit(console, 0, 0, 0, 1, WIDTH, HEIGHT - 2)
                    console.print_(0, HEIGHT - 3, repr(event))
            tcod.console_flush()

if __name__ == '__main__':
    _main()
