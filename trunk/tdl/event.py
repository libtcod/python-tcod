"""
    This module handles user input.

    Here's a quick reference to Event types and their attributes:
    QUIT
    KEYDOWN: key char alt ctrl shift lalt lctrl ralt rctrl
    KEYUP: key char alt ctrl shift lalt lctrl ralt rctrl
    MOUSEDOWN: button pos cell
    MOUSEUP: button pos cell
    MOUSEMOTION: pos cell motion cellmotion

    You will likely want to use the tdl.event.get function but you can still use the keyWait and isWindowClosed functions if you want.
"""

import ctypes

from .tcod import _lib, _Mouse
from . import local

# make sure that the program does not lock up from missing event flushes
_eventsflushed = False # not that it works to well to fix that problem

_autoflush = True

_mousel = 0
_mousem = 0
_mouser = 0

# this interpets the constants from local and makes a key to keyname dictionary
_keynames = {}
for attr in dir(local):
    if attr[:2] == 'K_':
        _keynames[getattr(local, attr)] = attr[2:]
del attr

class Event(object):
    __slots__ = ()
    type = None

    def __repr__(self):
        attrdict = {}
        for varname in dir(self):
            if '_' in varname:
                continue
            attrdict[varname] = self.__getattribute__(varname)
        return '%s Event %s' % (self.__class__.__name__, repr(attrdict))
    
    #def __tuple__(self):
    #    return tuple((getattr(self, attr) for attr in self.__slots__))

class Quit(Event):
    __slots__ = ()
    type = local.QUIT

class KeyEvent(Event):
    __slots__ = ('key', 'keyname', 'char', 'lalt', 'lctrl', 'ralt', 'rctrl',
                 'shift', 'alt', 'ctrl')

    def __init__(self, key, char, lalt, lctrl, ralt, rctrl, shift):
        self.key = key
        self.keyname = _keynames[key]
        char = char if isinstance(char, str) else char.decode()
        self.char = char.replace('\x00', '') # change null to empty string
        self.lalt = bool(lalt)
        self.ralt = bool(ralt)
        self.lctrl = bool(lctrl)
        self.rctrl = bool(rctrl)
        self.shift = bool(shift)
        self.alt = bool(lalt or ralt)
        self.ctrl = bool(lctrl or rctrl)

class KeyDown(KeyEvent):
    __slots__ = ()
    type = local.KEYDOWN

class KeyUp(KeyEvent):
    __slots__ = ()
    type = local.KEYUP

class MouseButtonEvent(Event):
    __slots__ = ('button', 'pos', 'cell')
    type = local.MOUSEDOWN

    def __init__(self, button, pos, cell):
        self.button = button
        self.pos = pos
        self.cell = cell

class MouseDown(MouseButtonEvent):
    __slots__ = ()
    type = local.MOUSEDOWN

class MouseUp(MouseButtonEvent):
    __slots__ = ()
    type = local.MOUSEUP

class MouseMotion(Event):
    __slots__ = ('pos',  'motion', 'cell', 'cellmotion', 'relpos', 'relcell')
    type = local.MOUSEMOTION

    def __init__(self, pos, cell, relpos, relcell):
        self.pos = pos
        self.cell = cell
        self.relpos = self.motion = relpos
        self.relcell = self.cellmotion = relcell

def get():
    """Flushes the event queue and returns the list of events.
    
    This function returns Event objects that can be ID'd and sorted with their type attribute:
    for event in tdl.event.get():
        if event.type == tdl.QUIT:
            raise SystemExit()
        elif event.type == tdl.MOUSEDOWN:
            print('Mouse button %i clicked at %i, %i' % (event.button, event.pos[0], event.pos[1]))
        elif event.type == tdl.KEYDOWN:
            print('Key #%i "%s" pressed' % (event.key, event.char))
    
    Here is a list of events and their attributes:
    QUIT
    KEYDOWN: key char alt ctrl shift lalt lctrl ralt rctrl
    KEYUP: key char alt ctrl shift lalt lctrl ralt rctrl
    MOUSEDOWN: button pos cell
    MOUSEUP: button pos cell
    MOUSEMOTION: pos motion cell cellmotion
    """
    global _mousel, _mousem, _mouser, _eventsflushed
    _eventsflushed = True
    events = []
    while 1:
        libkey = _lib.TCOD_console_check_for_keypress(3)
        if libkey.vk == local.K_NONE:
            break
        if libkey.pressed:
            keyevent = KeyDown
        else:
            keyevent = KeyUp
        events.append(keyevent(*tuple(libkey)))

    mouse = _Mouse()
    _lib.TCOD_mouse_get_status(mouse)
    if mouse.dx or mouse.dy:
        events.append(MouseMotion(*mouse.motion))

    mousepos = ((mouse.x, mouse.y), (mouse.cx, mouse.cy))

    for oldstate, newstate, released, button in zip((_mousel, _mousem, _mouser),
                                mouse.button, mouse.button_pressed, (1, 2, 3)):
        if released:
            if not oldstate:
                events.append(MouseDown(button, *mousepos))
            events.append(MouseUp(button, *mousepos))
            if newstate:
                events.append(MouseDown(button, *mousepos))
        elif newstate and not oldstate:
            events.append(MouseDown(button, *mousepos))

    _mousel = mouse.lbutton
    _mousem = mouse.mbutton
    _mouser = mouse.rbutton

    if _lib.TCOD_console_is_window_closed():
        events.append(Quit())

    return events

def keyWait():
    """Waits until the user presses a key.  Returns KeyDown events.
    """
    global _eventsflushed
    _eventsflushed = True
    flush = False
    libkey = _lib.TCOD_console_wait_for_keypress(flush)
    return KeyDown(*libkey)

def keyPressed(key):
    """Returns True when key is currently pressed.

    key can be a number or single length string
    """
    assert isinstance(key, (str, int)), "key must be a string or int"
    if not isinstance(key, int):
        assert len(key) == 1, "key can not be a multi character string"
        key = ord(key)
    return _lib.TCOD_console_check_for_keypress(key)

def isWindowClosed():
    """Returns True if the exit button on the window has been clicked and
    stays True afterwards.
    """
    return _lib.TCOD_console_is_window_closed()

__all__ = [var for var in locals().keys() if not '_' in var[0]]
