"""
    This module handles user input.

    Here's a quick reference to Event types and their attributes:
    QUIT
    KEYDOWN: keyname key char alt ctrl shift lalt lctrl ralt rctrl
    KEYUP: keyname key char alt ctrl shift lalt lctrl ralt rctrl
    MOUSEDOWN: button pos cell
    MOUSEUP: button pos cell
    MOUSEMOTION: pos cell motion cellmotion
    
    You will likely want to use the tdl.event.get function but you can still
    use keyWait and isWindowClosed to control your entire program.
"""

import ctypes

from .tcod import _lib, _Mouse, _Key
from . import tcod as _tcod

_mousel = 0
_mousem = 0
_mouser = 0

# this interpets the constants from libtcod and makes a key -> keyname dictionary
def _parse_keynames(module):
    """returns a dictionary mapping of human readable key names to their keycodes
    this parses constants with the names of K_* and makes code=name pairs
    this is for KeyEvent.keyname variable and that enables things like:
    if (event.keyname == 'PAGEUP'):
    """
    _keynames = {}
    for attr in dir(module): # from the modules variables
        if attr[:2] == 'K_': # get the K_* constants
            _keynames[getattr(_tcod, attr)] = attr[2:] # and make CODE=NAME pairs
    return _keynames

_keynames = _parse_keynames(_tcod)
    
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
    type = 'QUIT'

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
    type = 'KEYDOWN'

class KeyUp(KeyEvent):
    __slots__ = ()
    type = 'KEYUP'

class MouseButtonEvent(Event):
    __slots__ = ('button', 'pos', 'cell')

    def __init__(self, button, pos, cell):
        self.button = button
        self.pos = pos
        self.cell = cell

class MouseDown(MouseButtonEvent):
    __slots__ = ()
    type = 'MOUSEDOWN'

class MouseUp(MouseButtonEvent):
    __slots__ = ()
    type = 'MOUSEUP'

class MouseMotion(Event):
    __slots__ = ('pos',  'motion', 'cell', 'cellmotion')
    type = 'MOUSEMOTION'

    def __init__(self, pos, cell, motion, cellmotion):
        self.pos = pos
        self.cell = cell
        self.motion = motion
        self.cellmotion = cellmotion

def get():
    """Flushes the event queue and returns the list of events.
    
    This function returns Event objects that can be ID'd and sorted with their type attribute:
    for event in tdl.event.get():
        if event.type == 'QUIT':
            raise SystemExit()
        elif event.type == 'MOUSEDOWN':
            print('Mouse button %i clicked at %i, %i' % (event.button, event.pos[0], event.pos[1]))
        elif event.type == 'KEYDOWN':
            print('Key #%i "%s" pressed' % (event.key, event.char))
    
    Here is a list of events and their attributes:
    QUIT
    KEYDOWN: key char keyname alt ctrl shift lalt lctrl ralt rctrl
    KEYUP: key char keyname alt ctrl shift lalt lctrl ralt rctrl
    MOUSEDOWN: button pos cell
    MOUSEUP: button pos cell
    MOUSEMOTION: pos motion cell cellmotion
    """
    global _mousel, _mousem, _mouser, _eventsflushed
    _eventsflushed = True
    events = []
    
    mouse = _Mouse()
    libkey = _Key()
    while 1:
        if not _lib.TCOD_sys_check_for_event(_tcod.TCOD_EVENT_ANY, libkey, mouse):
            break
            
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
        
        if mouse.wheel_up:
            events.append(MouseDown(4, *mousepos))
        if mouse.wheel_down:
            events.append(MouseDown(5, *mousepos))
            
        _mousel = mouse.lbutton
        _mousem = mouse.mbutton
        _mouser = mouse.rbutton

        if libkey.vk == _tcod.K_NONE:
            break
        if libkey.pressed:
            keyevent = KeyDown
        else:
            keyevent = KeyUp
        events.append(keyevent(*tuple(libkey)))
    
    if _lib.TCOD_console_is_window_closed():
        events.append(Quit())

    return events

def keyWait():
    """Waits until the user presses a key.  Then returns a KeyDown event.
    """
    global _eventsflushed
    _eventsflushed = True
    flush = False
    libkey = _Key()
    _lib.TCOD_console_wait_for_keypress_wrapper(libkey, flush)
    return KeyDown(*libkey)

# tested this function recently, it did not work
#def keyPressed(key):
#    """Returns True when key is currently pressed.
#
#    key can be a number or single length string
#    """
#    assert isinstance(key, (str, int)), "key must be a single character string or int"
#    if not isinstance(key, int):
#        assert len(key) == 1, "key can not be a multi character string"
#        key = ord(key)
#    return _lib.TCOD_console_check_for_keypress(key).pressed # returns key object?

def isWindowClosed():
    """Returns True if the exit button on the window has been clicked and
    stays True afterwards.
    """
    return _lib.TCOD_console_is_window_closed()

__all__ = [var for var in locals().keys() if not '_' in var[0]]
