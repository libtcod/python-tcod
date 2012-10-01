"""
    This module handles user input.
    
    String constants can be found in the variable details of L{KeyEvent.key},
    L{MouseButtonEvent.button}, and L{Event.type}.
    
    You will likely want to use the L{event.get} function or L{event.App}
    class but you can still use L{event.keyWait} and L{event.isWindowClosed}
    to control your entire program.
"""

import time
import ctypes

from .__tcod import _lib, _Mouse, _Key
from . import __tcod as _tcod
import tdl as _tdl

_eventQueue = []

_mousel = 0
_mousem = 0
_mouser = 0

# this interpets the constants from libtcod and makes a key -> keyname dictionary
def _parseKeyNames(module):
    """
    returns a dictionary mapping of human readable key names to their keycodes
    this parses constants with the names of K_* and makes code=name pairs
    this is for KeyEvent.keyname variable and that enables things like:
    if (event.keyname == 'PAGEUP'):
    """
    _keyNames = {}
    for attr in dir(module): # from the modules variables
        if attr[:2] == 'K_': # get the K_* constants
            _keyNames[getattr(_tcod, attr)] = attr[2:] # and make CODE=NAME pairs
    return _keyNames

_keyNames = _parseKeyNames(_tcod)
    
class Event(object):
    __slots__ = ()
    type = None
    """String constant representing the type of event.
    
    Can be: 'QUIT', 'KEYDOWN', 'KEYUP', 'MOUSEDOWN', 'MOUSEUP', or 'MOUSEMOTION.'
    """

    def __repr__(self):
        """List an events public attributes in print calls.
        """
        attrdict = {}
        for varname in dir(self):
            if '_' == varname[0]:
                continue
            attrdict[varname] = self.__getattribute__(varname)
        return '%s Event %s' % (self.__class__.__name__, repr(attrdict))

class Quit(Event):
    """Fired when the window is closed by the user.
    """
    __slots__ = ()
    type = 'QUIT'

class KeyEvent(Event):
    __slots__ = ('key', 'keyname', 'char', 'shift', 'alt', 'control',
                 'leftAlt', 'leftCtrl', 'rightAlt', 'rightCtrl')

    def __init__(self, key, char, lalt, lctrl, ralt, rctrl, shift):
        self.key = _keyNames[key]
        """Human readable name of the key pressed.
        
        Can be one of
        'NONE', 'ESCAPE', 'BACKSPACE', 'TAB', 'ENTER', 'SHIFT', 'CONTROL',
        'ALT', 'PAUSE', 'CAPSLOCK', 'PAGEUP', 'PAGEDOWN', 'END', 'HOME', 'UP',
        'LEFT', 'RIGHT', 'DOWN', 'PRINTSCREEN', 'INSERT', 'DELETE', 'LWIN',
        'RWIN', 'APPS', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'KP0', 'KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6', 'KP7', 'KP8', 'KP9',
        'KPADD', 'KPSUB', 'KPDIV', 'KPMUL', 'KPDEC', 'KPENTER', 'F1', 'F2',
        'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
        'NUMLOCK', 'SCROLLLOCK', 'SPACE', 'CHAR'
        @type: string"""
        char = char if isinstance(char, str) else char.decode()
        self.char = char.replace('\x00', '') # change null to empty string
        """A single character string of the letter or symbol pressed.
        Special characters like delete and return are not cross platform.
        @type: string"""
        self.leftAlt = bool(lalt)
        self.rightAlt = bool(ralt)
        self.leftCtrl = bool(lctrl)
        self.rightCtrl = bool(rctrl)
        self.shift = bool(shift)
        """True if shift was held down during this event.
        @type: boolean"""
        self.alt = bool(lalt or ralt)
        """True if alt was held down during this event.
        @type: boolean"""
        self.control = bool(lctrl or rctrl)
        """True if control was held down during this event.
        @type: boolean"""

class KeyDown(KeyEvent):
    """Fired when the user presses a key on the keyboard or a key repeats.
    """
    __slots__ = ()
    type = 'KEYDOWN'

class KeyUp(KeyEvent):
    """Fired when the user releases a key on the keyboard.
    """
    __slots__ = ()
    type = 'KEYUP'

_mouseNames = {1: 'LEFT', 2: 'MIDDLE', 3: 'RIGHT', 4: 'SCROLLUP', 5: 'SCROLLDOWN'}
class MouseButtonEvent(Event):
    __slots__ = ('button', 'pos', 'cell')

    def __init__(self, button, pos, cell):
        self.button = _mouseNames[button]
        """Can be one of
        'LEFT', 'MIDDLE', 'RIGHT', 'SCROLLUP', 'SCROLLDOWN'
        @type: string"""
        self.pos = pos
        """(x, y) position of the mouse on the screen
        @type: (int, int)"""
        self.cell = cell
        """(x, y) position of the mouse snapped to a cell on the root console
        @type: (int, int)"""

class MouseDown(MouseButtonEvent):
    """Fired when a mouse button is pressed."""
    __slots__ = ()
    type = 'MOUSEDOWN'

class MouseUp(MouseButtonEvent):
    """Fired when a mouse button is released."""
    __slots__ = ()
    type = 'MOUSEUP'

class MouseMotion(Event):
    """Fired when the mouse is moved."""
    __slots__ = ('pos',  'motion', 'cell', 'cellmotion')
    type = 'MOUSEMOTION'

    def __init__(self, pos, cell, motion, cellmotion):
        self.pos = pos
        "(x, y) position of the mouse on the screen"
        self.cell = cell
        "(x, y) position of the mouse snapped to a cell on the root console"
        self.motion = motion
        "(x, y) motion of the mouse on the screen"
        self.cellmotion = cellmotion
        "(x, y) mostion of the mouse moving over cells on the root console"

class App(object):
    """
    
     - ev_*: Events are passed to methods based on their L{Event.type} attribute.  If an event
       type is 'KEYDOWN' for example the event will be sent to the ev_KEYDOWN method
       with the event as a parameter.
    
     - key_*: When a key is pressed another method will be called based on the
       keyname attribute.  For example the 'ENTER' keyname will call key_ENTER
       with the associated L{KeyDown} event as its parameter.
    
    """
    __slots__ = ('__running')
    
    def ev_QUIT(self, event):
        """Unless overridden this method raises a SystemExit exception closing
        the program."""
        raise SystemExit()
    
    def ev_KEYDOWN(self, event):
        "Override this method to handle this event."
    
    def ev_KEYUP(self, event):
        "Override this method to handle this event."
    
    def ev_MOUSEDOWN(self, event):
        "Override this method to handle this event."
    
    def ev_MOUSEUP(self, event):
        "Override this method to handle this event."
    
    def ev_MOUSEMOTION(self, event):
        "Override this method to handle this event."
    
    def update(self, deltaTime):
        """Override this method to handle per frame logic and drawing.
        
        @type deltaTime: float
        @param deltaTime: This parameter tells the amount of time passed since
                          the last call measured in seconds as a floating point
                          number.
        """
        pass
        
    def suspend(self):
        """When called the App will begin to return control to where
        L{App.run} was called.
        
        No further events are processed and the L{App.update} method will be
        called one last time before exiting
        (unless suspended during a call to L{App.update}.)
        """
        self._running = False
        
    def run(self):
        """Delegate control over to this App instance.  This function will
        process all events and send them to the special methods ev_* and key_*.
        
        A call to L{App.suspend} will return the control flow back to where
        this function is called.  And then the App can be run again.
        But a single App instance can not be run multiple times simultaneously.
        """
        if self.__running:
            raise _tdl.TDLError('An App can not be run multiple times simultaneously')
        self.__running = True
        prevTime = time.clock()
        while self._running:
            for event in get():
                if event.type: # exclude custom events with a blank type variable
                    # call the ev_* methods
                    method = 'ev_%s' % event.type # ev_TYPE
                    getattr(self, method)(event)
                if event.type == 'KEYDOWN':
                    # call the key_* methods
                    method = 'key_%s' % event.keyname # key_KEYNAME
                    if hasattr(self, method): # silently exclude undefined methods
                        getattr(self, method)(event)
                if not __running:
                    break # interupt event handing after suspend()
            newTime = time.clock()
            self.update(newTime - prevTime)
            prevTime = newTime
            _tdl.flush()

def _processEvents():
    """Flushes the event queue from libtcod into the global list _eventQueue"""
    global _mousel, _mousem, _mouser, _eventsflushed
    _eventsflushed = True
    events = []
    
    mouse = _Mouse()
    libkey = _Key()
    while 1:
        libevent = _lib.TCOD_sys_check_for_event(_tcod.TCOD_EVENT_ANY, libkey, mouse)
        if not libevent: # no more events from libtcod
            break
            
        #if mouse.dx or mouse.dy:
        if libevent & _tcod.TCOD_EVENT_MOUSE_MOVE:
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

    _eventQueue.extend(events)
    
def get():
    """Flushes the event queue and returns the list of events.
    
    This function returns L{Event} objects that can be indentified by their
    type attribute or their class.
    
    @rtype: iterator
    """
    _processEvents()
    while _eventQueue:
        yield(_eventQueue.pop(0))
    raise StopIteration()

def keyWait():
    """Waits until the user presses a key.  Then returns a L{KeyDown} event.
    
    @rtype: L{KeyDown}
    """
    global _eventsflushed
    _eventsflushed = True
    flush = False
    libkey = _Key()
    _lib.TCOD_console_wait_for_keypress_wrapper(libkey, flush)
    return KeyDown(*libkey)

def isWindowClosed():
    """Returns True if the exit button on the window has been clicked and
    stays True afterwards.
    
    @rtype: boolean
    """
    return _lib.TCOD_console_is_window_closed()

__all__ = [_var for _var in locals().keys() if _var[0] != '_' and _var not in ['time', 'ctypes']]
