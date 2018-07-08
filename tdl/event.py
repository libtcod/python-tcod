"""
    This module handles user input.

    To handle user input you will likely want to use the :any:`event.get`
    function or create a subclass of :any:`event.App`.

    - :any:`tdl.event.get` iterates over recent events.
    - :any:`tdl.event.App` passes events to the overridable methods: ``ev_*``
      and ``key_*``.

    But there are other options such as :any:`event.key_wait` and
    :any:`event.is_window_closed`.

    A few event attributes are actually string constants.
    Here's a reference for those:

    - :any:`Event.type`:
      'QUIT', 'KEYDOWN', 'KEYUP', 'MOUSEDOWN', 'MOUSEUP', or 'MOUSEMOTION.'
    - :any:`MouseButtonEvent.button` (found in :any:`MouseDown` and
      :any:`MouseUp` events):
      'LEFT', 'MIDDLE', 'RIGHT', 'SCROLLUP', 'SCROLLDOWN'
    - :any:`KeyEvent.key` (found in :any:`KeyDown` and :any:`KeyUp` events):
      'NONE', 'ESCAPE', 'BACKSPACE', 'TAB', 'ENTER', 'SHIFT', 'CONTROL',
      'ALT', 'PAUSE', 'CAPSLOCK', 'PAGEUP', 'PAGEDOWN', 'END', 'HOME', 'UP',
      'LEFT', 'RIGHT', 'DOWN', 'PRINTSCREEN', 'INSERT', 'DELETE', 'LWIN',
      'RWIN', 'APPS', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
      'KP0', 'KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6', 'KP7', 'KP8', 'KP9',
      'KPADD', 'KPSUB', 'KPDIV', 'KPMUL', 'KPDEC', 'KPENTER', 'F1', 'F2',
      'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
      'NUMLOCK', 'SCROLLLOCK', 'SPACE', 'CHAR'
"""

import time as _time

from tcod import ffi as _ffi
from tcod import lib as _lib

import tdl as _tdl
from . import style as _style

_eventQueue = []
_pushedEvents = []
_eventsflushed = False

_mousel = 0
_mousem = 0
_mouser = 0

# this interprets the constants from libtcod and makes a key -> keyname dictionary
def _parseKeyNames(lib):
    """
    returns a dictionary mapping of human readable key names to their keycodes
    this parses constants with the names of K_* and makes code=name pairs
    this is for KeyEvent.key variable and that enables things like:
    if (event.key == 'PAGEUP'):
    """
    _keyNames = {}
    for attr in dir(lib): # from the modules variables
        if attr[:6] == 'TCODK_': # get the K_* constants
            _keyNames[getattr(lib, attr)] = attr[6:] # and make CODE=NAME pairs
    return _keyNames

_keyNames = _parseKeyNames(_lib)

class Event(object):
    """Base Event class.

    You can easily subclass this to make your own events.  Be sure to set
    the class attribute L{Event.type} for it to be passed to a custom
    :any:`App` ev_* method.
    """
    type = None
    """String constant representing the type of event.

    The :any:`App` ev_* methods depend on this attribute.

    Can be: 'QUIT', 'KEYDOWN', 'KEYUP', 'MOUSEDOWN', 'MOUSEUP',
    or 'MOUSEMOTION.'
    """

    def __repr__(self):
        """List an events public attributes when printed.
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
    """Base class for key events."""

    def __init__(self, key='', char='', text='', shift=False,
                 left_alt=False, right_alt=False,
                 left_control=False, right_control=False,
                 left_meta=False, right_meta=False):
        # Convert keycodes into string, but use string if passed
        self.key = key if isinstance(key, str) else _keyNames[key]
        """Text: Human readable names of the key pressed.
        Non special characters will show up as 'CHAR'.

        Can be one of
        'NONE', 'ESCAPE', 'BACKSPACE', 'TAB', 'ENTER', 'SHIFT', 'CONTROL',
        'ALT', 'PAUSE', 'CAPSLOCK', 'PAGEUP', 'PAGEDOWN', 'END', 'HOME', 'UP',
        'LEFT', 'RIGHT', 'DOWN', 'PRINTSCREEN', 'INSERT', 'DELETE', 'LWIN',
        'RWIN', 'APPS', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'KP0', 'KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6', 'KP7', 'KP8', 'KP9',
        'KPADD', 'KPSUB', 'KPDIV', 'KPMUL', 'KPDEC', 'KPENTER', 'F1', 'F2',
        'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
        'NUMLOCK', 'SCROLLLOCK', 'SPACE', 'CHAR'

        For the actual character instead of 'CHAR' use :any:`keychar`.
        """
        self.char = char.replace('\x00', '') # change null to empty string
        """Text: A single character string of the letter or symbol pressed.

        Special characters like delete and return are not cross-platform.
        L{key} or L{keychar} should be used instead for special keys.
        Characters are also case sensitive.
        """
        # get the best out of self.key and self.char
        self.keychar = self.char if self.key == 'CHAR' else self.key
        """Similar to L{key} but returns a case sensitive letter or symbol
        instead of 'CHAR'.

        This variable makes available the widest variety of symbols and should
        be used for key-mappings or anywhere where a narrower sample of keys
        isn't needed.
        """
        self.text = text

        self.left_alt = self.leftAlt = bool(left_alt)
        """bool:"""
        self.right_alt = self.rightAlt = bool(right_alt)
        """bool:"""
        self.left_control = self.leftCtrl = bool(left_control)
        """bool:"""
        self.right_control = self.rightCtrl = bool(right_control)
        """bool:"""
        self.shift = bool(shift)
        """bool: True if shift was held down during this event."""
        self.alt = self.left_alt or self.right_alt
        """bool: True if alt was held down during this event."""
        self.control = self.left_control or self.right_control
        """bool: True if control was held down during this event."""
        self.left_meta = bool(left_meta)
        self.right_meta = bool(right_meta)
        self.meta = self.left_meta or self.right_meta

    def __repr__(self):
        parameters = []
        for attr in ('key', 'char', 'text', 'shift',
                     'left_alt', 'right_alt',
                     'left_control', 'right_control',
                     'left_meta', 'right_meta'):
            value = getattr(self, attr)
            if value:
                parameters.append('%s=%r' % (attr, value))
        return '%s(%s)' % (self.__class__.__name__, ', '.join(parameters))

class KeyDown(KeyEvent):
    """Fired when the user presses a key on the keyboard or a key repeats.
    """
    type = 'KEYDOWN'

class KeyUp(KeyEvent):
    """Fired when the user releases a key on the keyboard.
    """
    type = 'KEYUP'

_mouseNames = {1: 'LEFT', 2: 'MIDDLE', 3: 'RIGHT', 4: 'SCROLLUP', 5: 'SCROLLDOWN'}
class MouseButtonEvent(Event):
    """Base class for mouse button events."""

    def __init__(self, button, pos, cell):
        self.button = _mouseNames[button]
        """Text: Can be one of
        'LEFT', 'MIDDLE', 'RIGHT', 'SCROLLUP', 'SCROLLDOWN'
        """
        self.pos = pos
        """Tuple[int, int]: (x, y) position of the mouse on the screen."""
        self.cell = cell
        """Tuple[int, int]: (x, y) position of the mouse snapped to a cell on
        the root console
        """

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
    type = 'MOUSEMOTION'

    def __init__(self, pos, cell, motion, cellmotion):
        self.pos = pos
        """(x, y) position of the mouse on the screen.
        type: (int, int)"""
        self.cell = cell
        """(x, y) position of the mouse snapped to a cell on the root console.
        type: (int, int)"""
        self.motion = motion
        """(x, y) motion of the mouse on the screen.
        type: (int, int)"""
        self.cellmotion = cellmotion
        """(x, y) mostion of the mouse moving over cells on the root console.
        type: (int, int)"""

class App(object):
    """
    Application framework.

     - ev_*: Events are passed to methods based on their :any:`Event.type`
       attribute.
       If an event type is 'KEYDOWN' the ev_KEYDOWN method will be called
       with the event instance as a parameter.

     - key_*: When a key is pressed another method will be called based on the
       :any:`KeyEvent.key` attribute.  For example the 'ENTER' key will call
       key_ENTER with the associated :any:`KeyDown` event as its parameter.

     - :any:`update`: This method is called every loop.  It is passed a single
       parameter detailing the time in seconds since the last update
       (often known as deltaTime.)

       You may want to call drawing routines in this method followed by
       :any:`tdl.flush`.

    """
    __slots__ = ('__running', '__prevTime')

    def ev_QUIT(self, event):
        """Unless overridden this method raises a SystemExit exception closing
        the program."""
        raise SystemExit()

    def ev_KEYDOWN(self, event):
        """Override this method to handle a :any:`KeyDown` event."""

    def ev_KEYUP(self, event):
        """Override this method to handle a :any:`KeyUp` event."""

    def ev_MOUSEDOWN(self, event):
        """Override this method to handle a :any:`MouseDown` event."""

    def ev_MOUSEUP(self, event):
        """Override this method to handle a :any:`MouseUp` event."""

    def ev_MOUSEMOTION(self, event):
        """Override this method to handle a :any:`MouseMotion` event."""

    def update(self, deltaTime):
        """Override this method to handle per frame logic and drawing.

        Args:
            deltaTime (float):
                This parameter tells the amount of time passed since
                the last call measured in seconds as a floating point
                number.

                You can use this variable to make your program
                frame rate independent.
                Use this parameter to adjust the speed of motion,
                timers, and other game logic.
        """
        pass

    def suspend(self):
        """When called the App will begin to return control to where
        :any:`App.run` was called.

        Some further events are processed and the :any:`App.update` method
        will be called one last time before exiting
        (unless suspended during a call to :any:`App.update`.)
        """
        self.__running = False

    def run(self):
        """Delegate control over to this App instance.  This function will
        process all events and send them to the special methods ev_* and key_*.

        A call to :any:`App.suspend` will return the control flow back to where
        this function is called.  And then the App can be run again.
        But a single App instance can not be run multiple times simultaneously.
        """
        if getattr(self, '_App__running', False):
            raise _tdl.TDLError('An App can not be run multiple times simultaneously')
        self.__running = True
        while self.__running:
            self.runOnce()

    def run_once(self):
        """Pump events to this App instance and then return.

        This works in the way described in :any:`App.run` except it immediately
        returns after the first :any:`update` call.

        Having multiple :any:`App` instances and selectively calling runOnce on
        them is a decent way to create a state machine.
        """
        if not hasattr(self, '_App__prevTime'):
            self.__prevTime = _time.clock() # initiate __prevTime
        for event in get():
            if event.type: # exclude custom events with a blank type variable
                # call the ev_* methods
                method = 'ev_%s' % event.type # ev_TYPE
                getattr(self, method)(event)
            if event.type == 'KEYDOWN':
                # call the key_* methods
                method = 'key_%s' % event.key # key_KEYNAME
                if hasattr(self, method): # silently exclude undefined methods
                    getattr(self, method)(event)
        newTime = _time.clock()
        self.update(newTime - self.__prevTime)
        self.__prevTime = newTime
        #_tdl.flush()

def _processEvents():
    """Flushes the event queue from libtcod into the global list _eventQueue"""
    global _mousel, _mousem, _mouser, _eventsflushed, _pushedEvents
    _eventsflushed = True
    events = _pushedEvents # get events from event.push
    _pushedEvents = [] # then clear the pushed events queue

    mouse = _ffi.new('TCOD_mouse_t *')
    libkey = _ffi.new('TCOD_key_t *')
    while 1:
        libevent = _lib.TCOD_sys_check_for_event(_lib.TCOD_EVENT_ANY, libkey, mouse)
        if not libevent: # no more events from libtcod
            break

        #if mouse.dx or mouse.dy:
        if libevent & _lib.TCOD_EVENT_MOUSE_MOVE:
            events.append(MouseMotion((mouse.x, mouse.y),
                                      (mouse.cx, mouse.cy),
                                      (mouse.dx, mouse.dy),
                                      (mouse.dcx, mouse.dcy)))

        mousepos = ((mouse.x, mouse.y), (mouse.cx, mouse.cy))

        for oldstate, newstate, released, button in \
            zip((_mousel, _mousem, _mouser),
                (mouse.lbutton, mouse.mbutton, mouse.rbutton),
                (mouse.lbutton_pressed, mouse.mbutton_pressed,
                 mouse.rbutton_pressed),
                (1, 2, 3)):
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

        if libkey.vk == _lib.TCODK_NONE:
            break
        if libkey.pressed:
            keyevent = KeyDown
        else:
            keyevent = KeyUp

        events.append(
            keyevent(
                libkey.vk,
                libkey.c.decode('ascii', errors='ignore'),
                _ffi.string(libkey.text).decode('utf-8'),
                libkey.shift,
                libkey.lalt,
                libkey.ralt,
                libkey.lctrl,
                libkey.rctrl,
                libkey.lmeta,
                libkey.rmeta,
                )
            )

    if _lib.TCOD_console_is_window_closed():
        events.append(Quit())

    _eventQueue.extend(events)

def get():
    """Flushes the event queue and returns the list of events.

    This function returns :any:`Event` objects that can be identified by their
    type attribute or their class.

    Returns: Iterator[Type[Event]]: An iterable of Events or anything
        put in a :any:`push` call.

        If the iterator is deleted or otherwise interrupted before finishing
        the excess items are preserved for the next call.
    """
    _processEvents()
    return _event_generator()

def _event_generator():
    while _eventQueue:
        # if there is an interruption the rest of the events stay untouched
        # this means you can break out of a event.get loop without losing
        # the leftover events
        yield(_eventQueue.pop(0))


def wait(timeout=None, flush=True):
    """Wait for an event.

    Args:
        timeout (Optional[int]): The time in seconds that this function will
            wait before giving up and returning None.

            With the default value of None, this will block forever.
        flush (bool): If True a call to :any:`tdl.flush` will be made before
            listening for events.

    Returns: Type[Event]: An event, or None if the function
             has timed out.
             Anything added via :any:`push` will also be returned.
    """
    if timeout is not None:
        timeout = timeout + _time.clock() # timeout at this time
    while True:
        if _eventQueue:
            return _eventQueue.pop(0)
        if flush:
            # a full 'round' of events need to be processed before flushing
            _tdl.flush()
        if timeout and _time.clock() >= timeout:
            return None # return None on timeout
        _time.sleep(0.001) # sleep 1ms
        _processEvents()


def push(event):
    """Push an event into the event buffer.

    Args:
        event (Any): This event will be available on the next call to
            :any:`event.get`.

            An event pushed in the middle of a :any:`get` will not show until
            the next time :any:`get` called preventing push related
            infinite loops.

            This object should at least have a 'type' attribute.
    """
    _pushedEvents.append(event)

def key_wait():
    """Waits until the user presses a key.
    Then returns a :any:`KeyDown` event.

    Key events will repeat if held down.

    A click to close the window will be converted into an Alt+F4 KeyDown event.

    Returns:
        tdl.event.KeyDown: The pressed key.
    """
    while 1:
        for event in get():
            if event.type == 'KEYDOWN':
                return event
            if event.type == 'QUIT':
                # convert QUIT into alt+F4
                return KeyDown('F4', '', True, False, True, False, False)
        _time.sleep(.001)

def set_key_repeat(delay=500, interval=0):
    """Does nothing.
    """
    pass

def is_window_closed():
    """Returns True if the exit button on the window has been clicked and
    stays True afterwards.

    Returns: bool:
    """
    return _lib.TCOD_console_is_window_closed()

__all__ = [_var for _var in locals().keys() if _var[0] != '_']

App.runOnce = _style.backport(App.run_once)
keyWait = _style.backport(key_wait)
setKeyRepeat = _style.backport(set_key_repeat)
isWindowClosed = _style.backport(is_window_closed)

