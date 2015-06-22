"""
Implementation of Termbox Python API in tdl.

See README.md for details.
"""

import tdl

"""
Implementation status:
 [ ] tdl.init() needs a window, made 132x60
 [ ] Termbox.close() is not implemented, does nothing
 [ ] poll_event needs review, because it does not
     completely follows the original logic
 [ ] peek is stubbed, but not implemented
 [ ] not all keys/events are mapped
"""

class TermboxException(Exception):
	def __init__(self, msg):
		self.msg = msg
	def __str__(self):
		return self.msg

_instance = None

# keys ----------------------------------
KEY_F1               = (0xFFFF-0)
KEY_F2               = (0xFFFF-1)
KEY_F3               = (0xFFFF-2)
KEY_F4               = (0xFFFF-3)
KEY_F5               = (0xFFFF-4)
KEY_F6               = (0xFFFF-5)
KEY_F7               = (0xFFFF-6)
KEY_F8               = (0xFFFF-7)
KEY_F9               = (0xFFFF-8)
KEY_F10              = (0xFFFF-9)
KEY_F11              = (0xFFFF-10)
KEY_F12              = (0xFFFF-11)
KEY_INSERT           = (0xFFFF-12)
KEY_DELETE           = (0xFFFF-13)

KEY_PGUP             = (0xFFFF-16)
KEY_PGDN             = (0xFFFF-17)

KEY_MOUSE_LEFT      =(0xFFFF-22)
KEY_MOUSE_RIGHT      =(0xFFFF-23)
KEY_MOUSE_MIDDLE      =(0xFFFF-24)
KEY_MOUSE_RELEASE      =(0xFFFF-25)
KEY_MOUSE_WHEEL_UP      =(0xFFFF-26)
KEY_MOUSE_WHEEL_DOWN      =(0xFFFF-27)

KEY_CTRL_TILDE       = 0x00
KEY_CTRL_2           = 0x00
KEY_CTRL_A           = 0x01
KEY_CTRL_B           = 0x02
KEY_CTRL_C           = 0x03
KEY_CTRL_D           = 0x04
KEY_CTRL_E           = 0x05
KEY_CTRL_F           = 0x06
KEY_CTRL_G           = 0x07
KEY_BACKSPACE        = 0x08
KEY_CTRL_H           = 0x08
KEY_TAB              = 0x09
KEY_CTRL_I           = 0x09
KEY_CTRL_J           = 0x0A
KEY_CTRL_K           = 0x0B
KEY_CTRL_L           = 0x0C
KEY_ENTER            = 0x0D
KEY_CTRL_M           = 0x0D
KEY_CTRL_N           = 0x0E
KEY_CTRL_O           = 0x0F
KEY_CTRL_P           = 0x10
KEY_CTRL_Q           = 0x11
KEY_CTRL_R           = 0x12
KEY_CTRL_S           = 0x13
KEY_CTRL_T           = 0x14
KEY_CTRL_U           = 0x15
KEY_CTRL_V           = 0x16
KEY_CTRL_W           = 0x17
KEY_CTRL_X           = 0x18
KEY_CTRL_Y           = 0x19
KEY_CTRL_Z           = 0x1A


# -- mapped to tdl
KEY_HOME             = 'HOME'
KEY_END              = 'END'
KEY_ARROW_UP         = 'UP'
KEY_ARROW_DOWN       = 'DOWN'
KEY_ARROW_LEFT       = 'LEFT'
KEY_ARROW_RIGHT      = 'RIGHT'
KEY_ESC              = 'ESCAPE'
# /--


KEY_CTRL_LSQ_BRACKET = 0x1B
KEY_CTRL_3           = 0x1B
KEY_CTRL_4           = 0x1C
KEY_CTRL_BACKSLASH   = 0x1C
KEY_CTRL_5           = 0x1D
KEY_CTRL_RSQ_BRACKET = 0x1D
KEY_CTRL_6           = 0x1E
KEY_CTRL_7           = 0x1F
KEY_CTRL_SLASH       = 0x1F
KEY_CTRL_UNDERSCORE  = 0x1F
KEY_SPACE            = 0x20
KEY_BACKSPACE2       = 0x7F
KEY_CTRL_8           = 0x7F

MOD_ALT              = 0x01

# attributes ----------------------

#-- mapped to tdl
DEFAULT   = Ellipsis

BLACK     = 0x000000
RED       = 0xFF0000
GREEN     = 0x00FF00
YELLOW    = 0xFFFF00
BLUE      = 0x0000FF
MAGENTA   = 0xFF00FF
CYAN      = 0x00FFFF
WHITE     = 0xFFFFFF
#/--

BOLD      = 0x10
UNDERLINE = 0x20
REVERSE   = 0x40

# misc ----------------------------

HIDE_CURSOR      = -1
INPUT_CURRENT    = 0
INPUT_ESC        = 1
INPUT_ALT        = 2
OUTPUT_CURRENT   = 0
OUTPUT_NORMAL    = 1
OUTPUT_256       = 2
OUTPUT_216       = 3
OUTPUT_GRAYSCALE = 4


# -- mapped to tdl
EVENT_KEY        = 'KEYDOWN'
# /--
EVENT_RESIZE     = 2
EVENT_MOUSE		= 3

class Event:
    """ Aggregate for Termbox Event structure """
    type = None
    ch = None
    key = None
    mod = None
    width = None
    height = None
    mousex = None
    mousey = None

    def gettuple(self):
         return (self.type, self.ch, self.key, self.mod, self.width, self.height, self.mousex, self.mousey)

class Termbox:
	def __init__(self, width=132, height=60):
		global _instance
		if _instance:
			raise TermboxException("It is possible to create only one instance of Termbox")

		try:
			self.console = tdl.init(width, height)
                except tdl.TDLException as e:
			raise TermboxException(e)

                self.e = Event() # cache for event data

		_instance = self

	def __del__(self):
		self.close()

	def __exit__(self, *args):#t, value, traceback):
		self.close()

	def __enter__(self):
		return self

	def close(self):
		global _instance
		# tb_shutdown()
		_instance = None
                # TBD, does nothing

	def present(self):
		"""Sync state of the internal cell buffer with the terminal.
		"""
		tdl.flush()

	def change_cell(self, x, y, ch, fg, bg):
		"""Change cell in position (x;y).
		"""
		self.console.draw_char(x, y, ch, fg, bg)

	def width(self):
		"""Returns width of the terminal screen.
		"""
		return self.console.width

	def height(self):
		"""Return height of the terminal screen.
		"""
		return self.console.height

	def clear(self):
		"""Clear the internal cell buffer.
		"""
		self.console.clear()

	def set_cursor(self, x, y):
		"""Set cursor position to (x;y).

		   Set both arguments to HIDE_CURSOR or use 'hide_cursor' function to hide it.
		"""
		tb_set_cursor(x, y)

	def hide_cursor(self):
		"""Hide cursor.
		"""
		tb_set_cursor(-1, -1)

	def select_input_mode(self, mode):
		"""Select preferred input mode: INPUT_ESC or INPUT_ALT.

		   INPUT_CURRENT returns the selected mode without changing anything.
		"""
		return int(tb_select_input_mode(mode))

	def select_output_mode(self, mode):
		"""Select preferred output mode: one of OUTPUT_* constants.

		   OUTPUT_CURRENT returns the selected mode without changing anything.
		"""
		return int(tb_select_output_mode(mode))

	def peek_event(self, timeout=0):
		"""Wait for an event up to 'timeout' milliseconds and return it.

		   Returns None if there was no event and timeout is expired.
		   Returns a tuple otherwise: (type, unicode character, key, mod, width, height, mousex, mousey).
		"""
		"""
		cdef tb_event e
		with self._poll_lock:
			with nogil:
				result = tb_peek_event(&e, timeout)
		assert(result >= 0)
		if result == 0:
			return None
		if e.ch:
			uch = unichr(e.ch)
		else:
			uch = None
		"""
		pass #return (e.type, uch, e.key, e.mod, e.w, e.h, e.x, e.y)

	def poll_event(self):
		"""Wait for an event and return it.

		   Returns a tuple: (type, unicode character, key, mod, width, height, mousex, mousey).
		"""
		"""
		cdef tb_event e
		with self._poll_lock:
			with nogil:
				result = tb_poll_event(&e)
		assert(result >= 0)
		if e.ch:
			uch = unichr(e.ch)
		else:
			uch = None
		"""
                for e in tdl.event.get():
                  # [ ] not all events are passed thru
                  self.e.type = e.type
                  if e.type == 'KEYDOWN':
                    self.e.key = e.key
                    return self.e.gettuple()

		#return (e.type, uch, e.key, e.mod, e.w, e.h, e.x, e.y)
