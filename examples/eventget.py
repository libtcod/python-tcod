"""
    An interactive example of what events are available
"""

import sys
sys.path.insert(0, '../')

import tdl

console = tdl.init(80, 60)

def print_line_append(string):
    # move everything on the console up a line with blit
    console.blit(console, 0, 0, 80, 57, 0, 1)
    # clear the bottom line with drawRect
    console.drawRect(' ', 0, 57, None, 1)
    # print the string
    console.drawStr(string, 0, 57)

tdl.setFPS(24) # slow down the program so that the user can clearly see the motion events

while 1:
    for event in tdl.event.get():
        if event.type == tdl.QUIT:
            raise SystemExit()
        elif event.type == tdl.KEYDOWN:
            print_line_append('KEYDOWN event - key=%.2i char=%s keyname=%s alt=%i ctrl=%i shift=%i' % (event.key, repr(event.char), repr(event.keyname), event.alt, event.ctrl, event.shift))
        elif event.type == tdl.KEYUP:
            print_line_append('KEYUP   event - key=%.2i char=%s keyname=%s alt=%i ctrl=%i shift=%i' % (event.key, repr(event.char), repr(event.keyname), event.alt, event.ctrl, event.shift))
        elif event.type == tdl.MOUSEMOTION:
            console.drawRect(' ', 0, 59)
            console.drawStr('MOUSEMOTION event - pos=%i,%i cell=%i,%i motion=%i,%i cellmotion=%i,%i' % (event.pos + event.cell + event.motion + event.cellmotion), 0, 59)
        elif event.type == tdl.MOUSEDOWN:
            print_line_append('MOUSEDOWN event - pos=%i,%i cell=%i,%i button=%i' % (event.pos + event.cell + (event.button,)))
        elif event.type == tdl.MOUSEUP:
            print_line_append('MOUSEUP   event - pos=%i,%i cell=%i,%i button=%i' % (event.pos + event.cell + (event.button,)))
    tdl.flush()
