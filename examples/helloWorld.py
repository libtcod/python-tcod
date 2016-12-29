#!/usr/bin/env python
"""
    Example showing some of the most basic functions needed to say "Hello World"
"""
# you can skip past this part and go onto the next.
# what this does is allow tdl to be imported without installing it first.
import sys
sys.path.insert(0, '../')

# import the tdl library and all of it's functions, this gives us access to anything
# starting with "tdl."
import tdl

# start the main console, this will open the window that you see and give you a Console.
# we make a small window that's 20 tiles wide and 16 tile's tall for this example.
console = tdl.init(20, 16)

# draw the string "Hello World" at the top left corner using the default colors:
# a white forground on a black background.
console.draw_str(0, 0, 'Hello World')

# display the changes to the console with flush.
# if you forget this part the screen will stay black emptiness forever.
tdl.flush()

# wait for a key press, any key pressed now will cause the program flow to the next part
# which closes out of the program.
tdl.event.keyWait()

# if you run this example in IDLE then we'll need to delete the console manually
# otherwise IDLE prevents the window from closing causing it to hang
del console
