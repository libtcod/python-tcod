#!/usr/bin/env python
"""
    This script shows the basic use of the tdl module.

    The font is configured and the module is initialized, the script then goes
    into an infinite loop where it will draw "Hello World" on the screen and
    then check for an event that tells that the user has closed the window.

    When the window is closed the script quits out by raising a SystemExit
    exception.
"""

import tdl

# Define the window size (in character tiles.) We'll pick something small.
WIDTH, HEIGHT = 40, 30 # 320x240 when the font size is taken into account

# Set the font to the example font in the tutorial folder.  This font is
# equivalent to the default font you will get if you skip this call.
# With the characters rows first and the font size included in the filename
# you won't have to specify any parameters other than the font file itself.
tdl.setFont('terminal8x8_gs_ro.png')

# Call tdl.init to create the root console.
# We will call drawing operations on the returned object.
console = tdl.init(WIDTH, HEIGHT, 'python-tdl tutorial')

# Start an infinite loop.  Drawing and game logic will be put in this loop.
while True:

    # Reset the console to a blank slate before drawing on it.
    console.clear()

    # Now draw out 'Hello World' starting at an x,y of 1,2.
    console.draw_str(1, 2, 'Hello World')

    # Now to update the image on the window we make sure to call tdl.flush
    # in every loop.
    tdl.flush()

    # Handle events by iterating over the values returned by tdl.event.get
    for event in tdl.event.get():
        # Check if this is a 'QUIT' event
        if event.type == 'QUIT':
            # Later we may want to save the game or confirm if the user really
            # wants to quit but for now we break out of the loop by raising a
            # SystemExit exception.
            # The optional string parameter will be printed out on the
            # terminal after the script exits.
            raise SystemExit('The window has been closed.')
