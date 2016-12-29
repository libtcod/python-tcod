#!/usr/bin/env python
"""

"""

import tdl

WIDTH, HEIGHT = 40, 30 # Defines the window size.

# Create a dictionary that maps keys to vectors.
# Names of the available keys can be found in the online documentation:
# http://packages.python.org/tdl/tdl.event-module.html
MOVEMENT_KEYS = {
                 # standard arrow keys
                 'UP': [0, -1],
                 'DOWN': [0, 1],
                 'LEFT': [-1, 0],
                 'RIGHT': [1, 0],

                 # diagonal keys
                 # keep in mind that the keypad won't use these keys even if
                 # num-lock is off
                 'HOME': [-1, -1],
                 'PAGEUP': [1, -1],
                 'PAGEDOWN': [1, 1],
                 'END': [-1, 1],

                 # number-pad keys
                 # These keys will always show as KPx regardless if num-lock
                 # is on or off.  Keep in mind that some keyboards and laptops
                 # may be missing a keypad entirely.
                 # 7 8 9
                 # 4   6
                 # 1 2 3
                 'KP1': [-1, 1],
                 'KP2': [0, 1],
                 'KP3': [1, 1],
                 'KP4': [-1, 0],
                 'KP6': [1, 0],
                 'KP7': [-1, -1],
                 'KP8': [0, -1],
                 'KP9': [1, -1],
                 }


tdl.setFont('terminal8x8_gs_ro.png') # Configure the font.

# Create the root console.
console = tdl.init(WIDTH, HEIGHT, 'python-tdl tutorial')

# player coordinates
playerX, playerY = 1, 2

while True: # Continue in an infinite game loop.

    console.clear() # Blank the console.

    # Using "(x, y) in console" we can quickly check if a position is inside of
    # a console.  And skip a draw operation that would otherwise fail.
    if (playerX, playerY) in console:
        console.draw_char(playerX, playerY, '@')

    tdl.flush() # Update the window.

    for event in tdl.event.get(): # Iterate over recent events.
        if event.type == 'KEYDOWN':
            # We mix special keys with normal characters so we use keychar.
            if event.keychar.upper() in MOVEMENT_KEYS:
                # Get the vector and unpack it into these two variables.
                keyX, keyY = MOVEMENT_KEYS[event.keychar.upper()]
                # Then we add the vector to the current player position.
                playerX += keyX
                playerY += keyY

        if event.type == 'QUIT':
            # Halt the script using SystemExit
            raise SystemExit('The window has been closed.')
