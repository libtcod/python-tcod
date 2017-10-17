#!/usr/bin/env python

import tdl


WIDTH, HEIGHT = 80, 60
console = None

def main():
    global console
    tdl.set_font('data/terminal8x8_gs_ro.png')
    console = tdl.init(WIDTH, HEIGHT)
    console.draw_str(0, 0, "Hello World")
    tdl.flush()
    tdl.event.key_wait()


if __name__ == '__main__':
    main()
