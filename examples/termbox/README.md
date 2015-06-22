API of `termbox` Python module implemented in `tld`.

The code here are modified files from
[termbox repository](https://github.com/nsf/termbox/), so please consult
it for the license and other info.


The code consists of two part - `termbox.py` module with API, translation
of official binding form the description below into `tld`:

https://github.com/nsf/termbox/blob/b20c0a11/src/python/termboxmodule.pyx

And the example `termboxtest.py` which is copied verbatim from:

https://github.com/nsf/termbox/blob/b20c0a11/test_termboxmodule.py


### API Mapping Notes

Notes taken while mapping the Termbox class:

    tb_init() 			// initialization                           console = tdl.init(132, 60)
    tb_shutdown() 		// shutdown

    tb_width() 			// width of the terminal screen             console.width
    tb_height() 		// height of the terminal screen            console.height

    tb_clear() 			// clear buffer                             console.clear()
    tb_present() 		// sync internal buffer with terminal       tdl.flush()

    tb_put_cell()
    tb_change_cell()                                                    console.draw_char(x, y, ch, fg, bg)
    tb_blit() 			// drawing functions

    tb_select_input_mode() 	// change input mode
    tb_peek_event() 		// peek a keyboard event
    tb_poll_event() 		// wait for a keyboard event              * tdl.event.get()


     * - means the translation is not direct



   init...
     tdl doesn't allow to resize window (or rather libtcod)
     tb works in existing terminal window and queries it rather than making own

   colors...
     tdl uses RGB values
     tb uses it own constants

   event...
     tb returns event one by one
     tdl return an event iterator


   tb Event                       tdl Event
   .type                          .type
     EVENT_KEY                      KEYDOWN
