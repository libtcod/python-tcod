To use these unicode fonts with python-tdl you should call tdl.setFont with
columns=64 and rows=1024 leaving all other parameters at the defaults.
Remember to do this before the call to tdl.init

The unicode support varies accross the fonts here.  This is detailed in
README-X11.txt with some fonts meeting better support targets than others.

The fonts with the best unicode support are:
6x13.png 8x13.png 9x15.png 9x18.png 10x20.png

Also note that libtcod will be put under stress when using an extra large
tileset such as 10x20.png;  Uncompressed the 10x20.png image will take around
40MB of video memory.  Most cards can handle that but you should get around it
anyway by using the SDL rasterizer instead of OPENGL.
