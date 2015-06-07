To use these unicode fonts with python-tdl you should call tdl.setFont with only the filename leaving all other parameters at the defaults.  Remember to do this before the call to tdl.init.  If using libtcod then note that each tileset has 64 columns and 1024 rows.

unifont_16x16.png is the recommended font with the best unicode support.

Unicode support varies across the rest of the fonts.  This is detailed in README-X11.txt with some fonts meeting better support targets than others.

After unifont the fonts with the best unicode support are:
    6x13.png 8x13.png 9x15.png 9x18.png 10x20.png

Also note that libtcod will be put under stress when using an extra large tileset such as 10x20.png;  When decompressed the 10x20.png image will take around 40MB of video memory.  Modern graphic cards can handle this can get better compatibility  by using the SDL rasterizer instead of OPENGL.
