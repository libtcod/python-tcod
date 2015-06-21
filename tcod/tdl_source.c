
#include <libtcod.h>

// set functions are called conditionally for ch/fg/bg (-1 is ignored)/
// colors are converted to TCOD_color_t types in C and is much faster than in 
// Python.
// Also Python indexing is used, negative x/y will index to (width-x, etc.)
int TDL_set_char(TCOD_console_t console, int x, int y,
                 int ch, int fg, int bg, TCOD_bkgnd_flag_t blend){
    int width=TCOD_console_get_width(console);
    int height=TCOD_console_get_height(console);
    TCOD_color_t color;
    
    if(x < -width || x >= width || y < -height || y >= height){
        return -1; // outside of console
    }
    
    // normalize x, y
    if(x<0){x += width;};
    if(y<0){y += height;};
    
    if(ch != -1){
        TCOD_console_set_char(console, x, y, ch);
    }
    if(fg != -1){
        color.r = fg >> 16 & 0xff;
        color.g = fg >> 8 & 0xff;
        color.b = fg & 0xff;
        TCOD_console_set_char_foreground(console, x, y, color);
    }
    if(bg != -1){
        color.r = bg >> 16 & 0xff;
        color.g = bg >> 8 & 0xff;
        color.b = bg & 0xff;
        TCOD_console_set_char_background(console, x, y, color, blend);
    }
    return 0;
}
