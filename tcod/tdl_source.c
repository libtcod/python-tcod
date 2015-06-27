
// extra functions provided for the python-tdl library

#include "libtcod.h"
#include "wrappers.h"

// get a TCOD color type from a 0xRRGGBB formatted integer
static TCOD_color_t TDL_color_from_int(int color){
    TCOD_color_t tcod_color={(color >> 16) & 0xff,
                             (color >> 8) & 0xff,
                              color & 0xff};
    return tcod_color;
}

static int TDL_color_to_int(TCOD_color_t *color){
    return (color->r << 16) | (color->g << 8) | color->b;
}

static int TDL_color_RGB(int r, int g, int b){
    return ((r >> 16) & 0xff) | ((g >> 8) & 0xff) | (b & 0xff);
}

static int TDL_color_HSV(float h, float s, float v){
    TCOD_color_t tcod_color=TCOD_color_HSV(h, s, v);
    return TDL_color_to_int(&tcod_color);
}

static bool TDL_color_equals(int c1, int c2){
    return (c1 == c2);
}

static int TDL_color_add(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_add(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

static int TDL_color_subtract(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_subtract(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

static int TDL_color_multiply(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_multiply(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

static int TDL_color_multiply_scalar(int c, float value){
    TCOD_color_t tc=TDL_color_from_int(c);
    tc=TCOD_color_multiply_scalar(tc, value);
    return TDL_color_to_int(&tc);
}

static int TDL_color_lerp(int c1, int c2, float coef){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_lerp(tc1, tc2, coef);
    return TDL_color_to_int(&tc1);
}

static float TDL_color_get_hue(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_hue(tcod_color);
}
static float TDL_color_get_saturation(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_saturation(tcod_color);
}
static float TDL_color_get_value(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_value(tcod_color);
}
static int TDL_color_set_hue(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_hue(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
static int TDL_color_set_saturation(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_saturation(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
static int TDL_color_set_value(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_value(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
static int TDL_color_shift_hue(int color, float hshift){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_shift_hue(&tcod_color, hshift);
    return TDL_color_to_int(&tcod_color);
}
static int TDL_color_scale_HSV(int color, float scoef, float vcoef){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_scale_HSV(&tcod_color, scoef, vcoef);
    return TDL_color_to_int(&tcod_color);
}

/* HSV transformations */
//void TCOD_color_set_HSV (TCOD_color_t *c,float h, float s, float v);
//void TCOD_color_get_HSV (TCOD_color_t c,float * h, float * s, float * v);
//void TCOD_color_shift_hue (TCOD_color_t *c, float hshift);
//void TCOD_color_scale_HSV (TCOD_color_t *c, float scoef, float vcoef);
/* color map */
//void TCOD_color_gen_map(TCOD_color_t *map, int nb_key, TCOD_color_t const *key_color, int const *key_index);


// set functions are called conditionally for ch/fg/bg (-1 is ignored)/
// colors are converted to TCOD_color_t types in C and is much faster than in 
// Python.
// Also Python indexing is used, negative x/y will index to (width-x, etc.)
static int TDL_console_put_char_ex(TCOD_console_t console, int x, int y,
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
