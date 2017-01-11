
// extra functions provided for the python-tdl library

#include <libtcod.h>
#include <wrappers.h>

void SDL_main(void){};
void CustomSDLMain(void){}; /* CustomSDLMain stub for Mac build */

TCOD_value_t TDL_list_get_union(TCOD_list_t l,int idx){
    TCOD_value_t item;
    item.custom = TCOD_list_get(l, idx);
    return item;
}

bool TDL_list_get_bool(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).b;
}

char TDL_list_get_char(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).c;
}

int TDL_list_get_int(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).i;
}

float TDL_list_get_float(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).f;
}

char* TDL_list_get_string(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).s;
}

TCOD_color_t TDL_list_get_color(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).col;
}

TCOD_dice_t TDL_list_get_dice(TCOD_list_t l,int idx){
    return TDL_list_get_union(l, idx).dice;
}


// get a TCOD color type from a 0xRRGGBB formatted integer
TCOD_color_t TDL_color_from_int(int color){
    TCOD_color_t tcod_color={(color >> 16) & 0xff,
                             (color >> 8) & 0xff,
                              color & 0xff};
    return tcod_color;
}

int TDL_color_to_int(TCOD_color_t *color){
    return (color->r << 16) | (color->g << 8) | color->b;
}

int* TDL_color_int_to_array(int color){
    static int array[3];
    array[0] = (color >> 16) & 0xff;
    array[1] = (color >> 8) & 0xff;
    array[2] = color & 0xff;
    return array;
}

int TDL_color_RGB(int r, int g, int b){
    return ((r << 16) & 0xff) | ((g << 8) & 0xff) | (b & 0xff);
}

int TDL_color_HSV(float h, float s, float v){
    TCOD_color_t tcod_color=TCOD_color_HSV(h, s, v);
    return TDL_color_to_int(&tcod_color);
}

bool TDL_color_equals(int c1, int c2){
    return (c1 == c2);
}

int TDL_color_add(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_add(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

int TDL_color_subtract(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_subtract(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

int TDL_color_multiply(int c1, int c2){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_multiply(tc1, tc2);
    return TDL_color_to_int(&tc1);
}

int TDL_color_multiply_scalar(int c, float value){
    TCOD_color_t tc=TDL_color_from_int(c);
    tc=TCOD_color_multiply_scalar(tc, value);
    return TDL_color_to_int(&tc);
}

int TDL_color_lerp(int c1, int c2, float coef){
    TCOD_color_t tc1=TDL_color_from_int(c1);
    TCOD_color_t tc2=TDL_color_from_int(c2);
    tc1=TCOD_color_lerp(tc1, tc2, coef);
    return TDL_color_to_int(&tc1);
}

float TDL_color_get_hue(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_hue(tcod_color);
}
float TDL_color_get_saturation(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_saturation(tcod_color);
}
float TDL_color_get_value(int color){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    return TCOD_color_get_value(tcod_color);
}
int TDL_color_set_hue(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_hue(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
int TDL_color_set_saturation(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_saturation(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
int TDL_color_set_value(int color, float h){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_set_value(&tcod_color, h);
    return TDL_color_to_int(&tcod_color);
}
int TDL_color_shift_hue(int color, float hshift){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_shift_hue(&tcod_color, hshift);
    return TDL_color_to_int(&tcod_color);
}
int TDL_color_scale_HSV(int color, float scoef, float vcoef){
    TCOD_color_t tcod_color=TDL_color_from_int(color);
    TCOD_color_scale_HSV(&tcod_color, scoef, vcoef);
    return TDL_color_to_int(&tcod_color);
}


#define TRANSPARENT_BIT 1
#define WALKABLE_BIT 2
#define FOV_BIT 4

// set map transparent and walkable flags from a buffer
void TDL_map_data_from_buffer(TCOD_map_t map, uint8 *buffer){
    int width=TCOD_map_get_width(map);
    int height=TCOD_map_get_height(map);
    int x;
    int y;

    int i = width*height-1;
    int16 data;
    for(y=height-1;y>=0;y--){
        for(x=width-1;x>=0;x--){
            data = *(buffer + i--);
            TCOD_map_set_properties(map, x, y, (data & TRANSPARENT_BIT) != 0,
                                               (data & WALKABLE_BIT) != 0);
        }
    }
}

// get fov from tcod map
void TDL_map_fov_to_buffer(TCOD_map_t map, uint8 *buffer,
                                  bool cumulative){
    int width=TCOD_map_get_width(map);
    int height=TCOD_map_get_height(map);
    int x;
    int y;
    int i = width*height;
    for(y=height-1;y>=0;y--){
        for(x=width-1;x>=0;x--){
            i--;
            if(TCOD_map_is_in_fov(map, x, y)){
                *(buffer + i) |= FOV_BIT;
            }else if(!cumulative){
                *(buffer + i) &= ~FOV_BIT;
            }
        }
    }
}

// set functions are called conditionally for ch/fg/bg (-1 is ignored)/
// colors are converted to TCOD_color_t types in C and is much faster than in
// Python.
// Also Python indexing is used, negative x/y will index to (width-x, etc.)
int TDL_console_put_char_ex(TCOD_console_t console, int x, int y,
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
        color = TDL_color_from_int(fg);
        TCOD_console_set_char_foreground(console, x, y, color);
    }
    if(bg != -1){
        color = TDL_color_from_int(bg);
        TCOD_console_set_char_background(console, x, y, color, blend);
    }
    return 0;
}

int TDL_console_get_bg(TCOD_console_t console, int x, int y){
    TCOD_color_t tcod_color=TCOD_console_get_char_background(console, x, y);
    return TDL_color_to_int(&tcod_color);
}

int TDL_console_get_fg(TCOD_console_t console, int x, int y){
    TCOD_color_t tcod_color=TCOD_console_get_char_foreground(console, x, y);
    return TDL_color_to_int(&tcod_color);
}

void TDL_console_set_bg(TCOD_console_t console, int x, int y, int color,
                               TCOD_bkgnd_flag_t flag){
    TCOD_console_set_char_background(console, x, y, TDL_color_from_int(color),
                                     flag);
}

void TDL_console_set_fg(TCOD_console_t console, int x, int y, int color){
    TCOD_console_set_char_foreground(console, x, y, TDL_color_from_int(color));
}
