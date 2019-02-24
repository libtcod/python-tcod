
#ifndef TCOD_TCOD_H_
#define TCOD_TCOD_H_

#include "../libtcod/src/libtcod/color.h"
#include "../libtcod/src/libtcod/console.h"

#ifdef __cplusplus
extern "C" {
#endif
int LineWhere(int x1, int y1, int x2, int y2, int *x_out, int *y_out);

void draw_rect(
    TCOD_Console* console,
    int x,
    int y,
    int width,
    int height,
    int ch,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag);
void console_print(
    TCOD_Console* console,
    int x,
    int y,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    TCOD_alignment_t alignment);
int print_rect(
    TCOD_Console *con,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    TCOD_alignment_t alignment);
int get_height_rect(
    TCOD_Console *con,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n);
int get_height_rect2(
    int width,
    const char* str,
    int str_n);
void print_frame(
    TCOD_Console *con,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    bool empty);
#ifdef __cplusplus
} // extern "C"
#endif
#endif /* TCOD_TCOD_H_ */
