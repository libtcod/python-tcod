
#include <string>

#include "tcod.h"

#include "../libtcod/src/libtcod/bresenham.h"
#include "../libtcod/src/libtcod/console/drawing.h"
#include "../libtcod/src/libtcod/console/printing.h"
/**
 *  Write a Bresenham line to the `x_out` and `y_out` arrays.
 *
 *  `x_out` and `y_out` must be large enough to contain the entire line that
 *  this will output.  Typically `max(abs(x1 - x2), abs(y1 - y2)) + 1`.
 *
 *  This function includes both endpoints.
 */
int LineWhere(int x1, int y1, int x2, int y2, int *x_out, int *y_out) {
  TCOD_bresenham_data_t bresenham;
  *x_out = x1;
  *y_out = y1;
  if (x1 == x2 && y1 == y2) { return 0; }
  TCOD_line_init_mt(x1, y1, x2, y2, &bresenham);
  while (!TCOD_line_step_mt(++x_out, ++y_out, &bresenham)) {}
  return 0;
}

void draw_rect(
    TCOD_Console* console,
    int x,
    int y,
    int width,
    int height,
    int ch,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag)
{
  tcod::console::draw_rect(console, x, y, width, height, ch, fg, bg, flag);
}
void console_print(
    TCOD_Console* console,
    int x,
    int y,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    TCOD_alignment_t alignment)
{
  tcod::console::print(console, x, y, std::string(str, str_n),
                       fg, bg, flag, alignment);
}
int print_rect(
    TCOD_Console* console,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    TCOD_alignment_t alignment)
{
  return tcod::console::print_rect(console, x, y, width, height,
                                   std::string(str, str_n), fg, bg,
                                   flag, alignment);
}
int get_height_rect(
    TCOD_Console* console,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n)
{
  return tcod::console::get_height_rect(console, x, y, width, height,
                                        std::string(str, str_n));
}
int get_height_rect2(
    int width,
    const char* str,
    int str_n)
{
  return tcod::console::get_height_rect(width, std::string(str, str_n));
}
void print_frame(
    TCOD_Console* console,
    int x,
    int y,
    int width,
    int height,
    const char* str,
    int str_n,
    const TCOD_color_t* fg,
    const TCOD_color_t* bg,
    TCOD_bkgnd_flag_t flag,
    bool empty)
{
  tcod::console::print_frame(console, x, y, width, height,
                             std::string(str, str_n), fg, bg, flag, empty);
}
