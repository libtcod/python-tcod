
#include <string>

#include "tcod.h"

#include "../libtcod/src/libtcod/bresenham.h"
#include "../libtcod/src/libtcod/console_drawing.h"
#include "../libtcod/src/libtcod/console_printing.h"
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
