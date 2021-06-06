
#include "tcod.h"

#include <stdlib.h>

#include "../libtcod/src/libtcod/bresenham.h"
#include "../libtcod/src/libtcod/console_drawing.h"
#include "../libtcod/src/libtcod/console_printing.h"
#include "../libtcod/src/libtcod/error.h"
#include "../libtcod/src/libtcod/utility.h"
/**
    Write a Bresenham line to the `out[n * 2]` array.

    The result includes both endpoints.

    The length of the array is returned, when `out` is given `n` must be equal
    or greater then the length.
 */
int bresenham(int x1, int y1, int x2, int y2, int n, int* __restrict out) {
  // Bresenham length is Chebyshev distance.
  int length = MAX(abs(x1 - x2), abs(y1 - y2)) + 1;
  if (!out) { return length; }
  if (n < length) { return TCOD_set_errorv("Bresenham output length mismatched."); }
  TCOD_bresenham_data_t bresenham;
  out[0] = x1;
  out[1] = y1;
  out += 2;
  if (x1 == x2 && y1 == y2) { return length; }
  TCOD_line_init_mt(x1, y1, x2, y2, &bresenham);
  while (!TCOD_line_step_mt(&out[0], &out[1], &bresenham)) { out += 2; }
  return length;
}
