
#ifndef TCOD_TCOD_H_
#define TCOD_TCOD_H_

#include "../libtcod/src/libtcod/color.h"
#include "../libtcod/src/libtcod/console.h"

#ifdef __cplusplus
extern "C" {
#endif
int bresenham(int x1, int y1, int x2, int y2, int n, int* __restrict out);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif /* TCOD_TCOD_H_ */
