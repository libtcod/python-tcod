
/* This header is the entry point for the cffi parser.
   Anything included here will be accessible from tcod.libtcod.lib */

#ifndef TDL_NO_SDL2_EXPORTS
/* Ignore headers with issues. */
#define SDL_thread_h_
#include <SDL.h>
#endif

#include "../libtcod/src/libtcod/libtcod.h"
#include "../libtcod/src/libtcod/libtcod_int.h"
#include "../libtcod/src/libtcod/wrappers.h"
#include "../libtcod/src/libtcod/tileset/truetype.h"

#include "noise.h"
#include "path.h"
#include "random.h"
#include "tcod.h"
#include "tdl.h"
