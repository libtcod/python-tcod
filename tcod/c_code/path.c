#include "path.h"

#include <SDL_stdinc.h>

float PathCostArrayFloat32(const int x1, const int y1,
                           const int x2, const int y2,
                           const PathCostArray *map){
  return ((float*)map->array)[y2 * map->width + x2];
}

float PathCostArrayInt8(const int x1, const int y1,
                        const int x2, const int y2,
                        const PathCostArray *map){
  return (float)((int8_t*)map->array)[y2 * map->width + x2];
}

float PathCostArrayInt16(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map){
  return (float)((int16_t*)map->array)[y2 * map->width + x2];
}

float PathCostArrayInt32(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map){
  return (float)((int32_t*)map->array)[y2 * map->width + x2];
}

float PathCostArrayUInt8(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map){
  return (float)((uint8_t*)map->array)[y2 * map->width + x2];
}

float PathCostArrayUInt16(const int x1, const int y1,
                          const int x2, const int y2,
                          const PathCostArray *map){
  return (float)((uint16_t*)map->array)[y2 * map->width + x2];
}

float PathCostArrayUInt32(const int x1, const int y1,
                          const int x2, const int y2,
                          const PathCostArray *map){
  return (float)((uint32_t*)map->array)[y2 * map->width + x2];
}
