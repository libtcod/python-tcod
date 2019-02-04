#include "path.h"

#include "../libtcod/src/libtcod/pathfinding/generic.h"

#include <SDL_stdinc.h>

static char* PickArrayValue(const struct PathCostArray *map, int i, int j) {
  return map->array + map->strides[0] * i + map->strides[1] * j;
}

template <typename T>
static float GetPathCost(const struct PathCostArray *map, int i, int j) {
  return (float)(((T*)PickArrayValue(map, i, j))[0]);
}

float PathCostArrayFloat32(int x1, int y1, int x2, int y2,
                           const struct PathCostArray *map){
  return GetPathCost<float>(map, x2, y2);
}

float PathCostArrayInt8(int x1, int y1, int x2, int y2,
                        const struct PathCostArray *map){
  return GetPathCost<int8_t>(map, x2, y2);
}

float PathCostArrayInt16(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map){
  return GetPathCost<int16_t>(map, x2, y2);
}

float PathCostArrayInt32(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map){
  return GetPathCost<int32_t>(map, x2, y2);
}

float PathCostArrayUInt8(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map){
  return GetPathCost<uint8_t>(map, x2, y2);
}

float PathCostArrayUInt16(int x1, int y1, int x2, int y2,
                          const struct PathCostArray *map){
  return GetPathCost<uint16_t>(map, x2, y2);
}

float PathCostArrayUInt32(int x1, int y1, int x2, int y2,
                          const struct PathCostArray *map){
  return GetPathCost<uint32_t>(map, x2, y2);
}
