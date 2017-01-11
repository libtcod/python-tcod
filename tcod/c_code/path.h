#ifndef _TCOD_C_CODE_PATH_H_
#define _TCOD_C_CODE_PATH_H_

typedef struct {
    int width;
    void *array;
} PathCostArray;

float PathCostArrayFloat32(const int x1, const int y1,
                           const int x2, const int y2,
                           const PathCostArray *map);

float PathCostArrayUInt8(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map);

float PathCostArrayUInt16(const int x1, const int y1,
                          const int x2, const int y2,
                          const PathCostArray *map);

float PathCostArrayUInt32(const int x1, const int y1,
                          const int x2, const int y2,
                          const PathCostArray *map);

float PathCostArrayInt8(const int x1, const int y1,
                        const int x2, const int y2,
                        const PathCostArray *map);

float PathCostArrayInt16(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map);

float PathCostArrayInt32(const int x1, const int y1,
                         const int x2, const int y2,
                         const PathCostArray *map);

#endif
