#ifndef PYTHON_TCOD_PATH_H_
#define PYTHON_TCOD_PATH_H_

#ifdef __cplusplus
extern "C" {
#endif

struct PathCostArray {
    char *array;
    long long strides[2];
};

float PathCostArrayFloat32(int x1, int y1, int x2, int y2,
                           const struct PathCostArray *map);

float PathCostArrayUInt8(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map);

float PathCostArrayUInt16(int x1, int y1, int x2, int y2,
                          const struct PathCostArray *map);

float PathCostArrayUInt32(int x1, int y1, int x2, int y2,
                          const struct PathCostArray *map);

float PathCostArrayInt8(int x1, int y1, int x2, int y2,
                        const struct PathCostArray *map);

float PathCostArrayInt16(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map);

float PathCostArrayInt32(int x1, int y1, int x2, int y2,
                         const struct PathCostArray *map);

#ifdef __cplusplus
}
#endif

#endif /* PYTHON_TCOD_PATH_H_ */
