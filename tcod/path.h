#ifndef PYTHON_TCOD_PATH_H_
#define PYTHON_TCOD_PATH_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 *  Common NumPy data types.
 */
enum NP_Type {
  np_undefined = 0,
  np_int8,
  np_int16,
  np_int32,
  np_int64,
  np_uint8,
  np_uint16,
  np_uint32,
  np_uint64,
  np_float16,
  np_float32,
  np_float64,
};
/**
 *  A simple 4D NumPy array ctype.
 */
struct NArray4 {
  enum NP_Type type;
  char *data;
  ptrdiff_t shape[4];
  ptrdiff_t strides[4];
};

struct EdgeRule {
  int vector[4];
  int cost;
  struct NArray4 dest_cost;
  struct NArray4 staging;
};

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
