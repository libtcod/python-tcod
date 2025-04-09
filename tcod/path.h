#ifndef PYTHON_TCOD_PATH_H_
#define PYTHON_TCOD_PATH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../libtcod/src/libtcod/pathfinder_frontier.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 *  Common NumPy data types.
 */
typedef enum NP_Type {
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
} NP_Type;
/**
 *  A simple 4D NumPy array ctype.
 */
typedef struct NArray {
  NP_Type type;
  int8_t ndim;
  char* __restrict data;
  ptrdiff_t shape[5];    // TCOD_PATHFINDER_MAX_DIMENSIONS + 1
  ptrdiff_t strides[5];  // TCOD_PATHFINDER_MAX_DIMENSIONS + 1
} NArray;

struct PathfinderRule {
  /** Rule condition, could be uninitialized zeros. */
  NArray condition;
  /** Edge cost map, required. */
  NArray cost;
  /** Number of edge rules in `edge_array`. */
  int edge_count;
  /** Example of 2D edges: [i, j, cost, i_2, j_2, cost_2, ...] */
  int* __restrict edge_array;
};

struct PathfinderHeuristic {
  int cardinal;
  int diagonal;
  int z;
  int w;
  int target[TCOD_PATHFINDER_MAX_DIMENSIONS];
};

struct FrontierNode {
  int distance;
  int index[];
};

struct PathCostArray {
  char* __restrict array;
  long long strides[2];
};

float PathCostArrayFloat32(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayUInt8(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayUInt16(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayUInt32(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayInt8(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayInt16(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

float PathCostArrayInt32(int x1, int y1, int x2, int y2, const struct PathCostArray* map);

/**
    Return the value to add to the distance to sort nodes by A*.

    `heuristic` can be NULL.

    `index[ndim]` must not be NULL.
 */
int compute_heuristic(const struct PathfinderHeuristic* __restrict heuristic, int ndim, const int* __restrict index);
int dijkstra2d(
    struct NArray* __restrict dist,
    const struct NArray* __restrict cost,
    int edges_2d_n,
    const int* __restrict edges_2d);

int dijkstra2d_basic(struct NArray* __restrict dist, const struct NArray* __restrict cost, int cardinal, int diagonal);

int hillclimb2d(
    const struct NArray* __restrict dist_array,
    int start_i,
    int start_j,
    int edges_2d_n,
    const int* __restrict edges_2d,
    int* __restrict out);

int hillclimb2d_basic(
    const struct NArray* __restrict dist, int x, int y, bool cardinal, bool diagonal, int* __restrict out);

int path_compute_step(
    struct TCOD_Frontier* __restrict frontier,
    struct NArray* __restrict dist_map,
    struct NArray* __restrict travel_map,
    int n,
    const struct PathfinderRule* __restrict rules,  // rules[n]
    const struct PathfinderHeuristic* heuristic);

int path_compute(
    struct TCOD_Frontier* __restrict frontier,
    struct NArray* __restrict dist_map,
    struct NArray* __restrict travel_map,
    int n,
    const struct PathfinderRule* __restrict rules,  // rules[n]
    const struct PathfinderHeuristic* __restrict heuristic);
/**
    Find and get a path along `travel_map`.

    Returns the length of the path, `out` must be NULL or `out[n*ndim]`.
    Where `n` is the value return from a previous call with the same
    parameters.
 */
ptrdiff_t get_travel_path(
    int8_t ndim, const NArray* __restrict travel_map, const int* __restrict start, int* __restrict out);
/**
    Update the priority of nodes on the frontier and sort them.
 */
int update_frontier_heuristic(
    struct TCOD_Frontier* __restrict frontier, const struct PathfinderHeuristic* __restrict heuristic);
/**
    Update a frontier from a distance array.

    Assumes no heuristic is active.
 */
int rebuild_frontier_from_distance(struct TCOD_Frontier* __restrict frontier, const NArray* __restrict dist_map);
/**
    Return true if `index[frontier->ndim]` is a node in `frontier`.
 */
int frontier_has_index(const struct TCOD_Frontier* __restrict frontier, const int* __restrict index);
#ifdef __cplusplus
}
#endif

#endif /* PYTHON_TCOD_PATH_H_ */
