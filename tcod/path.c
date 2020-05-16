#include "path.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include "../libtcod/src/libtcod/error.h"
#include "../libtcod/src/libtcod/pathfinder_frontier.h"

static void* pick_array_pointer(const struct PathCostArray *map, int i, int j)
{
  return (void*)(map->array + map->strides[0] * i + map->strides[1] * j);
}
float PathCostArrayFloat32(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return *(float*)pick_array_pointer(map, x2, y2);
}
float PathCostArrayInt8(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return *(int8_t*)pick_array_pointer(map, x2, y2);
}
float PathCostArrayInt16(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return *(int16_t*)pick_array_pointer(map, x2, y2);
}
float PathCostArrayInt32(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return (float)(*(int32_t*)pick_array_pointer(map, x2, y2));
}
float PathCostArrayUInt8(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return *(uint8_t*)pick_array_pointer(map, x2, y2);
}
float PathCostArrayUInt16(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return *(uint16_t*)pick_array_pointer(map, x2, y2);
}
float PathCostArrayUInt32(
    int x1, int y1, int x2, int y2, const struct PathCostArray *map)
{
  return (float)(*(uint32_t*)pick_array_pointer(map, x2, y2));
}

static bool array_in_range(const struct NArray* arr, int n, const int* index)
{
  for (int i = 0; i < n; ++i) {
    if (index[i] < 0 || index[i] >= arr->shape[i]) { return 0; }
  }
  return 1;
}
static bool array2d_in_range(const struct NArray* arr, int i, int j)
{
  return 0 <= i && i < arr->shape[0] && 0 <= j && j < arr->shape[1];
}
static void* get_array_ptr(const struct NArray* arr, int n, const int* index)
{
  unsigned char* ptr = (unsigned char*)arr->data;
  for (int i = 0; i < n; ++i) {
    ptr += arr->strides[i] * index[i];
  }
  return (void*)ptr;
}
static int64_t get_array_int64(const struct NArray* arr, int n, const int* index)
{
  const void* ptr = get_array_ptr(arr, n, index);
  switch (arr->type) {
    case np_int8:
      return *(const int8_t*)ptr;
    case np_int16:
      return *(const int16_t*)ptr;
    case np_int32:
      return *(const int32_t*)ptr;
    case np_int64:
      return *(const int64_t*)ptr;
    case np_uint8:
      return *(const uint8_t*)ptr;
    case np_uint16:
      return *(const uint16_t*)ptr;
    case np_uint32:
      return *(const uint32_t*)ptr;
    case np_uint64:
      return *(const uint64_t*)ptr;
    default:
      return 0;
  }
}
static int get_array_int(const struct NArray* arr, int n, const int* index)
{
  return (int)get_array_int64(arr, n, index);
}
static void set_array_int64(
    struct NArray* arr, int n, const int* index, int64_t value)
{
  void* ptr = get_array_ptr(arr, n, index);
  switch (arr->type) {
    case np_int8:
      *(int8_t*)ptr = (int8_t)value;
    case np_int16:
      *(int16_t*)ptr = (int16_t)value;
      return;
    case np_int32:
      *(int32_t*)ptr = (int32_t)value;
      return;
    case np_int64:
      *(int64_t*)ptr = value;
      return;
    case np_uint8:
      *(uint8_t*)ptr = (uint8_t)value;
      return;
    case np_uint16:
      *(uint16_t*)ptr = (uint16_t)value;
      return;
    case np_uint32:
      *(uint32_t*)ptr = (uint32_t)value;
      return;
    case np_uint64:
      *(uint64_t*)ptr = (uint64_t)value;
      return;
    default:
      return;
  }
}
static void set_array2d_int64(struct NArray* arr, int i, int j, int64_t value)
{
  int index[2] = {i, j};
  set_array_int64(arr, 2, index, value);
}
static void set_array_int(struct NArray* arr, int n, const int* index, int value)
{
  set_array_int64(arr, n, index, value);
}
static void set_array2d_int(struct NArray* arr, int i, int j, int value)
{
  set_array2d_int64(arr, i, j, value);
}
static int64_t get_array_is_max(const struct NArray* arr, int n, const int* index)
{
  const void* ptr = get_array_ptr(arr, n, index);
  switch (arr->type) {
    case np_int8:
      return *(const int8_t*)ptr == SCHAR_MAX;
    case np_int16:
      return *(const int16_t*)ptr == SHRT_MAX;
    case np_int32:
      return *(const int32_t*)ptr == INT_MAX;
    case np_int64:
      return *(const int64_t*)ptr == LONG_MAX;
    case np_uint8:
      return *(const uint8_t*)ptr == UCHAR_MAX;
    case np_uint16:
      return *(const uint16_t*)ptr == USHRT_MAX;
    case np_uint32:
      return *(const uint32_t*)ptr == UINT_MAX;
    case np_uint64:
      return *(const uint64_t*)ptr == ULONG_MAX;
    default:
      return 0;
  }
}

static const int CARDINAL_[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
static const int DIAGONAL_[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};

static void dijkstra2d_add_edge(
    struct TCOD_Frontier* frontier,
    struct NArray* dist_array,
    const struct NArray* cost,
    int edge_cost,
    const int dir[2])
{
  const int index[2] = {
      frontier->active_index[0] + dir[0], frontier->active_index[1] + dir[1]
  };
  if (!array_in_range(dist_array, 2, index)) { return; }
  edge_cost *= get_array_int(cost, 2, index);
  if (edge_cost <= 0) { return; }
  int distance = frontier->active_dist + edge_cost;
  if (get_array_int(dist_array, 2, index) <= distance) { return; }
  set_array_int(dist_array, 2, index, distance);
  TCOD_frontier_push(frontier, index, distance, distance);
}

int dijkstra2d(
    struct NArray* dist_array,
    const struct NArray* cost,
    int edges_2d_n,
    const int* edges_2d)
{
  struct TCOD_Frontier* frontier = TCOD_frontier_new(2);
  if (!frontier) { return TCOD_E_ERROR; }
  for (int i = 0; i < dist_array->shape[0]; ++i) {
    for (int j = 0; j < dist_array->shape[1]; ++j) {
      const int index[2] = {i, j};
      if (get_array_is_max(dist_array, 2, index)) { continue; }
      int dist = get_array_int(dist_array, 2, index);
      TCOD_frontier_push(frontier, index, dist, dist);
    }
  }
  while (TCOD_frontier_size(frontier)) {
    TCOD_frontier_pop(frontier);
    int distance_here = get_array_int(dist_array, 2, frontier->active_index);
    if (frontier->active_dist != distance_here) { continue; }
    for (int i = 0; i < edges_2d_n; ++i) {
      dijkstra2d_add_edge(
          frontier, dist_array, cost, edges_2d[i * 3 + 2], &edges_2d[i * 3]);
    }
  }
  return TCOD_E_OK;
}

int dijkstra2d_basic(
    struct NArray* dist_array,
    const struct NArray* cost,
    int cardinal,
    int diagonal)
{
  struct TCOD_Frontier* frontier = TCOD_frontier_new(2);
  if (!frontier) { return TCOD_E_ERROR; }
  for (int i = 0; i < dist_array->shape[0]; ++i) {
    for (int j = 0; j < dist_array->shape[1]; ++j) {
      const int index[2] = {i, j};
      if (get_array_is_max(dist_array, 2, index)) { continue; }
      int dist = get_array_int(dist_array, 2, index);
      TCOD_frontier_push(frontier, index, dist, dist);
    }
  }
  while (TCOD_frontier_size(frontier)) {
    TCOD_frontier_pop(frontier);
    int distance_here = get_array_int(dist_array, 2, frontier->active_index);
    if (frontier->active_dist != distance_here) { continue; }
    if (cardinal > 0) {
      dijkstra2d_add_edge(frontier, dist_array, cost, cardinal, CARDINAL_[0]);
      dijkstra2d_add_edge(frontier, dist_array, cost, cardinal, CARDINAL_[1]);
      dijkstra2d_add_edge(frontier, dist_array, cost, cardinal, CARDINAL_[2]);
      dijkstra2d_add_edge(frontier, dist_array, cost, cardinal, CARDINAL_[3]);
    }
    if (diagonal > 0) {
      dijkstra2d_add_edge(frontier, dist_array, cost, diagonal, DIAGONAL_[0]);
      dijkstra2d_add_edge(frontier, dist_array, cost, diagonal, DIAGONAL_[1]);
      dijkstra2d_add_edge(frontier, dist_array, cost, diagonal, DIAGONAL_[2]);
      dijkstra2d_add_edge(frontier, dist_array, cost, diagonal, DIAGONAL_[3]);
    }
  }
  return TCOD_E_OK;
}
static void hillclimb2d_check_edge(
    const struct NArray* dist_array,
    int* distance_in_out,
    const int origin[2],
    const int dir[2],
    int index_out[2])
{
  const int next[2] = {origin[0] + dir[0], origin[1] + dir[1]};
  if (!array_in_range(dist_array, 2, next)) { return; }
  const int next_distance = get_array_int(dist_array, 2, next);
  if (next_distance < *distance_in_out) {
    *distance_in_out = next_distance;
    index_out[0] = next[0];
    index_out[1] = next[1];
  }
}
int hillclimb2d(
    const struct NArray* dist_array,
    int start_i,
    int start_j,
    int edges_2d_n,
    const int* edges_2d,
    int* out)
{
  int next[2] = {start_i, start_j};
  int old_dist = get_array_int(dist_array, 2, next);
  int new_dist = old_dist;
  int length = 0;
  while (1) {
    ++length;
    if (out) {
      out[0] = next[0];
      out[1] = next[1];
      out += 2;
    }
    const int origin[2] = {next[0], next[1]};
    for (int i = 0; i < edges_2d_n; ++i) {
      hillclimb2d_check_edge(
          dist_array, &new_dist, origin, &edges_2d[i * 2], next);
    }
    if (old_dist == new_dist) {
      return length;
    }
    old_dist = new_dist;
  }
}
int hillclimb2d_basic(
    const struct NArray* dist_array,
    int start_i,
    int start_j,
    bool cardinal,
    bool diagonal,
    int* out)
{
  int next[2] = {start_i, start_j};
  int old_dist = get_array_int(dist_array, 2, next);
  int new_dist = old_dist;
  int length = 0;
  while (1) {
    ++length;
    if (out) {
      out[0] = next[0];
      out[1] = next[1];
      out += 2;
    }
    const int origin[2] = {next[0], next[1]};
    if (cardinal) {
      hillclimb2d_check_edge(dist_array, &new_dist, origin, CARDINAL_[0], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, CARDINAL_[1], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, CARDINAL_[2], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, CARDINAL_[3], next);
    }
    if (diagonal) {
      hillclimb2d_check_edge(dist_array, &new_dist, origin, DIAGONAL_[0], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, DIAGONAL_[1], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, DIAGONAL_[2], next);
      hillclimb2d_check_edge(dist_array, &new_dist, origin, DIAGONAL_[3], next);
    }
    if (old_dist == new_dist) {
      return length;
    }
    old_dist = new_dist;
  }
}
void path_compute_add_edge(
    struct TCOD_Frontier* frontier,
    struct NArray* dist_map,
    struct NArray* travel_map,
    const struct NArray* cost_map,
    const int* edge_rule)
{
  int dest[TCOD_PATHFINDER_MAX_DIMENSIONS];
  for (int i = 0; i < frontier->ndim; ++i) {
    dest[i] = frontier->active_index[i] + edge_rule[i];
  }
  if (!array_in_range(dist_map, frontier->ndim, dest)) { return; }
  int edge_cost = edge_rule[frontier->ndim];
  edge_cost *= get_array_int(cost_map, frontier->ndim, dest);
  if (edge_cost <= 0) { return; }
  int distance = frontier->active_dist + edge_cost;
  if (get_array_int(dist_map, frontier->ndim, dest) <= distance) { return; }
  set_array_int(dist_map, frontier->ndim, dest, distance);
  int* path = get_array_ptr(travel_map, frontier->ndim, dest);
  for (int i = 0; i < frontier->ndim; ++i) {
    path[i] = frontier->active_index[i];
  }
  TCOD_frontier_push(frontier, dest, distance, distance);
}

int path_compute_step(
    struct TCOD_Frontier* frontier,
    struct NArray* dist_map,
    struct NArray* travel_map,
    int n,
    const struct PathfinderRule* rules)
{
  if (!frontier) {
    return TCOD_set_errorv("Missing frontier.");
  }
  if (frontier->ndim <= 0 || frontier->ndim > TCOD_PATHFINDER_MAX_DIMENSIONS) {
    return TCOD_set_errorv("Invalid frontier->ndim.");
  }
  if (!dist_map) {
    return TCOD_set_errorv("Missing dist_map.");
  }
  if (frontier->ndim != dist_map->ndim) {
    return TCOD_set_errorv("Invalid or corrupt input.");
  }
  if (travel_map && frontier->ndim + 1 != travel_map->ndim) {
    return TCOD_set_errorv("Invalid or corrupt input.");
  }
  TCOD_frontier_pop(frontier);
  for (int i = 0; i < n; ++i) {
    if (rules[i].condition.type) {
      if (!get_array_int(&rules[i].condition, frontier->ndim, frontier->active_index)) {
        continue;
      }
    }
    for (int edge_i = 0; edge_i < rules[i].edge_count; ++edge_i) {
      path_compute_add_edge(
          frontier,
          dist_map,
          travel_map,
          &rules[i].cost,
          &rules[i].edge_array[edge_i * (frontier->ndim + 1)]);
    }
  }
  return 0;
}
int path_compute(
    struct TCOD_Frontier* frontier,
    struct NArray* dist_map,
    struct NArray* travel_map,
    int n,
    const struct PathfinderRule* rules)
{
  if (!frontier) {
    return TCOD_set_errorv("Missing frontier.");
  }
  while (TCOD_frontier_size(frontier)) {
    int err = path_compute_step(
        frontier,
        dist_map,
        travel_map,
        n,
        rules);
    if (err < 0) { return err; }
  }
  return 0;
}
size_t get_travel_path(
    int8_t ndim, const struct NArray* travel_map, const int* start, int* out)
{
  if (ndim <= 0 || ndim > TCOD_PATHFINDER_MAX_DIMENSIONS) {
    return TCOD_set_errorv("Invalid ndim.");
  }
  if (!travel_map) {
    return TCOD_set_errorv("Missing travel_map.");
  }
  if (!start) {
    return TCOD_set_errorv("Missing start.");
  }
  if (ndim != travel_map->ndim - 1) {
    return TCOD_set_errorv("Invalid or corrupt input.");
  }
  const int* next = get_array_ptr(travel_map, ndim, start);
  const int* current = start;
  size_t max_loops = 1;
  size_t length = 0;
  for (int i = 0; i < ndim; ++i) { max_loops *= (size_t)travel_map->shape[i]; }
  while (current != next) {
    ++length;
    if (out) {
      for (int i = 0; i < ndim; ++i) { out[i] = current[i]; }
      out += ndim;
    }
    current = next;
    if (!array_in_range(travel_map, ndim, next)) {
      return TCOD_set_errorvf(
          "Index (%i, %i) is out of range.", next[0], next[1]);
    }
    next = get_array_ptr(travel_map, ndim, next);
    if (!out && length == max_loops) {
      return TCOD_set_errorv("Possible cyclic loop detected.");
    }
  }
  return length;
}