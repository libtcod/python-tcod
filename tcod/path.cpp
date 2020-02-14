#include "path.h"

#include <cstdint>
#include <exception>

#include <SDL_stdinc.h>
#include "../libtcod/src/libtcod/error.h"
#include "../libtcod/src/libtcod/pathfinding/generic.h"
#include "../libtcod/src/libtcod/pathfinding/dijkstra.h"
#include "../libtcod/src/libtcod/pathfinding/hill-climb.h"
#include "../libtcod/src/libtcod/utility/matrix.h"

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

template <typename T>
tcod::MatrixView<T, 2> to_matrix(struct NArray4* array)
{
  return tcod::MatrixView<T, 2>(
      reinterpret_cast<T*>(array->data),
      {array->shape[0], array->shape[1]},
      {array->strides[0], array->strides[1]}
  );
}
template <typename T>
const tcod::MatrixView<T, 2> to_matrix(const struct NArray4* array)
{
  return tcod::MatrixView<T, 2>(
      reinterpret_cast<T*>(array->data),
      {array->shape[0], array->shape[1]},
      {array->strides[0], array->strides[1]}
  );
}

template <typename DistType, typename CostType>
int dijkstra2d_3(
    DistType& dist,
    const CostType& cost,
    int cardinal,
    int diagonal)
{
  try {
    tcod::pathfinding::dijkstra2d(dist, cost, cardinal, diagonal);
  } catch (const std::exception& e) {
    return tcod::set_error(e);
  }
  return 0;
}
template <typename DistType>
int dijkstra2d_2(
    DistType& dist,
    const struct NArray4* cost,
    int cardinal,
    int diagonal)
{
  switch (cost->type) {
    case np_int8:
      return dijkstra2d_3(dist, to_matrix<int8_t>(cost), cardinal, diagonal);
    case np_int16:
      return dijkstra2d_3(dist, to_matrix<int16_t>(cost), cardinal, diagonal);
    case np_int32:
      return dijkstra2d_3(dist, to_matrix<int32_t>(cost), cardinal, diagonal);
    case np_int64:
      return dijkstra2d_3(dist, to_matrix<int64_t>(cost), cardinal, diagonal);
    case np_uint8:
      return dijkstra2d_3(dist, to_matrix<uint8_t>(cost), cardinal, diagonal);
    case np_uint16:
      return dijkstra2d_3(dist, to_matrix<uint16_t>(cost), cardinal, diagonal);
    case np_uint32:
      return dijkstra2d_3(dist, to_matrix<uint32_t>(cost), cardinal, diagonal);
    case np_uint64:
      return dijkstra2d_3(dist, to_matrix<uint64_t>(cost), cardinal, diagonal);
    default:
      return tcod::set_error("Expected distance map to be int type.");
  }
}
int dijkstra2d(
    struct NArray4* dist,
    const struct NArray4* cost,
    int cardinal,
    int diagonal)
{
  switch (dist->type) {
    case np_int8: {
      auto dist_ = to_matrix<int8_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_int16: {
      auto dist_ = to_matrix<int16_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_int32: {
      auto dist_ = to_matrix<int32_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_int64: {
      auto dist_ = to_matrix<int64_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_uint8: {
      auto dist_ = to_matrix<uint8_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_uint16: {
      auto dist_ = to_matrix<uint16_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_uint32: {
      auto dist_ = to_matrix<uint32_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    case np_uint64: {
      auto dist_ = to_matrix<uint64_t>(dist);
      return dijkstra2d_2(dist_, cost, cardinal, diagonal);
    }
    default:
      return tcod::set_error("Expected cost map to be int type.");
  }
}

static const std::array<std::array<int, 2>, 4> CARDINAL_{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
static const std::array<std::array<int, 2>, 4> DIAGONAL_{{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}}};

template <typename IndexType>
class PlainGraph {
 public:
  PlainGraph(const IndexType& size, bool cardinal, bool diagonal)
  : size_{size}, cardinal_{cardinal}, diagonal_{diagonal}
  {}
  template <typename F, typename index_type>
  void with_edges(const F& edge_func, const index_type& index) const
  {
    if (cardinal_) {
      for (const auto& edge : CARDINAL_) {
        index_type node{index[0] + edge[0], index[1] + edge[1]};
        if (!in_range(node)) { continue; }
        edge_func(node, 0);
      }
    }
    if (diagonal_) {
      for (const auto& edge : DIAGONAL_) {
        index_type node{index[0] + edge[0], index[1] + edge[1]};
        if (!in_range(node)) { continue; }
        edge_func(node, 0);
      }
    }
  }
 private:
  IndexType size_;
  bool cardinal_;
  bool diagonal_;
  bool in_range(const IndexType& index) const noexcept{
    return (0 <= index[0] && index[0] < size_[0] &&
            0 <= index[1] && index[1] < size_[1]);
  }
};

template <typename DistType>
int hillclimb2d_2(
    const DistType& dist,
    int x,
    int y,
    bool cardinal,
    bool diagonal,
    int* out)
{
  PlainGraph<typename DistType::shape_type> graph{
      dist.get_shape(), cardinal, diagonal
  };
  auto path = tcod::pathfinding::simple_hillclimb(dist, graph, {x, y});
  if (out) {
    for (const auto& index : path) {
      out[0] = static_cast<int>(index.at(0));
      out[1] = static_cast<int>(index.at(1));
      out += 2;
    }
  }
  return static_cast<int>(path.size());
}
int hillclimb2d(
    const struct NArray4* dist,
    int x,
    int y,
    bool cardinal,
    bool diagonal,
    int* out)
{
  switch (dist->type) {
    case np_int8:
      return hillclimb2d_2(to_matrix<int8_t>(dist), x, y, cardinal, diagonal, out);
    case np_int16:
      return hillclimb2d_2(to_matrix<int16_t>(dist), x, y, cardinal, diagonal, out);
    case np_int32:
      return hillclimb2d_2(to_matrix<int32_t>(dist), x, y, cardinal, diagonal, out);
    case np_int64:
      return hillclimb2d_2(to_matrix<int64_t>(dist), x, y, cardinal, diagonal, out);
    case np_uint8:
      return hillclimb2d_2(to_matrix<uint8_t>(dist), x, y, cardinal, diagonal, out);
    case np_uint16:
      return hillclimb2d_2(to_matrix<uint16_t>(dist), x, y, cardinal, diagonal, out);
    case np_uint32:
      return hillclimb2d_2(to_matrix<uint32_t>(dist), x, y, cardinal, diagonal, out);
    case np_uint64:
      return hillclimb2d_2(to_matrix<uint64_t>(dist), x, y, cardinal, diagonal, out);
    default:
      return tcod::set_error("Expected cost map to be int type.");
  }

}
