#include "noise.h"

#include "../libtcod/src/libtcod/libtcod.h"

float NoiseGetSample(TDLNoise* noise, float* __restrict xyzw) {
  switch (noise->implementation) {
    default:
    case kNoiseImplementationSimple:
      return TCOD_noise_get(noise->noise, xyzw);

    case kNoiseImplementationFBM:
      return TCOD_noise_get_fbm(noise->noise, xyzw, noise->octaves);

    case kNoiseImplementationTurbulence:
      return TCOD_noise_get_turbulence(noise->noise, xyzw, noise->octaves);
  }
}
void NoiseSampleMeshGrid(TDLNoise* noise, const long len, const float* __restrict in, float* __restrict out) {
  {
    long i;
    for (i = 0; i < len; ++i) {
      int axis;
      float xyzw[TCOD_NOISE_MAX_DIMENSIONS];
      for (axis = 0; axis < noise->dimensions; ++axis) { xyzw[axis] = in[axis * len + i]; }
      out[i] = NoiseGetSample(noise, xyzw);
    }
  }
}
static long GetSizeFromShape(const int ndim, const long* shape) {
  long size = 1;
  long i;
  for (i = 0; i < ndim; ++i) { size *= shape[i]; }
  return size;
}
static float GetOpenMeshGridValue(
    TDLNoise* __restrict noise,
    const int ndim,
    const long* __restrict shape,
    const float* __restrict* __restrict ogrid_in,
    const long index) {
  int axis = ndim - 1;
  long xyzw_indexes[TCOD_NOISE_MAX_DIMENSIONS];
  float xyzw_values[TCOD_NOISE_MAX_DIMENSIONS];
  /* Convert index -> xyzw_indexes -> xyzw_values */
  xyzw_indexes[axis] = index;
  xyzw_values[axis] = ogrid_in[axis][xyzw_indexes[axis] % shape[axis]];
  while (--axis >= 0) {
    xyzw_indexes[axis] = xyzw_indexes[axis + 1] / shape[axis + 1];
    xyzw_values[axis] = ogrid_in[axis][xyzw_indexes[axis] % shape[axis]];
  }
  return NoiseGetSample(noise, xyzw_values);
}
void NoiseSampleOpenMeshGrid(
    TDLNoise* __restrict noise,
    const int ndim_in,
    const long* __restrict shape,
    const float* __restrict* __restrict ogrid_in,
    float* __restrict out) {
  {
    long i;
    long len = GetSizeFromShape(ndim_in, shape);
    for (i = 0; i < len; ++i) { out[i] = GetOpenMeshGridValue(noise, ndim_in, shape, ogrid_in, i); }
  }
}
