#include "noise.h"

#include <libtcod.h>

float NoiseGetSample(TDLNoise *noise, float *xyzw) {
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
void NoiseSampleMeshGrid(
    TDLNoise *noise, const long len, const float *in, float *out) {
# pragma omp parallel
  {
    long i;
#   pragma omp for schedule(static)
    for (i = 0; i < len; ++i) {
      int axis;
      float xyzw[4];
      for (axis = 0; axis < noise->dimensions; ++axis) {
        xyzw[axis] = in[axis * len + i];
      }
      out[i] = NoiseGetSample(noise, xyzw);
    }
  }
}
static long GetSizeFromShape(const int ndim, const long *shape){
  long size=1;
  long i;
  for (i = 0; i < ndim; ++i){
    size *= shape[i];
  }
  return size;
}
static float GetOpenMeshGridValue(
    TDLNoise *noise, const int ndim, const long *shape,
    const float **ogrid_in, const long index) {
  int axis;
  long xyzw_indexes[4];
  float xyzw_values[4];
  /* Convert index -> xyzw_indexes -> xyzw_values */
  xyzw_indexes[0] = index;
  for (axis = 0; axis < ndim - 1; ++axis){
    xyzw_indexes[axis + 1] = xyzw_indexes[axis] / shape[axis];
    xyzw_values[axis] = ogrid_in[axis][xyzw_indexes[axis] % shape[axis]];
  }
  xyzw_values[ndim - 1] = ogrid_in[axis][xyzw_indexes[ndim - 1]];
  return NoiseGetSample(noise, xyzw_values);
}
void NoiseSampleOpenMeshGrid(TDLNoise *noise, const int ndim_in,
                             const long *shape,
                             const float **ogrid_in, float *out) {
# pragma omp parallel
  {
    long i;
    long len=GetSizeFromShape(ndim_in, shape);
#   pragma omp for schedule(static)
    for (i = 0; i < len; ++i){
      out[i] = GetOpenMeshGridValue(noise, ndim_in, shape, ogrid_in, i);
    }
  }
}
