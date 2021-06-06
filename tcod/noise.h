#ifndef PYTHON_TCOD_NOISE_H_
#define PYTHON_TCOD_NOISE_H_

#include "../libtcod/src/libtcod/mersenne.h"
#include "../libtcod/src/libtcod/noise.h"
#include "../libtcod/src/libtcod/noise_defaults.h"

typedef enum NoiseImplementation {
  kNoiseImplementationSimple,
  kNoiseImplementationFBM,
  kNoiseImplementationTurbulence,
} NoiseImplementation;

typedef struct TDLNoise {
  TCOD_noise_t noise;
  int dimensions;
  NoiseImplementation implementation;
  float octaves;
} TDLNoise;

/* Return a single sample from noise. */
float NoiseGetSample(TDLNoise* noise, float* __restrict xyzw);

/* Fill `out` with samples derived from the  mesh-grid `in`. */
void NoiseSampleMeshGrid(TDLNoise* noise, const long len, const float* __restrict in, float* __restrict out);

/* Fill `out` with samples derived from the open mesh-grid `in`. */
void NoiseSampleOpenMeshGrid(
    TDLNoise* __restrict noise,
    const int ndim,
    const long* __restrict shape,
    const float* __restrict* __restrict ogrid_in,
    float* __restrict out);

#endif /* PYTHON_TCOD_NOISE_H_ */
