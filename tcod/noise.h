#ifndef PYTHON_TCOD_NOISE_H_
#define PYTHON_TCOD_NOISE_H_

#include "../libtcod/src/libtcod/noise.h"
#include "../libtcod/src/libtcod/noise_defaults.h"
#include "../libtcod/src/libtcod/mersenne.h"

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
float NoiseGetSample(TDLNoise *noise, float *xyzw);

/* Fill `out` with samples derived from the  mesh-grid `in`. */
void NoiseSampleMeshGrid(
    TDLNoise *noise, const long len, const float *in, float *out);

/* Fill `out` with samples derived from the open mesh-grid `in`. */
void NoiseSampleOpenMeshGrid(TDLNoise *noise,
                             const int ndim, const long *shape,
                             const float **ogrid_in, float *out);

#endif /* PYTHON_TCOD_NOISE_H_ */
