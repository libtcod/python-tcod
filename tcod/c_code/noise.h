
#include <libtcod.h>

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
