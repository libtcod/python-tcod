
#include <noise.h>
#include <noise_defaults.h>
#include <mersenne.h>

/* Copied from libtcod's noise.c, needs to be kept up-to-date! */
typedef struct {
	int ndim;
	unsigned char map[256]; /* Randomized map of indexes into buffer */
	float buffer[256][TCOD_NOISE_MAX_DIMENSIONS]; 	/* Random 256 x ndim buffer */
	/* fractal stuff */
	float H;
	float lacunarity;
	float exponent[TCOD_NOISE_MAX_OCTAVES];
	float *waveletTileData;
	TCOD_random_t rand;
	/* noise type */
	TCOD_noise_type_t noise_type;
} perlin_data_t;

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
