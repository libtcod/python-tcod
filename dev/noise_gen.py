#!/usr/bin/env python

# allow a noise test
# give a directory and this will save simple bitmaps of each noise example
import sys, os

sys.path.insert(0, '..')
from tdl.noise import Noise

IMGSIZE = 128 # width and height of the saved image
NOISE_SCALE = 1 / 12
saveDir = ''
kargs = {}
if len(sys.argv) >= 2:
    saveDir = sys.argv[1] # get save directory
    kargs = dict((param.split('=') for param in sys.argv[2:])) # get parameters
if not saveDir:
    raise SystemExit('Provide a directory to save the noise examples.')
for algo in ['PERLIN', 'SIMPLEX', 'WAVELET']:
    for mode in ['FLAT', 'FBM', 'TURBULENCE']:
        noise = Noise(algo, mode)
        noiseFile = open(os.path.join(saveDir, '%s_%s.pgm' % (algo, mode)), 'wb')
        print('Generating %s' % noiseFile.name)
        # make a greyscale Netpbm file
        noiseFile.write(b'P5\n')
        noiseFile.write(('%i %i\n' % (IMGSIZE, IMGSIZE)).encode('ascii'))
        noiseFile.write(b'255\n')
        for y in range(IMGSIZE):
            noiseY = y * NOISE_SCALE
            for x in range(IMGSIZE):
                noiseX = x * NOISE_SCALE
                if x == 0 or x == IMGSIZE - 1 or y == 0 or y == IMGSIZE - 1:
                    val = 0 # use black border
                else:
                    val = int(noise.getPoint(noiseX, noiseY) * 255)
                noiseFile.write(bytes((val,)))
        