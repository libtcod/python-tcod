#!/usr/bin/env python
"""
    Sample pack showcasing the abilities of TDL
    
    This example is not finished quite yet
"""

import sys

import random

sys.path.insert(0, '../')
import tdl


class SampleApp(tdl.event.App):
    name = ''
    
    def key_UP(self, event):
        global sampleIndex
        sampleIndex = (sampleIndex - 1) % len(samples)
    
    def key_DOWN(self, event):
        global sampleIndex
        sampleIndex = (sampleIndex - 1) % len(samples)
    
class TrueColorSample(SampleApp):
    name = 'True Colors'
    
    def update(self, deltaTime):
        width, height = samplewin.getSize()
        for x in range(width):
            for y in range(height):
                char = random.getrandbits(8)
                samplewin.drawChar(x, y, char, (255, 255, 255), (0, 0, 0))

class NoiseSample(SampleApp):
    name = 'Noise'
    SPEED = 3
    
    NOISE_KEYS = {'1': 'PERLIN', '2': 'SIMPLEX', '3': 'WAVELET'}
    MODE_KEYS = {'4': 'FLAT', '5': 'FBM', '6': 'TURBULENCE'}
    
    def __init__(self):
        self.noiseType = 'PERLIN'
        self.noiseMode = 'FLAT'
        self.x = 0
        self.y = 0
        self.z = 0
        self.zoom = 4
        self.generateNoise()
    
    def generateNoise(self):
        self.noise = tdl.noise.Noise(self.noiseType, self.noiseMode, seed=42)
        
    def key_CHAR(self, event):
        if event.char in self.NOISE_KEYS:
            self.noiseType = self.NOISE_KEYS[event.char]
        if event.char in self.MODE_KEYS:
            self.noiseMode = self.MODE_KEYS[event.char]
        self.generateNoise()
    
    def update(self, deltaTime):
        self.x += self.SPEED * deltaTime# * self.zoom
        self.y += self.SPEED * deltaTime# * self.zoom
        self.z += deltaTime / 4
        
        width, height = samplewin.getSize()
        for x in range(width):
            for y in range(height):
                val = self.noise.getPoint((x + self.x) / width * self.zoom,
                                          (y + self.y) / height * self.zoom,
                                          self.z)
                bgcolor = (int(val * 255),) * 2 + (min(255, int(val * 2 * 255)),)
                samplewin.drawChar(x, y, ' ', (255, 255, 255), bgcolor)

WIDTH, HEIGHT = 80, 50
SAMPLE_WINDOW_RECT = (20, 10, 46, 20)
    
if __name__ == '__main__':
    console = tdl.init(WIDTH, HEIGHT)
    samplewin = tdl.Window(console, *SAMPLE_WINDOW_RECT)
    
    samples = [cls() for cls in [TrueColorSample, NoiseSample]]
    sampleIndex = 0
    
    while 1:
        console.clear()
        samples[sampleIndex].runOnce()
        for i, sample in enumerate(samples):
            bgcolor = (0, 0, 0)
            if sampleIndex == i:
                bgcolor = (0, 0, 192)
            console.drawStr(0, -5 + i, '%s' % sample.name, (255, 255, 255), bgcolor)
        console.drawStr(0, -1, '%i FPS' % tdl.getFPS())
        tdl.flush()
