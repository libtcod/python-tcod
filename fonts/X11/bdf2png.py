#!/usr/bin/python2
"""
    This script canverts bdf files into png unicode tilesets for use with
    programs such as libtcod or python-tdl.
    
    Requires scipy, numpy, and PIL.
"""

from __future__ import division

import sys
import os

import re
import math
import itertools
import glob

import scipy.misc
try:
    scipy.misc.imsave
except AttributeError:
    raise SystemExit('Must have python PIL installed')
import numpy

class Glyph:

    def __init__(self, data, bbox):
        "Make a new glyph with the data between STARTCHAR and ENDCHAR"
        # get character index
        self.encoding = int(re.search('ENCODING ([0-9-]+)', data).groups()[0])
        if self.encoding < 0:
            # I ran into a -1 encoding once, not sure what to do with it
            self.encoding += 65536 # just put it at the end I guess
        
        # get local bbox
        match = re.search('\nBBX ([0-9-]+) ([0-9-]+) ([0-9-]+) ([0-9-]+)', data)
        if match:
            gbbox = [int(i) for i in match.groups()]
        else:
            gbbox = bbox
        self.font_bbox = bbox
        self.bbox = gbbox
        self.width, self.height = self.bbox[:2]
        
        # get bitmap
        match = re.search('\nBITMAP\n([0-9A-F\n]*)', data)
        self.bitmap = numpy.empty([self.height, self.width], bool)
        if self.height == self.width == 0:
            return
        for y,hexcode in enumerate(match.groups()[0].split('\n')):
            for x, bit in self.parseBits(hexcode, self.width):
                self.bitmap[y,x] = bit
        
    def blit(self, image, x, y):
        """blit to the image array"""
        # adjust the position with the local bbox
        x += self.font_bbox[2] - self.bbox[2]
        y += self.font_bbox[3] - self.bbox[3]
        x += self.font_bbox[0] - self.bbox[0]
        y += self.font_bbox[1] - self.bbox[1]
        image[y:y+self.height, x:x+self.width] = self.bitmap * 255
            
    def parseBits(self, hexcode, width):
        """enumerate over bits in a line of data"""
        bitarray = []
        for byte in hexcode[::-1]:
            bits = int(byte, 16)
            for x in range(4):
                bitarray.append(bool((2 ** x) & bits))
        bitarray = bitarray[::-1]
        return enumerate(bitarray[:width])
                

def convert(filename):
    print('Converting %s...' % filename)
    bdf = open(filename, 'r').read()
    
    # name the output file
    outfile = os.path.basename(filename)
    if '.' in outfile:
        outfile = outfile.rsplit('.', 1)[0] + '.png'
    
    # print out comments
    for comment in re.findall('\nCOMMENT (.*)', bdf):
        print(comment)
    # and copyright
    match = re.search('\n(COPYRIGHT ".*")', bdf)
    if match:
        print(match.groups()[0])
        
    # get bounding box
    match = re.search('\nFONTBOUNDINGBOX ([0-9-]+) ([0-9-]+) ([0-9-]+) ([0-9-]+)', bdf)
    bbox = [int(i) for i in match.groups()]
    fontWidth, fontHeight, fontOffsetX, fontOffsetY = bbox
    print('Font size: %ix%i' % (fontWidth, fontHeight))
    print('Font offset: %i,%i' % (fontOffsetX, fontOffsetY))
    
    # generate glyphs
    glyphs = []
    for glyph in re.findall('\nSTARTCHAR [^\n]*\n(.*?)\nENDCHAR', bdf, re.DOTALL):
        glyphs.append(Glyph(glyph, bbox))
    print('Found %i glyphs' % len(glyphs))
    
    # start rendering to an array
    imgColumns = 64
    imgRows = 65536 // imgColumns
    imgWidth = imgColumns * fontWidth
    imgHeight = imgRows * fontHeight
    image = numpy.zeros([imgHeight, imgWidth], 'u1')
    for glyph in glyphs:
        y, x = divmod(glyph.encoding, imgColumns)
        x, y = x * fontWidth, y * fontHeight
        glyph.blit(image, x, y)
    
    # save as png
    
    #rgba = numpy.empty([imgHeight, imgWidth, 4])
    #rgba[...,...,0] = image
    #rgba[...,...,1] = image
    #rgba[...,...,2] = image
    #rgba[...,...,:3] = 255
    #rgba[...,...,3] = image
    #scipy.misc.imsave(outfile, rgba)
    
    scipy.misc.imsave(outfile, image)
    print('Saved as %s' % outfile)
    
    
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise SystemExit('Usage: %s [FILE]...' % sys.argv[0])
    for globs in (glob.iglob(arg) for arg in sys.argv[1:]):
        for filename in globs:
            convert(filename)
