#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys
import os.path
import unittest

from SDL import *

class SDL_Video_TestCase(unittest.TestCase):
    def setUp(self):
        SDL_Init(SDL_INIT_VIDEO)

class SDL_Video_MapTest(SDL_Video_TestCase):
    def testMapRGB(self):
        info = SDL_GetVideoInfo()
        pf = info.vfmt.contents
        for t in [(0, 0, 0),
                  (255, 0, 0),
                  (0xde, 0xea, 0xbe)]:
            opaque = SDL_MapRGB(pf, *t)
            assert t == SDL_GetRGB(opaque, pf)

    def testMapRGBA(self):
        # TODO init a surface w/ alpha so can test this properly
        info = SDL_GetVideoInfo()
        pf = info.vfmt.contents
        for t in [(0, 0, 0, 255),
                  (255, 0, 0, 255),
                  (255, 255, 255, 255),
                  (0xde, 0xea, 0xbe, 255)]:
            opaque = SDL_MapRGBA(pf, *t)
            assert t == SDL_GetRGBA(opaque, pf)

if __name__ == '__main__':
    unittest.main()
