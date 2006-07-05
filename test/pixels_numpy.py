#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import math
import random

from SDL import *
from numpy import *

class Wave:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.periodx = math.pi/2
        self.periody = math.pi/2
        self.dx = (random.random() - 0.5) * 0.2
        self.dy = (random.random() - 0.5) * 0.2
        self.x = (random.random() - 0.5) * self.w
        self.y = (random.random() - 0.5) * self.h
        self.color = [random.random() for i in range(3)]

    def update(self):
        '''Return a 2D numpy array of values in the range [0,1]'''
        self.x += self.dx
        self.y += self.dy
        cx = cos(arange(self.w) * (self.periodx / self.w) + self.x)
        cy = cos(arange(self.h) * (self.periody / self.h) + self.y)
        return multiply.outer(cy, cx)

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    SDL_WM_SetCaption('SDL-ctypes numpy demo', 'SDL-ctypes numpy demo')
    screen = SDL_SetVideoMode(160, 160, 32, SDL_SWSURFACE)
    format = screen.format

    waves = [Wave(screen.w, screen.h) for i in range(4)]

    while True:
        SDL_LockSurface(screen)
        pixels = screen.pixels.as_numpy()

        r = g = b = zeros((screen.h, screen.w))
        for w in waves:
            c = w.update()
            r = add(r, c * w.color[0])
            g = add(g, c * w.color[1])
            b = add(b, c * w.color[2])

        r = clip(r*0xff, 0, 0xff).astype(UInt32) << 16
        g = clip(g*0xff, 0, 0xff).astype(UInt32) << 8 
        b = clip(b*0xff, 0, 0xff).astype(UInt32)

        pixels[:] = bitwise_or(r, bitwise_or(g, b)).flat

        SDL_UnlockSurface(screen)
        SDL_Flip(screen)

