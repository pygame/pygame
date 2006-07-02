#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from SDL import *

if __name__ == '__main__':
    print 'Wait a few seconds, this is not very speedy...'

    SDL_Init(SDL_INIT_VIDEO)

    screen = SDL_SetVideoMode(800, 600, 32, SDL_SWSURFACE)
    format = screen.format

    '''
    Draw a 3 colour gradient by setting the pixel values directly.
    '''
    pixels = [0] * len(screen.pixels)
    deltaX = 255 / float(screen.w)
    deltaY = 255 / float(screen.h)
    b = 0
    i = 0
    for y in range(screen.h):
        r, g = 0, 255
        for x in range(screen.w):
            r += deltaX
            g -= deltaX
            pixels[i] = SDL_MapRGB(format, int(r), int(g), int(b))
            i += 1
        b += deltaY
        i += screen.pitch / format.BytesPerPixel - screen.w

    SDL_LockSurface(screen)
    screen.pixels[:] = pixels
    SDL_UnlockSurface(screen)
    SDL_Flip(screen)

    # Keep the window open
    while True:
        pass
