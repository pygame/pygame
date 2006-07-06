#!/usr/bin/env python

'''Another interoperability test with PIL.

This time the buffer of a single surface is updated using a slice assignment,
rather than creating a new buffer each frame.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import math
import os
import sys

from SDL import *
import SDL.array
import Image

DEFAULT_IMAGE = os.path.join(os.path.dirname(sys.argv[0]), 'sample.bmp')

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    image_filename = DEFAULT_IMAGE
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]

    # Load image using PIL and make canvas large enough to hold any rotation
    src_image = Image.open(image_filename)
    size = int(math.sqrt(src_image.size[0]**2 + src_image.size[1]**2) + 0.5)
    original_image = Image.new('RGBA', (size, size))
    original_image.paste(src_image, ((size - src_image.size[0]) / 2, 
                                     (size - src_image.size[1]) / 2))

    screen = SDL_SetVideoMode(size, size, 32, 0)

    surface = SDL_CreateRGBSurface(0,
        size, size, 32,
        0x000000ff,
        0x0000ff00,
        0x00ff0000,
        0x00000000)

    done = False
    angle = 0
    while not done:
        event = SDL_PollEventAndReturn()
        while event:
            if event.type in (SDL_QUIT, SDL_KEYDOWN):
                done = True
            event = SDL_PollEventAndReturn()

        angle = (angle + 1) % 360
        image = original_image.rotate(angle)


        SDL_LockSurface(surface)
        surface.pixels[:] = \
            [c[0] | (c[1] << 8) | (c[2] << 16) for c in image.getdata()]

        # This alternative also works
        #surface.pixels.as_bytes()[:] = \
        #    [ord(c) for c in image.tostring()]
        SDL_UnlockSurface(surface)

        SDL_BlitSurface(surface, None, screen, None)
        SDL_UpdateRect(screen, 0, 0, 0, 0)

    SDL_FreeSurface(surface)
    SDL_Quit()
