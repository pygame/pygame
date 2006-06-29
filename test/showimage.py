#!/usr/bin/env python

'''Test application for the SDL.image module.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *
from SDL.image import *

# Draw a Gimpish background pattern to show transparency in the image.
def draw_background(screen):
    dst = screen.pixels
    bpp = screen.format.BytesPerPixel
    col = (SDL_MapRGB(screen.format, 0x66, 0x66, 0x66),
           SDL_MapRGB(screen.format, 0x99, 0x99, 0x99))
    for y in range(screen.h):
        for x in range(screen.w):
            c = col[((x ^ y) >> 3) & 1]
            dst[y * screen.pitch/bpp + x] = c

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s <image_file>' % sys.argv[0]
        sys.exit(1)

    SDL_Init(SDL_INIT_VIDEO)

    i = 1
    flags = SDL_SWSURFACE
    if sys.argv[i] == '-fullscreen':
        SDL_ShowCursor(0)
        flags |= SDL_FULLSCREEN
        i += 1

    while i < len(sys.argv):
        # Open the image file
        image = IMG_Load(sys.argv[i])
        SDL_WM_SetCaption(sys.argv[i], 'showimage')

        # Create a display for the image
        depth = SDL_VideoModeOK(image.w, image.h, 32, flags)
        # Use the deepest native mode, expcept that we emulate 32bpp
        # for viewing non-indexed images on 8bpp screens.
        if depth == 0:
            if image.format.BytesPerPixel > 1:
                depth = 32
            else:
                depth = 8
        elif image.format.BytesPerPixel > 1 and depth == 8:
            depth = 32
        if depth == 8:
            flags |= SDL_HWPALETTE

        screen = SDL_SetVideoMode(image.w, image.h, depth, flags)
        
        # Set the palette, if one exists
        if image.format.palette:
            SDL_SetColors(screen, image.format.palette.colors, 0)

        # Draw a background pattern if the surface has transparency
        if image.flags & (SDL_SRCALPHA | SDL_SRCCOLORKEY):
            draw_background(screen)

        # Display the image
        SDL_BlitSurface(image, None, screen, None)
        SDL_UpdateRect(screen, 0, 0, 0, 0)

        done = 0
        while not done:
            event = SDL_WaitEventAndReturn()
            if event.type == SDL_KEYUP:
                if event.keysym.sym == SDLK_LEFT:
                    if i > 1:
                        i -= 2
                        done = 1
                elif event.keysym.sym == SDLK_RIGHT:
                    if i < len(sys.argv) - 1:
                        done = 1
                elif event.keysym.sym in (SDLK_ESCAPE, SDLK_q):
                    done = 1
                    i = len(sys.argv)
                elif event.keysym.sym in (SDLK_SPACE, SDLK_TAB):
                    done = 1
            elif event.type == SDL_MOUSEBUTTONDOWN:
                done = 1
            elif event.type == SDL_QUIT:
                i = len(sys.argv)
                done = 1

        SDL_FreeSurface(image)
        i += 1

    SDL_Quit()
