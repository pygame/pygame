#!/usr/bin/env python

'''Simple program: Fill a colormap with gray and stripe it down the screen
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import random
import sys

from SDL import *

#NUM_COLORS = 16
NUM_COLORS = 256

# Draw a randomly sized and colored box centered about X,Y
def DrawBox(screen, X, Y, width, height):
    area = SDL_Rect()

    # Get the bounds of the rectangle
    area.w = random.randint(0, width)
    area.h = random.randint(0, height)
    area.x = X-area.w/2
    area.y = Y-area.h/2
    randc = random.randint(0, NUM_COLORS)

    if screen.format.BytesPerPixel == 1:
        color = randc
    else:
        color = SDL_MapRGB(screen.format, randc, randc, randc)

    # Do it!
    SDL_FillRect(screen, area, color)
    if screen.flags & SDL_DOUBLEBUF:
        SDL_Flip(screen)
    else:
        SDL_UpdateRects(screen, [area])

def DrawBackground(screen):
    for j in range(2):
        SDL_LockSurface(screen)
        
        if screen.format.BytesPerPixel != 2:
            buffer = screen.pixels.as_bytes()
            for i in range(screen.h):
                buffer[screen.pitch*i:screen.pitch*(i+1)] = \
                    [(i*(NUM_COLORS-1))/screen.h] * screen.pitch
        else:
            buffer = screen.pixels
            for i in range(screen.h):
                gradient = i*(NUM_COLORS-1)/screen.h
                color = SDL_MapRGB(screen.format, gradient, gradient, gradient)
                buffer[screen.w*i:screen.w*(i+1)] = \
                    [color] * screen.w

        SDL_UnlockSurface(screen)
        if screen.flags & SDL_DOUBLEBUF:
            SDL_Flip(screen)
        else:
            SDL_UpdateRect(screen, 0, 0, 0, 0)

def CreateScreen(w, h, bpp, flags):
    screen = SDL_SetVideoMode(w, h, bpp, flags)
    if bpp == 8:
        # Set a gray color map, reverse order from white to black
        palette = []
        for i in range(NUM_COLORS):
            c = SDL_Color()
            c.r = c.g = c.b = (NUM_COLORS - 1) - i * (256 / NUM_COLORS)
            palette.append(c)
        SDL_SetColors(screen, palette, 0)
    return screen

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    width = 640
    height = 480
    bpp = 8
    videoflags = SDL_SWSURFACE
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-width':
            i += 1
            width = int(sys.argv[i])
        elif sys.argv[i] == '-height':
            i += 1
            height = int(sys.argv[i])
        elif sys.argv[i] == '-bpp':
            i += 1
            bpp = int(sys.argv[i])
        elif sys.argv[i] == '-hw':
            videoflags |= SDL_HWSURFACE
        elif sys.argv[i] == '-hwpalette':
            videoflags |= SDL_HWPALETTE
        elif sys.argv[i] == '-flip':
            videoflags |= SDL_DOUBLEBUF
        elif sys.argv[i] == '-noframe':
            videoflags |= SDL_NOFRAME
        elif sys.argv[i] == '-resize':
            videoflags |= SDL_RESIZABLE
        elif sys.argv[i] == '-fullscreen':
            videoflags |= SDL_FULLSCREEN
        else:
            print >> sys.stderr, ('Usage: %s [-width] [-height] [-bpp] ' + \
                '[-hw] [-hwpalette] [-flip] [-noframe] [-fullscreen] ' + \
                '[-resize]') % sys.argv[0]
            sys.exit(1)
        i += 1

    # Set a video mode
    screen = CreateScreen(width, height, bpp, videoflags)

    DrawBackground(screen)

    # Wait for a keystroke
    done = False
    while not done:
        event = SDL_WaitEventAndReturn()
        if event.type == SDL_MOUSEBUTTONDOWN:
            DrawBox(screen, event.x, event.y, width, height)
        elif event.type == SDL_KEYDOWN:
            # Ignore ALT_TAB for windows
            if event.keysym.sym == SDLK_LALT or event.keysym.sym == SDLK_TAB:
                pass
            elif event.keysym.sym == SDLK_SPACE:
                SDL_WarpMouse(width/2, height/2)
            elif event.keysym.sym == SDLK_RETURN:
                videoflags ^= SDL_FULLSCREEN
                screen = CreateScreen(screen.w, screen.h,
                                      screen.format.BitsPerPixel, videoflags)
                DrawBackground(screen)
            else:
                done = True
        elif event.type == SDL_QUIT:
            done = True
        elif event.type == SDL_VIDEOEXPOSE:
            DrawBackground(screen)
        elif event.type == SDL_VIDEORESIZE:
            screen = CreateScreen(event.resize.w, event.resize.h, 
                                  screen.format.BitsPerPixel, videoflags)

    SDL_Quit()

