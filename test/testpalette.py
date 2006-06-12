#!/usr/bin/env python

'''A simple test of runtime palette modification for animation
using the `SDL_SetPalette` API.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from copy import copy, deepcopy
import math
import os
import random
import sys

from SDL import *

SCRW = 640
SCRH = 480

NBOATS = 5
SPEED = 2

SAIL_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'sail.bmp')

# wave colours (regex'd from testpalette.c)
wavemap = [
    SDL_Color(0,2,103), SDL_Color(0,7,110), SDL_Color(0,13,117),
    SDL_Color(0,19,125), SDL_Color(0,25,133), SDL_Color(0,31,141),
    SDL_Color(0,37,150), SDL_Color(0,43,158), SDL_Color(0,49,166),
    SDL_Color(0,55,174), SDL_Color(0,61,182), SDL_Color(0,67,190),
    SDL_Color(0,73,198), SDL_Color(0,79,206), SDL_Color(0,86,214),
    SDL_Color(0,96,220), SDL_Color(5,105,224), SDL_Color(12,112,226),
    SDL_Color(19,120,227), SDL_Color(26,128,229), SDL_Color(33,135,230),
    SDL_Color(40,143,232), SDL_Color(47,150,234), SDL_Color(54,158,236),
    SDL_Color(61,165,238), SDL_Color(68,173,239), SDL_Color(75,180,241),
    SDL_Color(82,188,242), SDL_Color(89,195,244), SDL_Color(96,203,246),
    SDL_Color(103,210,248), SDL_Color(112,218,250), SDL_Color(124,224,250),
    SDL_Color(135,226,251), SDL_Color(146,229,251), SDL_Color(156,231,252),
    SDL_Color(167,233,252), SDL_Color(178,236,252), SDL_Color(189,238,252),
    SDL_Color(200,240,252), SDL_Color(211,242,252), SDL_Color(222,244,252),
    SDL_Color(233,247,252), SDL_Color(242,249,252), SDL_Color(237,250,252),
    SDL_Color(209,251,252), SDL_Color(174,251,252), SDL_Color(138,252,252),
    SDL_Color(102,251,252), SDL_Color(63,250,252), SDL_Color(24,243,252),
    SDL_Color(7,225,252), SDL_Color(4,203,252), SDL_Color(3,181,252),
    SDL_Color(2,158,252), SDL_Color(1,136,251), SDL_Color(0,111,248),
    SDL_Color(0,82,234), SDL_Color(0,63,213), SDL_Color(0,50,192),
    SDL_Color(0,39,172), SDL_Color(0,28,152), SDL_Color(0,17,132),
    SDL_Color(0,7,114) ]


# Create a background surface
def make_bg(screen, startcol):
    bg = SDL_CreateRGBSurface(SDL_SWSURFACE, screen.w, screen.h,
                              8, 0, 0, 0, 0)
    # Set the palette to the logical screen palette so that blits
    # won't be translated
    SDL_SetColors(bg, screen.format.palette.colors, 0)

    SDL_LockSurface(bg)
    for i in range(SCRH):
        p = bg.pixels.as_bytes()
        d = 0
        for j in range(SCRW):
            v = min(max(d, -2), 2)
            if i > 0:
                v += p[i * bg.pitch - bg.pitch] + 65 - startcol
            p[j + i * bg.pitch] = startcol + (v & 63)
            d += random.randint(-1, 1)
    SDL_UnlockSurface(bg)
    return bg

def main():
    vidflags = 0
    fade_max = 400
    gamma_fade = 0
    gamma_ramp = 0
    boat = [None, None]

    SDL_Init(SDL_INIT_VIDEO)

    for arg in sys.argv[1:]:
        if arg == '-hw':
            vidflags |= SDL_HWSURFACE
        elif arg == '-fullscreen':
            vidflags |= SDL_FULLSCREEN
        elif arg == '-nofade':
            fade_max = 1
        elif arg == '-gamma':
            gamma_fade = 1
        elif arg == '-gammaramp':
            gamma_ramp = 1
        else:
            print >> sys.stderr, 'usage: testpalette.py ' + \
                '[-hw] [-fullscreen] [-nofade] [-gamma] [-gammaramp]'
            SDL_Quit()
            sys.exit(1)

    # Ask explicitly for 8bpp and a hardware palette
    screen = SDL_SetVideoMode(SCRW, SCRH, 8, vidflags | SDL_HWPALETTE)
    if vidflags & SDL_FULLSCREEN:
        SDL_ShowCursor(False)

    boat[0] = SDL_LoadBMP(SAIL_BMP)
    boatcols = boat[0].format.palette.ncolors
    # We've chosen magenta (#ff00ff) as the colour key for the boat
    SDL_SetColorKey(boat[0], SDL_SRCCOLORKEY | SDL_RLEACCEL,
                    SDL_MapRGB(boat[0].format, 0xff, 0x00, 0xff))
    boat[1] = boat[0] # TODO hflip
    SDL_SetColorKey(boat[1], SDL_SRCCOLORKEY | SDL_RLEACCEL,
                    SDL_MapRGB(boat[0].format, 0xff, 0x00, 0xff))

    # First set the physical screen palette to black, so the user won't
    # see our initial drawing on the screen.
    cmap = [SDL_Color(0, 0, 0)] * 256
    #SDL_SetPalette(screen, SDL_PHYSPAL, cmap, 0)

    # Proper palette management is important when playing games with
    # the colormap.  We have divided the palette as follows:
    #
    # index 0..(boatcols-1):        used for the boat
    # index boatcols..(boatcols+63) used for the waves
    SDL_SetPalette(screen, SDL_LOGPAL, boat[0].format.palette.colors, 0)
    SDL_SetPalette(screen, SDL_LOGPAL, wavemap, boat[0].format.palette.ncolors)

    # Now the logical screen palette is set, and will remain unchanged.
    # The boats already have the same palette so fast blits can be used.
    cmap = screen.format.palette.colors[:]

    # Save the index of the red colour for lated
    red = SDL_MapRGB(screen.format, 0xff, 0x00, 0x00)
    bg = make_bg(screen, boatcols)

    # Initial screen contents
    SDL_BlitSurface(bg, None, screen, None)
    SDL_Flip(screen)

    # Determine initial boat placements
    boatx = [random.randint(-boat[0].w, SCRW - 1) for i in range(NBOATS)]
    boaty = [i * (SCRH - boat[0].h) / (NBOATS - 1) for i in range(NBOATS)]
    boatdir = [random.choice([-1, 1]) for i in range(NBOATS)]

    start = SDL_GetTicks()
    frames = 0
    fade_dir = 1
    fade_level = 1
    r = SDL_Rect()
    updates = [SDL_Rect() for i in range(NBOATS)]
    print 'enter loop'
    while fade_level > 0:
        # Exit on any key or mouse button event
        event = SDL_PollEventAndReturn()
        while event:
            if event.type == SDL_KEYDOWN or \
               event.type == SDL_QUIT or \
               event.type == SDL_MOUSEBUTTONDOWN:
               if fade_dir < 0:
                   fade_level = 0
               fade_dir = -1
            event = SDL_PollEventAndReturn()

        # Move boats
        for i in range(NBOATS):
            old_x = boatx[i]
            # Update boat position
            boatx[i] += boatdir[i] * SPEED
            if boatx[i] <= -boat[0].w or boatx[i] >= SCRW:
                boatdir[i] = -boatdir[i]

            # Paint over the old boat position
            r.x = old_x
            r.y = boaty[i]
            r.w = boat[0].w
            r.h = boat[0].h
            SDL_BlitSurface(bg, r, screen, r)

            # Construct update rectangle (bounding box of old and new pos)
            updates[i].x = min(old_x, boatx[i])
            updates[i].y = boaty[i]
            updates[i].w = boat[0].w + SPEED
            updates[i].h = boat[0].h
            # Clip update rectangle to screen
            if updates[i].x < 0:
                updates[i].w += updates[i].x
                updates[i].x = 0
            if updates[i].x + updates[i].w > SCRW:
                updates[i].w = SCRW - int(updates[i].x)

        for i in range(NBOATS):
            # Paint boat on new position
            r.x = boatx[i]
            r.y = boaty[i]
            SDL_BlitSurface(boat[(boatdir[i] + 1) / 2], None, screen, r)

        # Cycle wave palette
        for i in range(64):
            cmap[boatcols + ((i + frames) & 63)] = copy(wavemap[i])

        if fade_dir:
            fade_level += fade_dir

            if gamma_fade:
                level = float(fade_level) / fade_max
                SDL_SetGamma(level, level, level)
            elif gamma_ramp:
                ramp = [(i * fade_level / fade_max) << 8 for i in range(256)]
                SDL_SetGammaRamp(ramp, ramp, ramp)
            else:
                # Can't copy with slice notation; SDL_Colour in a list
                # would reference the same object.
                cmap[:boatcols] = \
                    deepcopy(screen.format.palette.colors[:boatcols])
                for i in range(boatcols + 64):
                    cmap[i].r = cmap[i].r * fade_level / fade_max
                    cmap[i].g = cmap[i].g * fade_level / fade_max
                    cmap[i].b = cmap[i].b * fade_level / fade_max
            if fade_level == fade_max:
                fade_dir = 0

        # Pulse the red colour (done after the fade, for a night effect)
        # TODO
        SDL_SetPalette(screen, SDL_PHYSPAL, cmap, 0)

        # Update changed areas of the screen
        SDL_UpdateRects(screen, updates)
        frames += 1

    print '%d frames, %.2f fps' % \
        (frames, 1000.0 * frames / (SDL_GetTicks() - start))

    if vidflags & SDL_FULLSCREEN:
        SDL_ShowCursor(True)
    
    SDL_Quit()

if __name__ == '__main__':
    main()
