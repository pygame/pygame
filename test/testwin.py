#!/usr/bin/env python

'''Bring up a window and play with it.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from copy import copy, deepcopy
import os
import sys

from SDL import *

BENCHMARK_SDL = True
SCREENSHOT = False

SAMPLE_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'sample.bmp')

def DrawPict(screen, bmpfile, speedy, flip, nofade):
    if not bmpfile:
        bmpfile = SAMPLE_BMP
    print >> sys.stderr, 'Loading picture: %s' % bmpfile
    picture = SDL_LoadBMP(bmpfile)

    if picture.format.palette:
        ncolors = picture.format.palette.ncolors
        colors = deepcopy(picture.format.palette.colors[:])
    else:
        ncolors = 256
        colors = [SDL_Color() for i in range(ncolors)]
        for r in range(8):
            for g in range(8):
                for b in range(4):
                    i = (r << 5) | (g << 2) | b
                    colors[i].r = r << 5
                    colors[i].g = g << 5
                    colors[i].b = b << 6

    print 'testwin: setting colors'
    SDL_SetColors(screen, colors, 0)

    # Set the screen to black (not really necessary)
    SDL_LockSurface(screen)
    black = SDL_MapRGB(screen.format, 0, 0, 0)
    pixels = screen.pixels.as_bytes()
    # XXX note setting single-byte pixel from multibyte black, bug inherited
    # from original testwin.c; not a bug when black == 0.
    for i in range(screen.h):
        pixels[i*screen.pitch:i*screen.pitch+screen.w] = [black] * screen.w
    SDL_UnlockSurface(screen)
    SDL_UpdateRect(screen, 0, 0, 0, 0)

    # Display the picture
    if speedy:
        print >> sys.stderr, 'Converting picture'
        displayfmt = SDL_DisplayFormat(picture)
        SDLFreeSurface(picture)
        picture = displayfmt

    if picture.flags & SDL_HWSURFACE:
        print '(image surface located in video memory)'
    else:
        print '(image surface located in system memory)'

    centered = (screen.w - picture.w) / 2
    if centered < 0:
        centered = 0
    dest = SDL_Rect()
    dest.y = (screen.h - picture.h) / 2
    dest.w = picture.w
    dest.h = picture.h
    print 'testwin: moving image'
    for i in range(centered + 1):
        dest.x = i
        update = copy(dest)
        SDL_BlitSurface(picture, None, screen, update)
        if flip:
            SDL_Flip(screen)
        else:
            SDL_UpdateRects(screen, [update])

    if SCREENSHOT:
        SDL_SaveBMP(screen, 'screen.bmp')

    if not BENCHMARK_SDL:
        SDL_Delay(5 * 1000)

    if not nofade:
        maxstep = 32 - 1
        final = SDL_Color(0xff, 0, 0)
        palcolors = deepcopy(colors)
        cmap = deepcopy(colors)
        cdist = [(final.r - palcolors[i].r, 
                  final.g - palcolors[i].g,
                  final.b - palcolors[i].b) for i in range(ncolors)]

        for i in range(maxstep/2 + 1):  # halfway fade
            for c in range(ncolors):
                colors[c].r = palcolors[c].r + cdist[c][0] * i / maxstep
                colors[c].g = palcolors[c].g + cdist[c][1] * i / maxstep
                colors[c].b = palcolors[c].b + cdist[c][2] * i / maxstep
            SDL_SetColors(screen, colors, 0)
            SDL_Delay(1)

        final.r = 0
        final.g = 0
        final.b = 0
        palcolors = deepcopy(colors)
        cdist = [(final.r - palcolors[i].r, 
                  final.g - palcolors[i].g,
                  final.b - palcolors[i].b) for i in range(ncolors)]
        maxstep /= 2
        for i in range(maxstep + 1):    # finish fade out
            for c in range(ncolors):
                colors[c].r = palcolors[c].r + cdist[c][0] * i / maxstep
                colors[c].g = palcolors[c].g + cdist[c][1] * i / maxstep
                colors[c].b = palcolors[c].b + cdist[c][2] * i / maxstep
            SDL_SetColors(screen, colors, 0)
            SDL_Delay(1)

        for i in range(ncolors):
            colors[i].r = final.r
            colors[i].g = final.g
            colors[i].b = final.b
        SDL_SetColors(screen, colors, 0)

        print 'testwin: fading in...'
        palcolors = deepcopy(colors)
        cdist = [(cmap[i].r - palcolors[i].r, 
                  cmap[i].g - palcolors[i].g,
                  cmap[i].b - palcolors[i].b) for i in range(ncolors)]
        for i in range(maxstep + 1):    # 32 step fade in (XXX 16?)
            for c in range(ncolors):
                colors[c].r = palcolors[c].r + cdist[c][0] * i / maxstep
                colors[c].g = palcolors[c].g + cdist[c][1] * i / maxstep
                colors[c].b = palcolors[c].b + cdist[c][2] * i / maxstep
            SDL_SetColors(screen, colors, 0)
            SDL_Delay(1)
        print 'testwin: fading over'

        SDL_FreeSurface(picture)

if __name__ == '__main__':
    speedy = 0
    flip = 0
    nofade = 0
    delay = 1
    w = 640
    h = 480
    desired_bpp = 0
    video_flags = 0

    SDL_Init(SDL_INIT_VIDEO)

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-speedy':
            speedy = 1
        elif arg == '-nofade':
            nofade = 1
        elif arg == '-delay':
            i += 1
            delay = int(sys.argv[i])
        elif arg == '-width':
            i += 1
            width = int(sys.argv[i])
        elif arg == '-height':
            i += 1
            height = int(sys.argv[i])
        elif arg == '-bpp':
            i += 1
            desired_bpp = int(sys.argv[i])
        elif arg == '-warp':
            video_flags |= SDL_HWPALETTE
        elif arg == '-hw':
            video_flags |= SDL_HWSURFACE
        elif arg == '-flip':
            video_flags |= SDL_DOUBLEBUF
        elif arg == '-fullscreen':
            video_flags |= SDL_FULLSCREEN
        i += 1

    bmp = None
    if i < len(sys.argv):
        bmp = sys.argv[i]

    screen = SDL_SetVideoMode(w, h, desired_bpp, video_flags)
    fsstr = ''
    if screen.flags & SDL_FULLSCREEN:
        fsstr = ' fullscreen'
    print 'Set%s %dx%dx%d mode' % \
        (fsstr, screen.w, screen.h, screen.format.BitsPerPixel)
    if screen.flags & SDL_HWSURFACE:
        print '(video surface located in video memory)'
    else:
        print '(video surface located in system memory)'
    if screen.flags & SDL_DOUBLEBUF:
        print 'Double-buffering enabled'
        flip = 1

    SDL_WM_SetCaption('SDL test window', 'testwin')

    if BENCHMARK_SDL:
        then = SDL_GetTicks()
        DrawPict(screen, bmp, speedy, flip, nofade)
        now = SDL_GetTicks()
        print 'Time: %d milliseconds' % (now - then)
    else:
        DrawPict(screen, bmp, speedy, flip, nofade)
    SDL_Delay(delay * 1000)
    SDL_Quit()
