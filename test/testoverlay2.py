#!/usr/bin/env python

'''Test of the overlay used for moving pictures.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import sys

from SDL import *

MOOSE_DAT = os.path.join(os.path.dirname(sys.argv[0]), 'moose.dat')
MOOSEPIC_W = 64
MOOSEPIC_H = 88

MOOSEFRAME_SIZE = MOOSEPIC_W * MOOSEPIC_H
MOOSEFRAMES_COUNT = 10

MooseColors = [
    SDL_Color( 49,  49,  49), SDL_Color( 66,  24,   0), 
    SDL_Color( 66,  33,   0), SDL_Color( 66,  66,  66),
    SDL_Color( 66, 115,  49), SDL_Color( 74,  33,   0), 
    SDL_Color( 74,  41,  16), SDL_Color( 82,  33,   8),
    SDL_Color( 82,  41,   8), SDL_Color( 82,  49,  16), 
    SDL_Color( 82,  82,  82), SDL_Color( 90,  41,   8),
    SDL_Color( 90,  41,  16), SDL_Color( 90,  57,  24),
    SDL_Color( 99,  49,  16), SDL_Color( 99,  66,  24),
    SDL_Color( 99,  66,  33), SDL_Color( 99,  74,  33), 
    SDL_Color(107,  57,  24), SDL_Color(107,  82,  41),
    SDL_Color(115,  57,  33), SDL_Color(115,  66,  33), 
    SDL_Color(115,  66,  41), SDL_Color(115,  74,   0),
    SDL_Color(115,  90,  49), SDL_Color(115, 115, 115), 
    SDL_Color(123,  82,   0), SDL_Color(123,  99,  57),
    SDL_Color(132,  66,  41), SDL_Color(132,  74,  41), 
    SDL_Color(132,  90,   8), SDL_Color(132,  99,  33),
    SDL_Color(132,  99,  66), SDL_Color(132, 107,  66), 
    SDL_Color(140,  74,  49), SDL_Color(140,  99,  16),
    SDL_Color(140, 107,  74), SDL_Color(140, 115,  74), 
    SDL_Color(148, 107,  24), SDL_Color(148, 115,  82),
    SDL_Color(148, 123,  74), SDL_Color(148, 123,  90), 
    SDL_Color(156, 115,  33), SDL_Color(156, 115,  90),
    SDL_Color(156, 123,  82), SDL_Color(156, 132,  82), 
    SDL_Color(156, 132,  99), SDL_Color(156, 156, 156),
    SDL_Color(165, 123,  49), SDL_Color(165, 123,  90), 
    SDL_Color(165, 132,  82), SDL_Color(165, 132,  90),
    SDL_Color(165, 132,  99), SDL_Color(165, 140,  90), 
    SDL_Color(173, 132,  57), SDL_Color(173, 132,  99),
    SDL_Color(173, 140, 107), SDL_Color(173, 140, 115), 
    SDL_Color(173, 148,  99), SDL_Color(173, 173, 173),
    SDL_Color(181, 140,  74), SDL_Color(181, 148, 115), 
    SDL_Color(181, 148, 123), SDL_Color(181, 156, 107),
    SDL_Color(189, 148, 123), SDL_Color(189, 156,  82), 
    SDL_Color(189, 156, 123), SDL_Color(189, 156, 132),
    SDL_Color(189, 189, 189), SDL_Color(198, 156, 123), 
    SDL_Color(198, 165, 132), SDL_Color(206, 165,  99),
    SDL_Color(206, 165, 132), SDL_Color(206, 173, 140), 
    SDL_Color(206, 206, 206), SDL_Color(214, 173, 115),
    SDL_Color(214, 173, 140), SDL_Color(222, 181, 148), 
    SDL_Color(222, 189, 132), SDL_Color(222, 189, 156),
    SDL_Color(222, 222, 222), SDL_Color(231, 198, 165), 
    SDL_Color(231, 231, 231), SDL_Color(239, 206, 173)
]

formats = {
    'YV12': SDL_YV12_OVERLAY,
    'IYUV': SDL_IYUV_OVERLAY,
    'YUY2': SDL_YUY2_OVERLAY,
    'UYVY': SDL_UYVY_OVERLAY,
    'YVYU': SDL_YVYU_OVERLAY,
}

def RGBtoYUV(rgb, monochrome, luminance):
    yuv = [0, 0, 0]

    if monochrome:
        yuv[0] = int(0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2])
        yuv[1] = 128
        yuv[2] = 128
    else:
        yuv[0] = int(0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2])
        yuv[1] = int((rgb[2] - yuv[0]) * 0.565 + 128)
        yuv[2] = int((rgb[0] - yuv[0]) * 0.713 + 128)

    if luminance != 100:
        yuv[0] = min(yuv[0] * luminance / 100, 255)

    return yuv

def ConvertRGBtoYV12(s, o, monochrome, luminance):
    SDL_LockSurface(s)
    SDL_LockYUVOverlay(o)

    p = s.pixels.as_bytes()
    op = o.pixels
    pdx = s.format.BytesPerPixel
    for y in range(min(s.h, o.h)):
        pi = s.pitch * y
        oi = o.pitches[0] * y
        oi2 = o.pitches[2] * (y/2)
        for x in range(min(s.w, o.w)):
            yuv = RGBtoYUV((p[pi], p[pi+1], p[pi+2]), monochrome, luminance)
            op[0][oi] = yuv[0]
            oi += 1
            if x % 2 == 0 and y % 2 == 0:
                op[1][oi2] = yuv[2]
                op[2][oi2] = yuv[1]
                oi2 += 1
            pi += pdx
    SDL_UnlockYUVOverlay(o)
    SDL_UnlockSurface(s)

def ConvertRGBtoIYUV(s, o, monochrome, luminance):
    SDL_LockSurface(s)
    SDL_LockYUVOverlay(o)

    p = s.pixels.as_bytes()
    op = o.pixels
    pdx = s.format.BytesPerPixel
    for y in range(min(s.h, o.h)):
        pi = s.pitch * y
        oi = o.pitches[0] * y
        oi2 = o.pitches[2] * (y/2)
        for x in range(min(s.w, o.w)):
            yuv = RGBtoYUV((p[pi], p[pi+1], p[pi+2]), monochrome, luminance)
            op[0][oi] = yuv[0]
            oi += 1
            if x % 2 == 0 and y % 2 == 0:
                op[1][oi2] = yuv[1]
                op[2][oi2] = yuv[2]
                oi2 += 1
            pi += pdx
    SDL_UnlockYUVOverlay(o)
    SDL_UnlockSurface(s)

def ConvertRGBtoUYVY(s, o, monochrome, luminance):
    SDL_LockSurface(s)
    SDL_LockYUVOverlay(o)

    p = s.pixels.as_bytes()
    op = o.pixels[0]
    pdx = s.format.BytesPerPixel
    for y in range(min(s.h, o.h)):
        pi = s.pitch * y
        oi = o.pitches[0] * y
        for x in range(min(s.w, o.w)):
            yuv = RGBtoYUV((p[pi], p[pi+1], p[pi+2]), monochrome, luminance)
            if x % 2 == 0:
                op[oi] = yuv[1]
                op[oi + 1] = yuv[0]
                op[oi + 2] = yuv[2]
                oi += 3
            else:
                op[oi] = yuv[0]
                oi += 1

            pi += pdx
    SDL_UnlockYUVOverlay(o)
    SDL_UnlockSurface(s)

def ConvertRGBtoYVYU(s, o, monochrome, luminance):
    SDL_LockSurface(s)
    SDL_LockYUVOverlay(o)

    p = s.pixels.as_bytes()
    op = o.pixels[0]
    pdx = s.format.BytesPerPixel
    for y in range(min(s.h, o.h)):
        pi = s.pitch * y
        oi = o.pitches[0] * y
        for x in range(min(s.w, o.w)):
            yuv = RGBtoYUV((p[pi], p[pi+1], p[pi+2]), monochrome, luminance)
            if x % 2 == 0:
                op[oi] = yuv[0]
                op[oi + 1] = yuv[2]
                op[oi + 3] = yuv[1]
            else:
                op[oi] = yuv[0]

            pi += pdx
            oi += 2
    SDL_UnlockYUVOverlay(o)
    SDL_UnlockSurface(s)


def ConvertRGBtoYUY2(s, o, monochrome, luminance):
    SDL_LockSurface(s)
    SDL_LockYUVOverlay(o)

    p = s.pixels.as_bytes()
    op = o.pixels[0]
    pdx = s.format.BytesPerPixel
    for y in range(min(s.h, o.h)):
        pi = s.pitch * y
        oi = o.pitches[0] * y
        for x in range(min(s.w, o.w)):
            yuv = RGBtoYUV((p[pi], p[pi+1], p[pi+2]), monochrome, luminance)
            if x % 2 == 0:
                op[oi] = yuv[0]
                op[oi + 1] = yuv[1]
                op[oi + 3] = yuv[2]
            else:
                op[oi] = yuv[0]

            pi += pdx
            oi += 2
    SDL_UnlockYUVOverlay(o)
    SDL_UnlockSurface(s)

def PrintUsage():
    print >> sys.stderr, '''Usage: %s [arg] [arg] [arg] ...

Where 'arg' is any of the following options:

    -fps <frames per second>
    -format <fmt> (one of %s)
    -scale <scale factor> (initial scale of the overlay)
    -help (shows this help)

Press ESC to exit, or SPACE to freeze the movie while application running.
''' % (sys.argv[0], ', '.join(formats.keys()))


if __name__ == '__main__':
    overlay_format = SDL_YUY2_OVERLAY
    fps = 12
    scale = 5

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE)

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-fps':
            i += 1
            fps = int(sys.argv[i])
        elif arg == '-format':
            i += 1
            overlay_format = formats[sys.argv[i]]
        elif arg == '-scale':
            i += 1
            scale = int(sys.argv[i])
        elif arg in ('-help', '-h'):
            PrintUsage()
        else:
            print >> sys.stderr, 'Unrecognised option: %s.' % arg
            sys.exit(10)
        i += 1

    RawMooseData = open(MOOSE_DAT, 'rb').read()

    screen = SDL_SetVideoMode(MOOSEPIC_W*scale, MOOSEPIC_H*scale, 0,
                              SDL_RESIZABLE | SDL_SWSURFACE)

    SDL_WM_SetCaption('SDL test overlay: running moose', 'testoverlay2')

    MooseFrame = []
    for i in range(MOOSEFRAMES_COUNT):
        frame = SDL_CreateRGBSurfaceFrom(\
            RawMooseData[i*MOOSEFRAME_SIZE:(i+1)*MOOSEFRAME_SIZE],
            MOOSEPIC_W, MOOSEPIC_H, 8, MOOSEPIC_W, 0, 0, 0, 0)
        SDL_SetColors(frame, MooseColors, 0)
        format = SDL_PixelFormat()
        format.BitsPerPixel = 32
        format.BytesPerPixel = 4
        format.Rshift = 0 # TODO big endian
        format.Gshift = 8
        format.Bshift = 16
        format.Rmask = 0xff << format.Rshift
        format.Gmask = 0xff << format.Gshift
        format.Bmask = 0xff << format.Bshift
        newsurf = SDL_ConvertSurface(frame, format, SDL_SWSURFACE)
        SDL_FreeSurface(frame)
        MooseFrame.append(newsurf)

    overlay = SDL_CreateYUVOverlay(MOOSEPIC_W, MOOSEPIC_H, 
                                   overlay_format, screen)
    
    formatstr = 'Unknown'
    for key, val in formats.items():
        if val == overlay.format:
            formatstr = key
    typestr = 'software'
    if overlay.hw_overlay:
        typestr = 'hardware'
    print 'Created %dx%dx%d %s %s overlay' % (overlay.w, overlay.h,
        overlay.planes, typestr, formatstr)

    for i in range(overlay.planes):
        print '  plane %d: pitch=%d' % (i, overlay.pitches[i])

    overlayrect = SDL_Rect(0, 0, MOOSEPIC_W*scale, MOOSEPIC_H*scale)
    i = 0
    fpsdelay = 1000/fps
    paused = 0
    resized = 0

    SDL_EventState(SDL_KEYUP, SDL_IGNORE)
    lastftick = SDL_GetTicks()

    while True:
        event = SDL_PollEventAndReturn()
        if event:
            if event.type == SDL_VIDEORESIZE:
                screen = SDL_SetVideoMode(event.w, event.h, 
                                          0, SDL_RESIZABLE | SDL_SWSURFACE)
                overlayrect.w = event.w
                overlayrect.h = event.h
                if paused:
                    resized = 1
            elif event.type == SDL_MOUSEBUTTONDOWN:
                overlayrect.x = event.x - overlayrect.w/2
                overlayrect.y = event.y - overlayrect.h/2
            elif event.type == SDL_KEYDOWN:
                if event.keysym.sym == SDLK_SPACE:
                    paused = not paused
            elif (event.type == SDL_KEYDOWN and \
                  event.keysym.sym == SDLK_ESCAPE) or event.type == SDL_QUIT:
                SDL_FreeYUVOverlay(overlay)
                for f in MooseFrame:
                    SDL_FreeSurface(f)
                SDL_Quit()
                sys.exit(0)
        
        if (not paused) or resized:
            if SDL_GetTicks() - lastftick > fpsdelay or resized:
                lastftick = SDL_GetTicks()

                if overlay_format == SDL_YUY2_OVERLAY:
                    ConvertRGBtoYUY2(MooseFrame[i], overlay, 0, 100)
                elif overlay_format == SDL_YVYU_OVERLAY:
                    ConvertRGBtoYVYU(MooseFrame[i], overlay, 0, 100)
                elif overlay_format == SDL_UYVY_OVERLAY:
                    ConvertRGBtoUYVY(MooseFrame[i], overlay, 0, 100)
                elif overlay_format == SDL_IYUV_OVERLAY:
                    ConvertRGBtoIYUV(MooseFrame[i], overlay, 0, 100)
                elif overlay_format == SDL_YV12_OVERLAY:
                    ConvertRGBtoYV12(MooseFrame[i], overlay, 0, 100)

                SDL_DisplayYUVOverlay(overlay, overlayrect)
                if not resized:
                    i += 1
                    if i == 10:
                        i = 0
                resized = 0

            SDL_Delay(1)
