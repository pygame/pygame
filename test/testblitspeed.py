#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import random
import os
import sys

from SDL import *

SAMPLE_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'sample.bmp')

src = None
dest = None
testSeconds = 10

def percent(val, total):
    return int(float(val) / total * 100.0)

def output_videoinfo_details():
    info = SDL_GetVideoInfo()
    print 'SDL_GetVideoInfo:'
    if not info:
        print '  (null.)'
    else:
        print '  hardware surface available: %r' % info.hw_available
        print '  window manager available: %r' % info.wm_available
        print '  accelerated hardware->hardware blits: %r' % info.blit_hw
        print '  accelerated hardware->hardware colorkey blits: %r' % \
            info.blit_hw_CC
        print '  accelerated hardware->hardware alpha blits: %r' % \
            info.blit_hw_A
        print '  accelerated software->hardware blits: %r' % info.blit_sw
        print '  accelerated software->hardware colorkey blits: %r' % \
            info.blit_sw_CC
        print '  accelerated software->hardware alpha blits: %r' % \
            info.blit_sw_A
        print '  accelerated color fills: %r' % info.blit_fill
        print '  video memory: %d' % info.video_mem
    print

def format_flag(flags, name):
    if flags & eval(name):
        return ' %s' % name
    else:
        return ''

def output_surface_details(name, surface):
    print 'Details for %s:' % name
    print '  width      : %d' % surface.w 
    print '  height     : %d' % surface.h 
    print '  depth      : %d bits per pixel' % surface.format.BitsPerPixel
    print '  pitch      : %d' % surface.pitch
    print '  alpha      : %d' % surface.format.alpha
    print '  colorkey   : 0x%X' % surface.format.colorkey
    print '  red bits   : 0x%08X mask, %d shift, %d loss' % \
        (surface.format.Rmask,
         surface.format.Rshift,
         surface.format.Rloss)
    print '  green bits : 0x%08X mask, %d shift, %d loss' % \
        (surface.format.Gmask,
         surface.format.Gshift,
         surface.format.Gloss)
    print '  blue bits  : 0x%08X mask, %d shift, %d loss' % \
        (surface.format.Bmask,
         surface.format.Bshift,
         surface.format.Bloss)
    print '  alpha bits : 0x%08X mask, %d shift, %d loss' % \
        (surface.format.Amask,
         surface.format.Ashift,
         surface.format.Aloss)
    
    flags = ''
    if surface.flags & SDL_HWSURFACE == 0:
        flags = ' SDL_SWSURFACE'

    flags += format_flag(surface.flags, 'SDL_HWSURFACE')
    flags += format_flag(surface.flags, 'SDL_ASYNCBLIT')
    flags += format_flag(surface.flags, 'SDL_ANYFORMAT')
    flags += format_flag(surface.flags, 'SDL_HWPALETTE')
    flags += format_flag(surface.flags, 'SDL_DOUBLEBUF')
    flags += format_flag(surface.flags, 'SDL_FULLSCREEN')
    flags += format_flag(surface.flags, 'SDL_OPENGL')
    flags += format_flag(surface.flags, 'SDL_OPENGLBLIT')
    flags += format_flag(surface.flags, 'SDL_RESIZABLE')
    flags += format_flag(surface.flags, 'SDL_NOFRAME')
    flags += format_flag(surface.flags, 'SDL_HWACCEL')
    flags += format_flag(surface.flags, 'SDL_SRCCOLORKEY')
    flags += format_flag(surface.flags, 'SDL_RLEACCELOK')
    flags += format_flag(surface.flags, 'SDL_RLEACCEL')
    flags += format_flag(surface.flags, 'SDL_SRCALPHA')
    flags += format_flag(surface.flags, 'SDL_PREALLOC')

    print '  flags      :%s' % flags
    print

def output_details():
    output_videoinfo_details()
    output_surface_details('Source Surface', src)
    output_surface_details('Destination Surface', dest)

def blit(dst, src, x, y):
    srcRect = SDL_Rect(0, 0, src.w, src.h) # SDL will clip as appropriate
    dstRect = SDL_Rect(x, y, src.w, src.h)

    start = SDL_GetTicks()
    SDL_BlitSurface(src, srcRect, dst, dstRect)
    return SDL_GetTicks() - start

def blitCentered(dst, src):
    x = (dst.w - src.w) / 2
    y = (dst.h - src.h) / 2
    blit(dst, src, x, y)

def setup_test():
    global src, dest
    dstbpp = 32
    dstrmask = 0x00FF0000
    dstgmask = 0x0000FF00
    dstbmask = 0x000000FF
    dstamask = 0x00000000
    dstflags = 0
    dstw = 640
    dsth = 480
    srcbpp = 32
    srcrmask = 0x00FF0000
    srcgmask = 0x0000FF00
    srcbmask = 0x000000FF
    srcamask = 0x00000000
    srcflags = 0
    srcw = 640
    srch = 480
    origsrcalphaflags = 0
    origdstalphaflags = 0
    srcalphaflags = 0
    dstalphaflags = 0
    srcalpha = 255
    dstalpha = 255
    screenSurface = 0
    dumpfile = None

    i = 1
    for arg in sys.argv[1:]:
        if arg[:2] != '--':
            continue
        elif arg == '--dstbpp':
            dstbpp = int(sys.argv[i + 1])
        elif arg == '--dstrmask':
            dstrmask = int(sys.argv[i + 1], 16)
        elif arg == '--dstgmask':
            dstgmask = int(sys.argv[i + 1], 16)
        elif arg == '--dstbmask':
            dstbmask = int(sys.argv[i + 1], 16)
        elif arg == '--dstamask':
            dstamask = int(sys.argv[i + 1], 16)
        elif arg == '--dstwidth':
            dstw = int(sys.argv[i + 1])
        elif arg == '--dstheight':
            dsth = int(sys.argv[i + 1])
        elif arg == '--dsthwsurface':
            dstflags |= SDL_HWSURFACE
        elif arg == '--srcbpp':
            srcbpp = int(sys.argv[i + 1])
        elif arg == '--srcrmask':
            srcrmask = int(sys.argv[i + 1], 16)
        elif arg == '--srcgmask':
            srcgmask = int(sys.argv[i + 1], 16)
        elif arg == '--srcbmask':
            srcbmask = int(sys.argv[i + 1], 16)
        elif arg == '--srcamask':
            srcamask = int(sys.argv[i + 1], 16)
        elif arg == '--srcwidth':
            srcw = int(sys.argv[i + 1])
        elif arg == '--srcheight':
            srch = int(sys.argv[i + 1])
        elif arg == '--srchwsurface':
            srcflags |= SDL_HWSURFACE
        elif arg == '--seconds':
            global testSeconds
            testSeconds = int(sys.argv[i + 1])
        elif arg == '--screen':
            screenSurface = 1
        elif arg == '--dumpfile':
            dumpfile = sys.argv[i + 1]
        i += 1

    SDL_Init(SDL_INIT_VIDEO)
    bmp = SDL_LoadBMP(SAMPLE_BMP)

    if dstflags & SDL_HWSURFACE == 0:
        dstflags |= SDL_SWSURFACE
    if srcflags & SDL_HWSURFACE == 0:
        srcflags |= SDL_SWSURFACE

    if screenSurface:
        dest = SDL_SetVideoMode(dstw, dsth, dstbpp, dstflags)
    else:
        dest = SDL_CreateRGBSurface(dstflags, dstw, dsth, dstbpp,
                                    dstrmask,dstgmask, dstbmask, dstamask)

    src = SDL_CreateRGBSurface(srcflags, srcw, srch, srcbpp,
                               srcrmask, srcgmask, srcbmask, srcamask)
    
    # handle alpha settings
    srcalphaflags = (src.flags & SDL_SRCALPHA) | (src.flags & SDL_RLEACCEL)
    dstalphaflags = (dest.flags & SDL_SRCALPHA) | (dest.flags & SDL_RLEACCEL)
    origsrcalphaflags = srcalphaflags
    origdstalphaflags = dstalphaflags

    i = 1
    for arg in sys.argv[1:]:
        if arg[:2] != '--':
            continue
        elif arg == '--srcalpha':
            srcalpha = int(sys.argv[i + 1])
        elif arg == '--dstalpha':
            dstalpha = int(sys.argv[i + 1])
        elif arg == '--srcsrcalpha':
            srcalphaflags |= SDL_SRCALPHA
        elif arg == '--srcnosrcalpha':
            srcalphaflags &= ~SDL_SRCALPHA
        elif arg == '--srcrelaccel':
            srcalphaflags |= SDL_RLEACCEL
        elif arg == '--srcnorleaccel':
            srcalphaflags &= ~SDL_RLEACCEL
        elif arg == '--dstsrcalpha':
            dstalphaflags |= SDL_SRCALPHA
        elif arg == '--dstnosrcalpha':
            dstalphaflags &= ~SDL_SRCALPHA
        elif arg == '--dstrelaccel':
            dstalphaflags |= SDL_RLEACCEL
        elif arg == '--dstnorleaccel':
            dstalphaflags &= ~SDL_RLEACCEL

    if dstalphaflags != origdstalphaflags or dstalpha != dest.format.alpha:
        SDL_SetAlpha(dest, dstalphaflags, dstalpha)
    if srcalphaflags != origsrcalphaflags or srcalpha != src.format.alpha:
        SDL_SetAlpha(src, srcalphaflags, srcalpha)

    # set some sane defaults so we can see if the blit code is broken
    SDL_FillRect(dest, None, SDL_MapRGB(dest.format, 0, 0, 0))
    SDL_FillRect(src, None, SDL_MapRGB(src.format, 0, 0, 0))

    blitCentered(src, bmp)
    SDL_FreeSurface(bmp)

    if dumpfile:
        SDL_SaveBMP(src, dumpfile)  # make sure initial convert is sane

    output_details()

def test_blit_speed():
    clearColor = SDL_MapRGB(dest.format, 0, 0, 0)
    iterations = 0
    elapsed = 0
    end = 0
    last = 0
    testms = testSeconds * 1000
    wmax = dest.w - src.w
    hmax = dest.h - src.h
    isScreen = SDL_GetVideoSurface() is not None

    print 'Testing blit speed for %d seconds...' % testSeconds

    now = SDL_GetTicks()
    end = now + testms

    while now < end:
        if now - last > 1000:
            # Pump the event queue occasionally to keep OS happy
            last = now
            while SDL_PollEventAndReturn(): 
                pass
        
        iterations += 1
        elapsed += blit(dest, src, random.randint(0, wmax), 
                        random.randint(0, hmax))
        if isScreen:
            SDL_Flip(dest)
            SDL_FillRect(dest, None, clearColor)

        now = SDL_GetTicks()

    print 'Non-blitting crap accounted for %d percent of this run.' % \
        percent(testms - elapsed, testms)

    print '%d blits took %d ms (%d fps).' % \
        (iterations, elapsed, iterations / (elapsed / 1000.0))

if __name__ == '__main__':
    setup_test()
    test_blit_speed()
    SDL_Quit()
