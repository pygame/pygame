#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys
import os.path
import random

from SDL import *

NUM_BLITS = 10
NUM_UPDATES = 500

FLAG_MASK = SDL_HWSURFACE | SDL_FULLSCREEN | SDL_DOUBLEBUF | \
            SDL_SRCCOLORKEY | SDL_SRCALPHA | SDL_RLEACCEL  | \
            SDL_RLEACCELOK

SAMPLE_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'sample.bmp')

def PrintFlags(flags):
    print '0x%8.8x' % (flags & FLAG_MASK),
    if flags & SDL_HWSURFACE:
        print 'SDL_HWSURFACE',
    else:
        print 'SDL_SWSURFACE',
    if flags & SDL_FULLSCREEN:
        print '| SDL_FULLSCREEN',
    if flags & SDL_DOUBLEBUF:
        print '| SDL_DOUBLEBUF',
    if flags & SDL_SRCCOLORKEY:
        print '| SDL_SRCCOLORKEY',
    if flags & SDL_SRCALPHA:
        print '| SDL_SRCALPHA',
    if flags & SDL_RLEACCEL:
        print '| SDL_RLEACCEL',
    if flags & SDL_RLEACCELOK:
        print '| SDL_RLEACCELOK',

def RunBlitTests(screen, bmp, blitcount):
    maxx = screen.w - bmp.w
    maxy = screen.h - bmp.h
    dst = SDL_Rect()
    for i in range(NUM_UPDATES):
        for j in range(blitcount):
            dst.x = random.randint(0, maxx)
            dst.y = random.randint(0, maxy)
            dst.w = bmp.w
            dst.h = bmp.h
            SDL_BlitSurface(bmp, None, screen, dst)
        SDL_Flip(screen)
    return i

def RunModeTests(screen):
    # while (SDL_PollEvent):
    #   etc

    print 'Running color fill and fullscreen update test'
    then = SDL_GetTicks()
    frames = 0
    for i in range(256):
        SDL_FillRect(screen, None, SDL_MapRGB(screen.format, i, 0, 0))
        SDL_Flip(screen)
        frames += 1
    for i in range(256):
        SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, i, 0))
        SDL_Flip(screen)
        frames += 1
    for i in range(256):
        SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, i))
        SDL_Flip(screen)
        frames += 1
    now = SDL_GetTicks()
    seconds = (now - then) / 1000.0
    if seconds > 0.0:
        print '%d fills and flips in %2.2f seconds, %2.2f FPS' % \
            (frames, seconds, frames / seconds)
    else:
        print '%d fills and flips in zero seconds!' % frames

    # clear the screen after fill test
    SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, 0))
    SDL_Flip(screen)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    bmp = SDL_LoadBMP(SAMPLE_BMP)
    print 'Running freshly loaded blit test: %dx%d at %d bpp, flags: ' % \
        (bmp.w, bmp.h, bmp.format.BitsPerPixel),
    PrintFlags(bmp.flags)
    print
    then = SDL_GetTicks()
    frames = RunBlitTests(screen, bmp, NUM_BLITS)
    now = SDL_GetTicks()
    seconds = (now - then) / 1000.0
    if seconds > 0.0:
        print '%d blits / %d updates in %2.2f seconds, %2.2f FPS' % \
            (NUM_BLITS * frames, frames, seconds, frames / seconds)
    else:
        print '%d blits / %d updates in zero seconds!' % \
            (NUM_BLITS * frames, frames)
    # clear the screen after fill test
    SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, 0))
    SDL_Flip(screen)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    # run the colorkeyed blit test
    bmpcc = SDL_LoadBMP(SAMPLE_BMP)
    print 'Running freshly loaded cc blit test: %dx%d at %d bpp, flags: ' % \
        (bmpcc.w, bmpcc.h, bmpcc.format.BitsPerPixel)
    SDL_SetColorKey(bmpcc, SDL_SRCCOLORKEY | SDL_RLEACCEL, bmpcc.pixels[0])
    PrintFlags(bmpcc.flags)
    print
    then = SDL_GetTicks()
    frames = RunBlitTests(screen, bmpcc, NUM_BLITS)
    now = SDL_GetTicks()
    seconds = (now - then) / 1000.0
    if seconds > 0.0:
        print '%d cc blits / %d updates in %2.2f seconds, %2.2f FPS' % \
            (NUM_BLITS * frames, frames, seconds, frames / seconds)
    else:
        print '%d cc blits / %d updates in zero seconds!' % \
            (NUM_BLITS * frames, frames)

    # clear the screen after cc blit test
    SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, 0))
    SDL_Flip(screen)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    # run the generic blit test 
    tmp = bmp
    bmp = SDL_DisplayFormat(bmp)
    SDL_FreeSurface(tmp)
    print 'Running display format blit test: %dx%d at %d bpp, flags: ' % \
        (bmp.w, bmp.h, bmp.format.BitsPerPixel)
    PrintFlags(bmp.flags)
    print
    then = SDL_GetTicks()
    frames = RunBlitTests(screen, bmp, NUM_BLITS)
    now = SDL_GetTicks()
    seconds = (now - then) / 1000.0
    if seconds > 0.0:
        print '%d blits / %d updates in %2.2f seconds, %2.2f FPS' % \
            (NUM_BLITS * frames, frames, seconds, frames / seconds)
    else:
        print '%d blits / %d updates in zero seconds!' % \
            (NUM_BLITS * frames, frames)

    # clear the screen after blit test
    SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, 0))
    SDL_Flip(screen)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    # run the colorkeyed blit test 
    tmp = bmpcc
    bmpcc = SDL_DisplayFormat(bmp)
    SDL_FreeSurface(tmp)
    print 'Running display format blit test: %dx%d at %d bpp, flags: ' % \
        (bmpcc.w, bmpcc.h, bmpcc.format.BitsPerPixel)
    PrintFlags(bmpcc.flags)
    print
    then = SDL_GetTicks()
    frames = RunBlitTests(screen, bmpcc, NUM_BLITS)
    now = SDL_GetTicks()
    seconds = (now - then) / 1000.0
    if seconds > 0.0:
        print '%d cc blits / %d updates in %2.2f seconds, %2.2f FPS' % \
            (NUM_BLITS * frames, frames, seconds, frames / seconds)
    else:
        print '%d cc blits / %d updates in zero seconds!' % \
            (NUM_BLITS * frames, frames)

    # clear the screen after cc blit test
    SDL_FillRect(screen, None, SDL_MapRGB(screen.format, 0, 0, 0))
    SDL_Flip(screen)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    # run the cc+alpha blit test only if screen bpp > 8
    if bmp.format.BitsPerPixel > 8:
        SDL_FreeSurface(bmpcc)
        bmpcc = SDL_LoadBMP(SAMPLE_BMP)
        SDL_SetAlpha(bmpcc, SDL_SRCALPHA, 85) # 85 - 33% alpha
        SDL_SetColorKey(bmpcc, SDL_SRCCOLORKEY | SDL_RLEACCEL, bmpcc.pixels[0])
        tmp = bmpcc
        bmpcc = SDL_DisplayFormat(bmpcc)
        SDL_FreeSurface(tmp)
        print 'Running display format cc+alpha blit test: ' + \
            '%dx%d at %d bpp, flags: ' % \
            (bmpcc.w, bmpcc.h, bmpcc.format.BitsPerPixel)
        PrintFlags(bmpcc.flags)
        print
        then = SDL_GetTicks()
        frames = RunBlitTests(screen, bmpcc, NUM_BLITS)
        now = SDL_GetTicks()
        seconds = (now - then) / 1000.0
        if seconds > 0.0:
            print \
              '%d cc+alpha blits / %d updates in %2.2f seconds, %2.2f FPS' % \
                (NUM_BLITS * frames, frames, seconds, frames / seconds)
        else:
            print '%d cc+alpha blits / %d updates in zero seconds!' % \
                (NUM_BLITS * frames, frames)

    SDL_FreeSurface(bmpcc)
    SDL_FreeSurface(bmp)

    while SDL_PollEvent():
        if SDL_PollEventAndReturn().type == SDL_KEYDOWN:
            return False

    return True

def RunVideoTests():
    mode_list = [
        (640, 480, 8), (640, 480, 16), (640, 480, 32),
        (800, 600, 8), (800, 600, 16), (800, 600, 32),
        (1024, 768, 8), (1024, 768, 16), (1024, 768, 32)]
    flags_list = [
        SDL_SWSURFACE,
        SDL_SWSURFACE | SDL_FULLSCREEN,
        SDL_HWSURFACE | SDL_FULLSCREEN,
        SDL_HWSURFACE | SDL_FULLSCREEN | SDL_DOUBLEBUF]

    SDL_WM_SetCaption('SDL Video Benchmark', 'vidtest')
    SDL_ShowCursor(0)
    for mode in mode_list:
        for flags in flags_list:
            print '==================================='
            print 'Setting video mode: %dx%d at %d bpp, flags:' % mode,
            PrintFlags(flags)
            print
            screen = SDL_SetVideoMode(mode[0], mode[1], mode[2], flags)
            if screen.flags & FLAG_MASK != flags:
                print "Flags didn't match:",
                PrintFlags(screen.flags)
                print
                continue
            if not RunModeTests(screen):
                return

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)
    print 'Video driver: %s' % SDL_VideoDriverName()

    info = SDL_GetVideoInfo()
    try:
        print 'Current display: %dx%d, %d bits-per-pixel' % \
            (info.current_w, info.current_h, info.vfmt.BitsPerPixel)
    except:
        print 'Current w and h not available (incompatible version)'

    if not info.vfmt.palette:
        print '    Red Mask = 0x%.8x' % info.vfmt.Rmask
        print '    Green Mask = 0x%.8x' % info.vfmt.Gmask
        print '    Blue Mask = 0x%.8x' % info.vfmt.Bmask

    modes = SDL_ListModes(None, SDL_FULLSCREEN)
    if not modes:
        print 'No available fullscreen video modes'
    elif modes == -1:
        print 'No special fullscreen video modes'
    else:
        print 'Fullscreen video modes:'
        for mode in modes:
            print '\t%dx%dx%d' % \
                (mode.w, mode.h, info.vfmt.BitsPerPixel)
    
    if info.wm_available:
        print 'A window manager is available'
    if info.hw_available:
        print 'Hardware surfaces are available (%dK video memory)' % \
            info.video_mem
    if info.blit_hw:
        print 'Copy blits between hardware surfaces are accelerated'
    if info.blit_hw_CC:
        print 'Colorkey blits between hardware surfaces are accelerated'
    if info.blit_hw_A:
        print 'Alpha blits between hardware surfaces are acclerated'
    if info.blit_sw:
        print 'Copy bits from software surfaces to hardware sufaces ' + \
              'are accelerated'
    if info.blit_sw_CC:
        print 'Colorkey blits from software surfaces to hardware ' + \
              'surfaces are accelerated'
    if info.blit_sw_A:
        print 'Alpha blits from software surfaces to hardware surfaces ' + \
              'are accelerated'
    if info.blit_fill:
        print 'Color fills on hardware surfaces are accelerated'

    if len(sys.argv) > 1 and sys.argv[1] == '-benchmark':
        RunVideoTests()

    SDL_Quit()
