#!/usr/bin/env python

'''Simple program: Test bitmap blits
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *

# From picture.xbm
picture_width = 32
picture_height = 32
picture_bits = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x80,
    0x01, 0x18, 0x64, 0x6f, 0xf6, 0x26, 0x0a, 0x00, 0x00, 0x50, 0xf2, 0xff,
    0xff, 0x4f, 0x14, 0x04, 0x00, 0x28, 0x14, 0x0e, 0x00, 0x28, 0x10, 0x32,
    0x00, 0x08, 0x94, 0x03, 0x00, 0x08, 0xf4, 0x04, 0x00, 0x08, 0xb0, 0x08,
    0x00, 0x08, 0x34, 0x01, 0x00, 0x28, 0x34, 0x01, 0x00, 0x28, 0x12, 0x00,
    0x40, 0x48, 0x12, 0x20, 0xa6, 0x48, 0x14, 0x50, 0x11, 0x29, 0x14, 0x50,
    0x48, 0x2a, 0x10, 0x27, 0xac, 0x0e, 0xd4, 0x71, 0xe8, 0x0a, 0x74, 0x20,
    0xa8, 0x0a, 0x14, 0x20, 0x00, 0x08, 0x10, 0x50, 0x00, 0x08, 0x14, 0x00,
    0x00, 0x28, 0x14, 0x00, 0x00, 0x28, 0xf2, 0xff, 0xff, 0x4f, 0x0a, 0x00,
    0x00, 0x50, 0x64, 0x6f, 0xf6, 0x26, 0x18, 0x80, 0x01, 0x18, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

def LoadXBM(screen, w, h, bits):
    bitmap = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 1, 0, 0, 0, 0)
    
    # Reverse bits in each bit
    # (from http://graphics.stanford.edu/~seander/bithacks.html)
    bytes = [((b * 0x0202020202 & 0x010884422010) % 1023) & 0xff \
              for b in picture_bits]

    buffer = bitmap.pixels
    pitch = bitmap.pitch * 8 / bitmap.format.BitsPerPixel
    for y in range(h):
        buffer[y*pitch:y*pitch+w] = bytes[y*w:(y+1)*w]
    
    return bitmap

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    video_bpp = 0
    videoflags = SDL_SWSURFACE
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-bpp':
            i += 1
            bpp = int(sys.argv[i])
        elif sys.argv[i] == '-hw':
            videoflags |= SDL_HWSURFACE
        elif sys.argv[i] == '-warp':
            videoflags |= SDL_HWPALETTE
        elif sys.argv[i] == '-fullscreen':
            videoflags |= SDL_FULLSCREEN
        else:
            print >> sys.stderr, \
                'Usage: %s [-bpp N] [-warp] [-hw] [-fullscreen]' % sys.argv[0]
        i += 1
    screen = SDL_SetVideoMode(640, 480, video_bpp, videoflags)

    if video_bpp == 8:
        # Set a gray colormap, reverse order from white to black
        palette = [SDL_Color(255-i, 255-i, 255-i) for i in range(256)]
        SDL_SetColors(screen, palette, 0)

    SDL_LockSurface(screen)
    buffer = screen.pixels
    if screen.format.BytesPerPixel != 2:
        buffer = screen.pixels.as_bytes()
        for i in range(screen.h):
            buffer[screen.pitch*i:screen.pitch*(i+1)] = \
                [(i*255)/screen.h] * screen.pitch
    else:
        buffer = screen.pixels
        for i in range(screen.h):
            gradient = i*255/screen.h
            color = SDL_MapRGB(screen.format, gradient, gradient, gradient)
            buffer[screen.w*i:screen.w*(i+1)] = [color] * screen.w

    SDL_UnlockSurface(screen)
    SDL_UpdateRect(screen, 0, 0, 0, 0)

    # Load the bitmap
    bitmap = LoadXBM(screen, picture_width, picture_height, picture_bits)

    # Wait for a keystroke
    done = False
    while not done:
        event = SDL_WaitEventAndReturn()
        if event.type == SDL_MOUSEBUTTONDOWN:
            dst = SDL_Rect(event.x - bitmap.w / 2, event.y - bitmap.h / 2,
                           bitmap.w, bitmap.h)
            SDL_BlitSurface(bitmap, None, screen, dst)
            SDL_UpdateRects(screen, [dst])
        elif event.type in (SDL_KEYDOWN, SDL_QUIT):
            done = True

    SDL_FreeSurface(bitmap)
    SDL_Quit()

