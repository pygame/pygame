#!/usr/bin/env python

'''Bring up a window and manipulate the gamma on it.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys
import os

from SDL import *

SAMPLE_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'sample.bmp')

def CalculateGamma(gamma, ramp):
    gamma = 1.0 / gamma
    ramp[0] = 0     # Can't raise 0 to negative number
    for i in range(1, 256):
        value = int(((i/256.0)**gamma) * 65535.0 + 0.5)
        value = min(value, 65535)
        ramp[i] = value

# This can be used as a general routine for all of the test programs
def get_video_args():
    w = 640
    h = 480
    bpp = 0
    flags = SDL_SWSURFACE
    args = []

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-width':
            i += 1
            w = int(sys.argv[i])
        elif sys.argv[i] == '-height':
            i += 1
            h = int(sys.argv[i])
        elif sys.argv[i] == '-bpp':
            i += 1
            bpp = int(sys.argv[i])
        elif sys.argv[i] == '-fullscreen':
            flags |= SDL_FULLSCREEN
        elif sys.argv[i] == '-hw':
            flags |= SDL_HWSURFACE
        elif sys.argv[i] == '-hwpalette':
            flags |= SDL_HWPALETTE
        else:
            args = sys.argv[i:]
            break
    return w, h, bpp, flags, args

def main():
    w, h, bpp, flags, args = get_video_args()

    SDL_Init(SDL_INIT_VIDEO)
    screen = SDL_SetVideoMode(w, h, bpp, flags)
    SDL_WM_SetCaption('SDL gamma test', 'testgamma')

    gamma = 1.0
    if args:
        gamma = float(args[0])
    SDL_SetGamma(gamma, gamma, gamma)

    image = SDL_LoadBMP(SAMPLE_BMP)
    dst = SDL_Rect()
    dst.x = (screen.w - image.w) / 2
    dst.y = (screen.h - image.h) / 2
    dst.w = image.w
    dst.h = image.h
    SDL_BlitSurface(image, None, screen, dst)
    SDL_UpdateRects(screen, [dst])

    then = SDL_GetTicks()
    timeout = 5 * 1000
    while SDL_GetTicks() - then < timeout:
        event = SDL_PollEventAndReturn()
        while event:
            if event.type == SDL_QUIT:
                timeout = 0
            elif event.type == SDL_KEYDOWN:
                key = event.keysym.sym
                if key == SDLK_SPACE:   # Go longer..
                    timeout += 5*1000
                elif key == SDLK_UP:
                    gamma += 0.2
                    # XXX This has no effect on my nvidia, only
                    # SDL_SetGammaRamp works as expected.
                    #SDL_SetGamma(gamma, gamma, gamma)
                    ramp = [0] * 256
                    CalculateGamma(gamma, ramp)
                    SDL_SetGammaRamp(ramp, ramp, ramp)
                elif key == SDLK_DOWN:
                    gamma -= 0.2
                    # XXX SDL_SetGamma(gamma, gamma, gamma) 
                    ramp = [0] * 256
                    CalculateGamma(gamma, ramp)
                    SDL_SetGammaRamp(ramp, ramp, ramp)
                elif key == SDLK_ESCAPE:
                    timeout = 0
            event = SDL_PollEventAndReturn()

    while gamma < 10:
        # Increase red gamma and decrease everything else
        gamma += 0.1
        red_ramp = [0] * 256
        ramp = [0] * 256
        CalculateGamma(gamma, red_ramp)
        CalculateGamma(1.0/gamma, ramp)
        SDL_SetGammaRamp(red_ramp, ramp, ramp)
        SDL_Delay(10) # Delay inserted here so fade is visible

    # Finish completely red
    red_ramp = [65535] * 256
    ramp = [0] * 256
    SDL_SetGammaRamp(red_ramp, ramp, ramp)

    # Fade out to black
    for i in range(255, 0, -1):
        red_ramp = [i*255] * 256
        SDL_SetGammaRamp(red_ramp, ramp, ramp)
        SDL_Delay(10) # Delay inserted here so fade is visible
    SDL_Delay(1000)

    SDL_Quit()

if __name__ == '__main__':
    main()
