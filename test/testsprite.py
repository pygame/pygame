#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from copy import copy
import os
import random
import sys

from SDL import *

ICON_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'icon.bmp')
NUM_SPRITES = 100
MAX_SPEED = 1

sprite = None
numsprites = 0
sprite_rects = None
positions = None
velocities = None
sprites_visible = 0
debug_flip = 0
sprite_w = 0
sprite_h = 0

def LoadSprite(file):
    global sprite
    sprite = SDL_LoadBMP(file)

    if sprite.format.palette:
        SDL_SetColorKey(sprite, (SDL_SRCCOLORKEY | SDL_RLEACCEL), 
                        sprite.pixels[0])

    temp = SDL_DisplayFormat(sprite)
    SDL_FreeSurface(sprite)
    sprite = temp

t = 0
def MoveSprites(screen, background):
    global t, sprite_rects, positions, velocities, sprites_visible

    if sprites_visible:
        SDL_FillRect(screen, None, background)

    sprite_rects = []
    for i in range(numsprites):
        position = positions[i]
        velocity = velocities[i]
        position.x += velocity.x
        if position.x < 0 or position.x >= (screen.w - sprite_w):
            velocity.x = -velocity.x
            position.x += velocity.x
        position.y += velocity.y
        if position.y < 0 or position.y >= (screen.h - sprite_h):
            velocity.y = -velocity.y
            position.y += velocity.y
        area = copy(position)
        SDL_BlitSurface(sprite, None, screen, area)
        sprite_rects.append(area)

    if debug_flip:
        if screen.flags & SDL_DOUBLEBUF:
            color = SDL_MapRGB(screen.format, 255, 0, 0)
            r = SDL_Rect((math.sin(t*2*math.pi) + 1) / 2 * (screen.w - 20),
                         0, 20, screen.h)
            SDL_FillRect(screen, r, color)
            t += 2

    if screen.flags & SDL_DOUBLEBUF:
        SDL_Flip(screen)
    else:
        SDL_UpdateRects(screen, sprite_rects)
    sprites_visible = 1

def FastestFlags(flags, width, height, bpp):
    flags |= SDL_FULLSCREEN

    info = SDL_GetVideoInfo()
    if info.blit_hw_CC and info.blit_fill:
        flags |= SDL_HWSURFACE

    if flags & SDL_HWSURFACE:
        if info.video_mem*1024 > height * width * bpp / 8:
            flags |= SDL_DOUBLEBUF
        else:
            flags &= ~SDL_HWSURFACE

    return flags

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    numsprites = NUM_SPRITES
    videoflags = SDL_SWSURFACE | SDL_ANYFORMAT
    width = 640
    height = 480
    video_bpp = 8
    debug_flip = 0

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-width':
            i += 1
            width = int(sys.argv[i])
        elif arg == '-height':
            i += 1
            height = int(sys.argv[i])
        elif arg == '-bpp':
            i += 1
            video_bpp = int(sys.argv[i])
            videoflags &= ~SDL_ANYFORMAT
        elif arg == '-fast':
            videoflags = FastestFlags(videoflags, width, height, video_bpp)
        elif arg == '-hw':
            videoflags ^= SDL_HWSURFACE
        elif arg == '-flip':
            videoflags ^= SDL_DOUBLEBUF
        elif arg == '-debugflip':
            debug_flip ^= 1
        elif arg == '-fullscreen':
            videoflags ^= SDL_FULLSCREEN
        elif arg[0].isdigit():
            numsprites = int(arg)
        else:
            print >> sys.stderr, ('Usage: %s [-bpp N] [-hw] [-flip] ' + \
                '[-fast] [-fullscreen] [numsprites]') % sys.argv[0]
            sys.exit(1)
        i += 1

    screen = SDL_SetVideoMode(width, height, video_bpp, videoflags)

    LoadSprite(ICON_BMP)
    
    sprite_w = sprite.w
    sprite_h = sprite.h
    positions = []
    velocities = []
    for i in range(numsprites):
        positions.append(SDL_Rect(random.randint(0, screen.w - sprite.w),
                                  random.randint(0, screen.h - sprite.h),
                                  sprite.w, sprite.h))
        velocities.append(SDL_Rect(random.randint(0,2)*2-1,
                                   random.randint(0,2)*2-1,
                                   0, 0))
    background = SDL_MapRGB(screen.format, 0, 0, 0)

    print 'Screen is at %d bits per pixel' % screen.format.BitsPerPixel
    if screen.flags & SDL_HWSURFACE:
        print 'Screen is in video memory'
    else:
        print 'Screen is in system memory'
    if screen.flags & SDL_DOUBLEBUF:
        print 'Screen has double-buffering enabled'
    if sprite.flags & SDL_HWSURFACE:
        print 'Sprite is in video memory'
    else:
        print 'Sprite is in system memory'

    # Run a sample blit to trigger blit acceleration
    dst = SDL_Rect(0, 0, sprite.w, sprite.h)
    SDL_BlitSurface(sprite, None, screen, dst)
    SDL_FillRect(screen, dst, background)

    if sprite.flags & SDL_HWACCEL:
        print 'Sprite blit uses hardware acceleration'
    if sprite.flags & SDL_RLEACCEL:
        print 'Sprite blit uses RLE acceleration'

    frames = 0
    then = SDL_GetTicks()
    done = 0
    sprites_visible = 0
    while not done:
        frames += 1
        event = SDL_PollEventAndReturn()
        while event:
            if event.type == SDL_MOUSEBUTTONDOWN:
                SDL_WarpMouse(screen.w/2, screen.h/2)
            elif event.type in (SDL_KEYDOWN, SDL_QUIT):
                done = 1
            event = SDL_PollEventAndReturn()
        MoveSprites(screen, background)

    SDL_FreeSurface(sprite)
    now = SDL_GetTicks()
    if now > then:
        print '%2.2f frames per second' % (frames * 1000.0 / (now - then))

    SDL_Quit()
