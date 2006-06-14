#!/usr/bin/env python

'''Simple program: Fill a colormap with gray and stripe it down the screen,
then move an alpha valued sprite around the screen.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from copy import copy
import math
import os
import sys

from SDL import *

ICON_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'icon.bmp')
FRAME_TICKS = 1000/30

# Fill the screen with a gradient
def FillBackground(screen):
    SDL_LockSurface(screen)
    if screen.format.BytesPerPixel != 2:
        buffer = screen.pixels.as_bytes()
        for i in range(screen.h):
            buffer[screen.pitch*i:screen.pitch*(i+1)] = \
                [(i*(255))/screen.h] * screen.pitch
    else:
        buffer = screen.pixels
        for i in range(screen.h):
            gradient = i*(255)/screen.h
            color = SDL_MapRGB(screen.format, gradient, gradient, gradient)
            buffer[screen.w*i:screen.w*(i+1)] = [color] * screen.w

    SDL_UnlockSurface(screen)
    SDL_UpdateRect(screen, 0, 0, 0, 0)

# Create a "light" -- a yellowish surface with variable alpha
def CreateLight(radius):
    # 32 bit only atm.
    alphamask = 0x000000ff
    light = SDL_CreateRGBSurface(SDL_SWSURFACE, 2*radius, 2*radius, 32,
        0xff000000, 0x00ff0000, 0x0000ff00, alphamask)
    
    # Fill with a light yellow-orange color
    buf = light.pixels

    # Get a transparent pixel value - we'll add alpha later
    pixel = SDL_MapRGBA(light.format, 0xff, 0xdd, 0x88, 0)
    buf[:] = [pixel] * len(buf)

    # Calculate alpha values for the surface
    for y in range(light.h):
        for x in range(light.w):
            xdist = x - light.w / 2
            ydist = y - light.h / 2
            rng = math.sqrt(xdist**2 + ydist**2)

            # Scale distance to range of transparence (0-255)
            if rng > radius:
                trans = alphamask
            else:
                # Increasing transparency with distance
                trans = int((rng * alphamask) / radius) & 0xff

                # Lights are very transparent
                addition = (alphamask+1)/8
                if trans + addition > alphamask:
                    trans = alphamask
                else:
                    trans += addition
            # We set the alpha component as the right N bits
            buf[y * light.w + x] |= 255 - trans

    # Enable RLE acceleration of this alpha surface
    SDL_SetAlpha(light, SDL_SRCALPHA | SDL_RLEACCEL, 0)

    return light

flashes = 0
flashtime = 0

def FlashLight(screen, light, x, y):
    global flashes
    global flashtime

    position = SDL_Rect()

    # Easy, center light
    position.x = x - light.w / 2
    position.y = y - light.h / 2
    position.w = light.w
    position.h = light.h
    ticks1 = SDL_GetTicks()
    SDL_BlitSurface(light, None, screen, position)
    ticks2 = SDL_GetTicks()
    SDL_UpdateRects(screen, [position])
    flashes += 1

    # Update time spend doing alpha blitting
    flashtime += ticks2 - ticks1

sprite_visible = 0
sprite = None
backing = None
position = SDL_Rect()
x_vel = y_vel = 0
alpha_vel = 0

def LoadSprite(screen, file):
    global sprite
    global backing

    # Load the sprite image
    sprite = SDL_LoadBMP(file)

    # Set transparent pixel as the pixel at (0,0)
    if sprite.format.palette:
        SDL_SetColorKey(sprite, SDL_SRCCOLORKEY, sprite.pixels.as_bytes()[0])

    # Convert sprite to video format
    converted = SDL_DisplayFormat(sprite)
    SDL_FreeSurface(sprite)
    sprite = converted

    # Create the background
    backing = SDL_CreateRGBSurface(SDL_SWSURFACE, sprite.w, sprite.h, 8, 
        0, 0, 0, 0)
    converted = SDL_DisplayFormat(backing)
    SDL_FreeSurface(backing)
    backing = converted

    # Set the initial position of the sprite
    position.x = (screen.w - sprite.w) / 2
    position.y = (screen.h - sprite.h) / 2
    position.w = sprite.w
    position.h = sprite.h
    global x_vel
    global y_vel
    global alpha_vel
    x_vel = 0
    y_vel = 0
    alpha_vel = 1

def AttractSprite(x, y):
    global x_vel
    global y_vel
    x_vel = (x - position.x) / 10
    y_vel = (y - position.y) / 10

def MoveSprite(screen, light):
    global sprite_visible
    updates = [SDL_Rect(), SDL_Rect()]

    if sprite_visible:
        updates[0] = copy(position)
        SDL_BlitSurface(backing, None, screen, updates[0])
    else:
        updates[0].x = 0
        updates[0].y = 0
        updates[0].w = 0
        updates[0].h = 0
        sprite_visible = 1

    # Since the sprite is off the screen, we can do other drawing without
    # being overwritten by the saved area behind the sprite
    if light:
        state, x, y = SDL_GetMouseState()
        FlashLight(screen, light, x, y)

    # Move the sprite, bounce at the wall
    global x_vel
    global y_vel
    position.x += x_vel
    if position.x < 0 or position.x >= screen.w:
        x_vel = -x_vel
        position.x += x_vel
    position.y += y_vel
    if position.y < 0 or position.y >= screen.h:
        y_vel = -y_vel
        position.y += y_vel

    # Update transparency (fade in and out)
    global alpha_vel
    alpha = sprite.format.alpha
    if alpha + alpha_vel < 0:
        alpha_vel = -alpha_vel
    elif alpha + alpha_vel > 255:
        alpha_vel = -alpha_vel
    SDL_SetAlpha(sprite, SDL_SRCALPHA, int(alpha + alpha_vel) & 0xff)

    # Save the area behind the sprite
    updates[1] = copy(position)
    SDL_BlitSurface(screen, updates[1], backing, None)

    # Blit the sprite onto the screen
    updates[1] = copy(position)
    SDL_BlitSurface(sprite, None, screen, updates[1])

    # Make it so!
    SDL_UpdateRects(screen, updates)

def WarpSprite(screen, x, y):
    updates = [SDL_Rect(), SDL_Rect()]
    updates[0] = copy(position)
    SDL_BlitSurface(backing, None, updates[0])
    position.x = x - sprite.w / 2 # Center about X
    position.y = y - sprite.h / 2 # Center about Y
    updates[1] = copy(position)
    SDL_BlitSurface(screen, updates[1], backing, None)
    updates[1] = copy(position)
    SDL_BlitSurface(sprite, None, screen, updates[1])
    SDL_UpdateRects(screen, updates)

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)
    w = 640
    h = 480
    info = SDL_GetVideoInfo()
    if info.vfmt.BitsPerPixel > 8:
        video_bpp = info.vfmt.BitsPerPixel
    else:
        video_bpp = 16
        print >> sys.stderr, 'forced 16 bpp mode'
    videoflags = SDL_SWSURFACE
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-width':
            i += 1
            w = int(sys.argv[i])
        elif sys.argv[i] == '-height':
            i += 1
            h = int(sys.argv[i])
        elif sys.argv[i] == '-bpp':
            i += 1
            bpp = int(sys.argv[i])
        elif sys.argv[i] == '-hw':
            videoflags |= SDL_HWSURFACE
        elif sys.argv[i] == '-warp':
            videoflags |= SDL_HWPALETTE
        elif sys.argv[i] == '-noframe':
            videoflags |= SDL_NOFRAME
        elif sys.argv[i] == '-resize':
            videoflags |= SDL_RESIZABLE
        elif sys.argv[i] == '-fullscreen':
            videoflags |= SDL_FULLSCREEN
        else:
            print >> sys.stderr, ('Usage: %s [-width N] [-height N] ' + \
                '[-bpp N] [-hw] [-warp] [-noframe] [-fullscreen] ' + \
                '[-resize]') % sys.argv[0]
            sys.exit(1)
        i += 1

    # Set video mode
    screen = SDL_SetVideoMode(w, h, video_bpp, videoflags)
    
    FillBackground(screen)

    # Create the light
    light = CreateLight(82)
    
    # Load the sprite
    LoadSprite(screen, ICON_BMP)

    # Print out information about our surfaces
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
    MoveSprite(screen, None)
    if sprite.flags & SDL_HWACCEL:
        print 'Sprite blit uses hardware alpha acceleration'
    else:
        print 'Sprite blit doesn\'t uses hardware alpha acceleration'


    # Set a clipping rectangle to clip the outside edge of the screen
    clip = SDL_Rect(32, 32, screen.w - 2 * 32, screen.h - 2 * 32)
    SDL_SetClipRect(screen, clip)

    # Wait for a keystroke
    lastticks = SDL_GetTicks()
    done = False
    mouse_pressed = False
    while not done:
        if mouse_pressed:
            MoveSprite(screen, light)
            mouse_pressed = False
        else:
            MoveSprite(screen, None)

        # Slow down the loop to 30 frames/second
        ticks = SDL_GetTicks()
        if ticks - lastticks < FRAME_TICKS:
            SDL_Delay(FRAME_TICKS - (ticks - lastticks))
        lastticks = ticks

        while SDL_PollEvent():
            event = SDL_PollEventAndReturn()
            if event.type == SDL_VIDEORESIZE:
                screen = SDL_SetVideoMode(event.w, event.h, video_bpp,
                    videoflags)
                FillBackground(screen)
            elif event.type == SDL_MOUSEMOTION:
                if event.state != 0:
                    AttractSprite(event.x, event.y)
                    mouse_pressed = True
            elif event.type == SDL_MOUSEBUTTONDOWN:
                if event.button == 1:
                    AttractSprite(event.x, event.y)
                    mouse_pressed = True
                else:
                    area = SDL_Rect(event.x - 16, event.y - 16, 32, 32)
                    SDL_FillRect(screen, area, 0)
                    SDL_UpdateRects(screen, [area])
            elif event.type == SDL_KEYDOWN:
                if event.keysym.sym == SDLK_ESCAPE:
                    done = True
            elif event.type == SDL_QUIT:
                done = True
    
    SDL_FreeSurface(light)
    SDL_FreeSurface(sprite)
    SDL_FreeSurface(backing)


    if flashes > 0:
        print '%d alpha blits, ~%4.4f ms per blit' % \
            (flashes, float(flashtime)/flashes)

    SDL_Quit()
