#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from copy import copy
import os
import sys

from SDL import *
from SDL.image import *

#our game object class
class GameObject:
    def __init__(self, image, height, speed):
        self.speed = speed
        self.image = image
        self.pos = SDL_Rect(0, height, image.w, image.h)
    def move(self):
        self.pos.x += self.speed
        if self.pos.x + self.pos.w > 600:
            self.pos.x = 0


#quick function to load an image
def load_image(name):
    path = os.path.join(os.path.dirname(sys.argv[0]), 
                        os.path.join('data', name))
    tmp = IMG_Load(path)
    img = SDL_DisplayFormat(tmp)
    SDL_FreeSurface(tmp)
    return img


#here's the full code
def main():
    SDL_Init(SDL_INIT_VIDEO)
    screen = SDL_SetVideoMode(640, 480, 0, 0)

    player = load_image('player1.gif')
    background = load_image('liquid.bmp')

    # scale the background image so that it fills the window and
    #   successfully overwrites the old sprite position.
    scaled = SDL_CreateRGBSurface(SDL_SWSURFACE, 
                                  background.w * 4,
                                  background.h * 4,
                                  background.format.BitsPerPixel,
                                  background.format.Rmask,
                                  background.format.Gmask,
                                  background.format.Bmask,
                                  background.format.Amask)
    scaled_buf = scaled.pixels
    scaled_pitch = scaled.pitch / scaled.format.BytesPerPixel
    print scaled_pitch
    background_buf = background.pixels
    background_pitch = background.pitch / background.format.BytesPerPixel
    print scaled.format.BytesPerPixel, background.format.BytesPerPixel
    for y in range(scaled.h):
        for x in range(scaled.w):
            scaled_buf[y * scaled_pitch + x] = \
                background_buf[(y/4) * background_pitch  + x / 4]
                                    
    SDL_FreeSurface(background)
    background = scaled
    #background = pygame.transform.scale2x(background)
    #background = pygame.transform.scale2x(background)

    SDL_BlitSurface(background, None, screen, None)

    objects = []
    for x in range(10):
        o = GameObject(player, x*40, x)
        objects.append(o)

    frames = 0
    then = SDL_GetTicks()
    while 1:
        frames += 1
        event = SDL_PollEventAndReturn()
        while event:
            if event.type in (SDL_QUIT, SDL_KEYDOWN):
                time = SDL_GetTicks() - then
                print '%d frames in %.2f seconds' % (frames, time / 1000.0)
                print '  %f milliseconds per frame (%d FPS)' % \
                    (float(time) / frames, frames * 1000 / time)
                return
            event = SDL_PollEventAndReturn()

        for o in objects:
            SDL_BlitSurface(background, copy(o.pos), screen, copy(o.pos))
        for o in objects:
            o.move()
            SDL_BlitSurface(o.image, None, screen, copy(o.pos))

        SDL_UpdateRect(screen, 0, 0, 0, 0)



if __name__ == '__main__': main()
