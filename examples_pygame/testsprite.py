#!/usr/bin/env python
# like the testsprite.c that comes with sdl, this pygame version shows 
#   lots of sprites moving around.


import pygame, sys, os
from pygame.locals import *
from random import randint
from time import time
import pygame.joystick

if "-psyco" in sys.argv:
    try:
        import psyco
        psyco.full()
    except Exception:
        pass


# use this to use update rects or not.
#  If the screen is mostly full, then update rects are not useful.
update_rects = True
if "-update_rects" in sys.argv:
    update_rects = True
if "-noupdate_rects" in sys.argv:
    update_rects = False

flags = 0
if "-flip" in sys.argv:
    flags ^= DOUBLEBUF

if "-fullscreen" in sys.argv:
    flags ^= FULLSCREEN

if "-sw" in sys.argv:
    flags ^= SWSURFACE

use_rle = True

if "-hw" in sys.argv:
    flags ^= HWSURFACE
    use_rle = False


screen_dims = [640, 480]

if "-height" in sys.argv:
    i = sys.argv.index("-height")
    screen_dims[1] = int(sys.argv[i+1])

if "-width" in sys.argv:
    i = sys.argv.index("-width")
    screen_dims[0] = int(sys.argv[i+1])


print screen_dims


class Thingy(pygame.sprite.Sprite):
    images = None
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = Thingy.images[0]
        self.rect = self.image.get_rect()
        self.rect.x = randint(0, screen_dims[0])
        self.rect.y = randint(0, screen_dims[1])
        #self.vel = [randint(-10, 10), randint(-10, 10)]
        self.vel = [randint(-1, 1), randint(-1, 1)]

    def move(self):
        for i in [0, 1]:
            nv = self.rect[i] + self.vel[i]
            if nv >= screen_dims[i] or nv < 0:
                self.vel[i] = -self.vel[i]
                nv = self.rect[i] + self.vel[i]
            self.rect[i] = nv



def main():
    global update_rects, flags
    #pygame.init()
    pygame.display.init()



    #if "-fast" in sys.argv:

    screen = pygame.display.set_mode(screen_dims, flags)


    # this is mainly for GP2X, so it can quit.
    pygame.joystick.init()
    num_joysticks = pygame.joystick.get_count()
    if num_joysticks > 0:
        stick = pygame.joystick.Joystick(0)
        stick.init() # now we will receive events for the joystick


    screen.fill([0,0,0])
    pygame.display.flip()
    sprite_surface = pygame.image.load(os.path.join("data", "asprite.bmp"))

    if use_rle:
        sprite_surface.set_colorkey([0xFF, 0xFF, 0xFF], SRCCOLORKEY|RLEACCEL)
    else:
        sprite_surface.set_colorkey([0xFF, 0xFF, 0xFF], SRCCOLORKEY)


    sprite_surface = sprite_surface.convert()
    Thingy.images = [sprite_surface]

    if len(sys.argv) > 1:
        try:
            numsprites = int(sys.argv[-1])
        except:
            numsprites = 100
    else:
        numsprites = 100
    if update_rects:
        sprites = pygame.sprite.RenderUpdates()
    else:
        sprites = pygame.sprite.Group()

    for i in xrange(0, numsprites):
        sprites.add(Thingy())

    done = False
    frames = 0
    start = time()

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill([0,0,0])


    while not done:
        if not update_rects:
            screen.fill([0,0,0])

        for sprite in sprites:
            sprite.move()

        if update_rects:
            sprites.clear(screen, background)
        sprites.update()

        rects = sprites.draw(screen)
        if update_rects:
            pygame.display.update(rects)
        else:
            pygame.display.flip()


        for event in pygame.event.get():
            if event.type in [KEYDOWN, QUIT, JOYBUTTONDOWN]:
                done = True


        frames += 1
    end = time()
    print "FPS: %f" % (frames / ((end - start)))



if __name__ == "__main__":
    main()
