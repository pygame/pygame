#!/usr/bin/env python

# fake additive blending.  Using Numeric.  it doesn't clamp.
# press r,g,b

import os, pygame
from pygame.locals import *

import pygame.surfarray
import Numeric
import time
        

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))

    im1= pygame.Surface(screen.get_size())
    #im1= im1.convert()
    im1.fill((100, 0, 0))

    

    im2= pygame.Surface(screen.get_size())
    im2.fill((0, 50, 0))
    # we make a srcalpha copy of it.
    #im3= im2.convert(SRCALPHA)
    im3 = im2
    im3.set_alpha(127)


    screen.blit(im1, (0, 0))
    pygame.display.flip()
    clock = pygame.time.Clock()
    print "one pixel is:%s:" % [im1.get_at((0,0))]
    
    while 1:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            elif event.type == KEYDOWN and event.key == K_SPACE:
                # this additive blend without clamp two surfaces.
                #im1.set_alpha(127)
                #im1.blit(im1, (0,0))
                #im1.set_alpha(255)
                t1 = time.time()

                im1p = pygame.surfarray.pixels2d(im1)
                im2p = pygame.surfarray.pixels2d(im2)
                im1p += im2p
                del im1p
                del im2p
                t2 = time.time()
                print "one pixel is:%s:" % [im1.get_at((0,0))]
                print "time to do:%s:" % (t2-t1)

            elif event.type == KEYDOWN and event.key in [K_z]:
                t1 = time.time()
                im1p = pygame.surfarray.pixels3d(im1)
                im2p = pygame.surfarray.pixels3d(im2)
                im1p16 = im1p.astype(Numeric.UInt16)
                im2p16 = im1p.astype(Numeric.UInt16)
                im1p16 += im2p16
                im1p16 = Numeric.minimum(im1p16, 255)
                pygame.surfarray.blit_array(im1, im1p16)

                del im1p
                del im2p
                t2 = time.time()
                print "one pixel is:%s:" % [im1.get_at((0,0))]
                print "time to do:%s:" % (t2-t1)

            elif event.type == KEYDOWN and event.key in [K_r, K_g, K_b]:
                # this adds one to each pixel.
                colmap={}
                colmap[K_r] = 0x10000
                colmap[K_g] = 0x00100
                colmap[K_b] = 0x00001
                im1p = pygame.surfarray.pixels2d(im1)
                im1p += colmap[event.key]
                del im1p
                print "one pixel is:%s:" % [im1.get_at((0,0))]

            elif event.type == KEYDOWN and event.key == K_p:
                print "one pixel is:%s:" % [im1.get_at((0,0))]


            elif event.type == KEYDOWN and event.key in [K_a]:
                t1 = time.time()
                im1.blit(im2, (0,0), im1.get_rect(), BLEND_ADD)
                t2 = time.time()
                print "one pixel is:%s:" % [im1.get_at((0,0))]
                print "time to do:%s:" % (t2-t1)


            elif event.type == KEYDOWN and event.key in [K_s]:
                t1 = time.time()
                im1.blit(im2, (0,0), im1.get_rect(), BLEND_SUB)
                t2 = time.time()
                print "one pixel is:%s:" % [im1.get_at((0,0))]
                print "time to do:%s:" % (t2-t1)


            elif event.type == KEYDOWN and event.key == K_f:
                # this additive blend without clamp two surfaces.

                t1 = time.time()
                im1.set_alpha(127)
                im1.blit(im2, (0,0))
                im1.set_alpha(255)

                t2 = time.time()
                print "one pixel is:%s:" % [im1.get_at((0,0))]
                print "time to do:%s:" % (t2-t1)


        screen.blit(im1, (0, 0))
        pygame.display.flip()



if __name__ == '__main__': main()
