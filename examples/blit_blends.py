#!/usr/bin/env python

# fake additive blending.  Using Numeric.  it doesn't clamp.
# press r,g,b

import os, pygame
from pygame.locals import *

try:
    import pygame.surfarray
    import Numeric
except:
    print ("no surfarray for you!  install Numeric")

import time
        
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

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

    images = {}
    images[K_1] = im2
    images[K_2] = pygame.image.load(os.path.join(data_dir, "chimp.bmp"))
    images[K_3] = pygame.image.load(os.path.join(data_dir, "alien3.gif"))
    images[K_4] = pygame.image.load(os.path.join(data_dir, "liquid.bmp"))
    img_to_blit = im2.convert()
    iaa = img_to_blit.convert_alpha()



    blits = {}
    blits[K_a] = BLEND_ADD
    blits[K_s] = BLEND_SUB
    blits[K_m] = BLEND_MULT
    blits[K_EQUALS] = BLEND_MAX
    blits[K_MINUS] = BLEND_MIN

    blitsn = {}
    blitsn[K_a] = "BLEND_ADD"
    blitsn[K_s] = "BLEND_SUB"
    blitsn[K_m] = "BLEND_MULT"
    blitsn[K_EQUALS] = "BLEND_MAX"
    blitsn[K_MINUS] = "BLEND_MIN"


    screen.blit(im1, (0, 0))
    pygame.display.flip()
    clock = pygame.time.Clock()
    print ("one pixel is:%s:" % [im1.get_at((0,0))])

    going = True
    while going:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                going = False
            if event.type == KEYDOWN:
                usage()

            if event.type == KEYDOWN and event.key == K_ESCAPE:
                going = False

            elif event.type == KEYDOWN and event.key in images.keys():
                img_to_blit = images[event.key]
                iaa = img_to_blit.convert_alpha()

            elif event.type == KEYDOWN and event.key in blits.keys():
                t1 = time.time()
                # blits is a dict keyed with key -> blit flag.  eg BLEND_ADD.
                im1.blit(img_to_blit, (0,0), None, blits[event.key])
                t2 = time.time()
                print ("one pixel is:%s:" % [im1.get_at((0,0))])
                print ("time to do:%s:" % (t2-t1))


            elif event.type == KEYDOWN and event.key in [K_t]:

                for bkey in blits.keys():
                    t1 = time.time()

                    for x in range(300):
                        im1.blit(img_to_blit, (0,0), None, blits[bkey])

                    t2 = time.time()

                    # show which key we're doing...
                    onedoing = blitsn[bkey]
                    print ("time to do :%s: is :%s:" % (onedoing, t2-t1))


            elif event.type == KEYDOWN and event.key in [K_o]:
                t1 = time.time()
                # blits is a dict keyed with key -> blit flag.  eg BLEND_ADD.
                im1.blit(iaa, (0,0))
                t2 = time.time()
                print ("one pixel is:%s:" % [im1.get_at((0,0))])
                print ("time to do:%s:" % (t2-t1))


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
                print ("one pixel is:%s:" % [im1.get_at((0,0))])
                print ("time to do:%s:" % (t2-t1))

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
                print ("one pixel is:%s:" % [im1.get_at((0,0))])
                print ("time to do:%s:" % (t2-t1))

            elif event.type == KEYDOWN and event.key in [K_r, K_g, K_b]:
                # this adds one to each pixel.
                colmap={}
                colmap[K_r] = 0x10000
                colmap[K_g] = 0x00100
                colmap[K_b] = 0x00001
                im1p = pygame.surfarray.pixels2d(im1)
                im1p += colmap[event.key]
                del im1p
                print ("one pixel is:%s:" % [im1.get_at((0,0))])

            elif event.type == KEYDOWN and event.key == K_p:
                print ("one pixel is:%s:" % [im1.get_at((0,0))])





            elif event.type == KEYDOWN and event.key == K_f:
                # this additive blend without clamp two surfaces.

                t1 = time.time()
                im1.set_alpha(127)
                im1.blit(im2, (0,0))
                im1.set_alpha(255)

                t2 = time.time()
                print ("one pixel is:%s:" % [im1.get_at((0,0))])
                print ("time to do:%s:" % (t2-t1))


        screen.blit(im1, (0, 0))
        pygame.display.flip()

    pygame.quit()

def usage():
    print ("press keys 1-5 to change image to blit.")
    print ("A - ADD, S- SUB, M- MULT, - MIN, + MAX")
    print ("T - timing test for special blend modes.")

if __name__ == '__main__': 
    usage()
    main()
