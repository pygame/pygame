#!/usr/bin/env python
""" pygame.examples.blit_blends

Blending colors in different ways with different blend modes.

It also shows some tricks with the surfarray.
Including how to do additive blending.

Keyboard Controls:

* R, G, B
* A - Add blend mode
* S - Subtractive blend mode
* M - Multiply blend mode
* = key BLEND_MAX blend mode.
* - key BLEND_MIN blend mode.
* 1, 2, 3, 4 - use different images.

"""
import os
import pygame


try:
    import pygame.surfarray
    import numpy
except ImportError:
    print("no surfarray for you!  install numpy")

import time

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")


def main():
    pygame.init()
    pygame.mixer.quit()  # remove ALSA underflow messages for Debian squeeze
    screen = pygame.display.set_mode((640, 480))

    im1 = pygame.Surface(screen.get_size())
    # im1= im1.convert()
    im1.fill((100, 0, 0))

    im2 = pygame.Surface(screen.get_size())
    im2.fill((0, 50, 0))
    # we make a srcalpha copy of it.
    # im3= im2.convert(SRCALPHA)
    im3 = im2
    im3.set_alpha(127)

    images = {}
    images[pygame.K_1] = im2
    images[pygame.K_2] = pygame.image.load(os.path.join(data_dir, "chimp.bmp"))
    images[pygame.K_3] = pygame.image.load(os.path.join(data_dir, "alien3.gif"))
    images[pygame.K_4] = pygame.image.load(os.path.join(data_dir, "liquid.bmp"))
    img_to_blit = im2.convert()
    iaa = img_to_blit.convert_alpha()

    blits = {}
    blits[pygame.K_a] = pygame.BLEND_ADD
    blits[pygame.K_s] = pygame.BLEND_SUB
    blits[pygame.K_m] = pygame.BLEND_MULT
    blits[pygame.K_EQUALS] = pygame.BLEND_MAX
    blits[pygame.K_MINUS] = pygame.BLEND_MIN

    blitsn = {}
    blitsn[pygame.K_a] = "BLEND_ADD"
    blitsn[pygame.K_s] = "BLEND_SUB"
    blitsn[pygame.K_m] = "BLEND_MULT"
    blitsn[pygame.K_EQUALS] = "BLEND_MAX"
    blitsn[pygame.K_MINUS] = "BLEND_MIN"

    screen.blit(im1, (0, 0))
    pygame.display.flip()
    clock = pygame.time.Clock()
    print("one pixel is:%s:" % [im1.get_at((0, 0))])

    going = True
    while going:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                going = False
            if event.type == pygame.KEYDOWN:
                usage()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                going = False

            elif event.type == pygame.KEYDOWN and event.key in images.keys():
                img_to_blit = images[event.key]
                iaa = img_to_blit.convert_alpha()

            elif event.type == pygame.KEYDOWN and event.key in blits.keys():
                t1 = time.time()
                # blits is a dict keyed with key -> blit flag.  eg BLEND_ADD.
                im1.blit(img_to_blit, (0, 0), None, blits[event.key])
                t2 = time.time()
                print("one pixel is:%s:" % [im1.get_at((0, 0))])
                print("time to do:%s:" % (t2 - t1))

            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_t]:

                for bkey in blits.keys():
                    t1 = time.time()

                    for x in range(300):
                        im1.blit(img_to_blit, (0, 0), None, blits[bkey])

                    t2 = time.time()

                    # show which key we're doing...
                    onedoing = blitsn[bkey]
                    print("time to do :%s: is :%s:" % (onedoing, t2 - t1))

            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_o]:
                t1 = time.time()
                # blits is a dict keyed with key -> blit flag.  eg BLEND_ADD.
                im1.blit(iaa, (0, 0))
                t2 = time.time()
                print("one pixel is:%s:" % [im1.get_at((0, 0))])
                print("time to do:%s:" % (t2 - t1))

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # this additive blend without clamp two surfaces.
                # im1.set_alpha(127)
                # im1.blit(im1, (0,0))
                # im1.set_alpha(255)
                t1 = time.time()

                im1p = pygame.surfarray.pixels2d(im1)
                im2p = pygame.surfarray.pixels2d(im2)
                im1p += im2p
                del im1p
                del im2p
                t2 = time.time()
                print("one pixel is:%s:" % [im1.get_at((0, 0))])
                print("time to do:%s:" % (t2 - t1))

            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_z]:
                t1 = time.time()
                im1p = pygame.surfarray.pixels3d(im1)
                im2p = pygame.surfarray.pixels3d(im2)
                im1p16 = im1p.astype(numpy.uint16)
                im2p16 = im1p.astype(numpy.uint16)
                im1p16 += im2p16
                im1p16 = numpy.minimum(im1p16, 255)
                pygame.surfarray.blit_array(im1, im1p16)

                del im1p
                del im2p
                t2 = time.time()
                print("one pixel is:%s:" % [im1.get_at((0, 0))])
                print("time to do:%s:" % (t2 - t1))

            elif event.type == pygame.KEYDOWN and event.key in [
                pygame.K_r,
                pygame.K_g,
                pygame.K_b,
            ]:
                # this adds one to each pixel.
                colmap = {}
                colmap[pygame.K_r] = 0x10000
                colmap[pygame.K_g] = 0x00100
                colmap[pygame.K_b] = 0x00001
                im1p = pygame.surfarray.pixels2d(im1)
                im1p += colmap[event.key]
                del im1p
                print("one pixel is:%s:" % [im1.get_at((0, 0))])

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                print("one pixel is:%s:" % [im1.get_at((0, 0))])

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                # this additive blend without clamp two surfaces.

                t1 = time.time()
                im1.set_alpha(127)
                im1.blit(im2, (0, 0))
                im1.set_alpha(255)

                t2 = time.time()
                print("one pixel is:%s:" % [im1.get_at((0, 0))])
                print("time to do:%s:" % (t2 - t1))

        screen.blit(im1, (0, 0))
        pygame.display.flip()

    pygame.quit()


def usage():
    print("press keys 1-5 to change image to blit.")
    print("A - ADD, S- SUB, M- MULT, - MIN, + MAX")
    print("T - timing test for special blend modes.")


if __name__ == "__main__":
    usage()
    main()
