#!/usr/bin/env python
""" pygame.examples.arraydemo

Welcome to the arraydemo!

Use the numpy array package to manipulate pixels.

This demo will show you a few things:

* scale up, scale down, flip,
* cross fade
* soften
* put stripes on it!

"""


import os

import pygame as pg
from pygame import surfarray

main_dir = os.path.split(os.path.abspath(__file__))[0]


def surfdemo_show(array_img, name):
    "displays a surface, waits for user to continue"
    screen = pg.display.set_mode(array_img.shape[:2], 0, 32)
    surfarray.blit_array(screen, array_img)
    pg.display.flip()
    pg.display.set_caption(name)
    while 1:
        e = pg.event.wait()
        if e.type == pg.MOUSEBUTTONDOWN:
            break
        elif e.type == pg.KEYDOWN and e.key == pg.K_s:
            # pg.image.save(screen, name+'.bmp')
            # s = pg.Surface(screen.get_size(), 0, 32)
            # s = s.convert_alpha()
            # s.fill((0,0,0,255))
            # s.blit(screen, (0,0))
            # s.fill((222,0,0,50), (0,0,40,40))
            # pg.image.save_extended(s, name+'.png')
            # pg.image.save(s, name+'.png')
            # pg.image.save(screen, name+'_screen.png')
            # pg.image.save(s, name+'.tga')
            pg.image.save(screen, name + ".png")
        elif e.type == pg.QUIT:
            pg.quit()
            raise SystemExit()


def main():
    """show various surfarray effects"""
    import numpy as N
    from numpy import int32, uint8, uint

    pg.init()
    print("Using %s" % surfarray.get_arraytype().capitalize())
    print("Press the mouse button to advance image.")
    print('Press the "s" key to save the current image.')

    # allblack
    allblack = N.zeros((128, 128), int32)
    surfdemo_show(allblack, "allblack")

    # striped
    # the element type is required for N.zeros in numpy else
    # an array of float is returned.
    striped = N.zeros((128, 128, 3), int32)
    striped[:] = (255, 0, 0)
    striped[:, ::3] = (0, 255, 255)
    surfdemo_show(striped, "striped")

    # rgbarray
    imagename = os.path.join(main_dir, "data", "arraydemo.bmp")
    imgsurface = pg.image.load(imagename)
    rgbarray = surfarray.array3d(imgsurface)
    surfdemo_show(rgbarray, "rgbarray")

    # flipped
    flipped = rgbarray[:, ::-1]
    surfdemo_show(flipped, "flipped")

    # scaledown
    scaledown = rgbarray[::2, ::2]
    surfdemo_show(scaledown, "scaledown")

    # scaleup
    # the element type is required for N.zeros in numpy else
    # an #array of floats is returned.
    shape = rgbarray.shape
    scaleup = N.zeros((shape[0] * 2, shape[1] * 2, shape[2]), int32)
    scaleup[::2, ::2, :] = rgbarray
    scaleup[1::2, ::2, :] = rgbarray
    scaleup[:, 1::2] = scaleup[:, ::2]
    surfdemo_show(scaleup, "scaleup")

    # redimg
    redimg = N.array(rgbarray)
    redimg[:, :, 1:] = 0
    surfdemo_show(redimg, "redimg")

    # soften
    # having factor as an array forces integer upgrade during multiplication
    # of rgbarray, even for numpy.
    factor = N.array((8,), int32)
    soften = N.array(rgbarray, int32)
    soften[1:, :] += rgbarray[:-1, :] * factor
    soften[:-1, :] += rgbarray[1:, :] * factor
    soften[:, 1:] += rgbarray[:, :-1] * factor
    soften[:, :-1] += rgbarray[:, 1:] * factor
    soften //= 33
    surfdemo_show(soften, "soften")

    # crossfade (50%)
    src = N.array(rgbarray)
    dest = N.zeros(rgbarray.shape)  # dest is float64 by default.
    dest[:] = 20, 50, 100
    diff = (dest - src) * 0.50
    xfade = src + diff.astype(uint)
    surfdemo_show(xfade, "xfade")

    # alldone
    pg.quit()


if __name__ == "__main__":
    main()
