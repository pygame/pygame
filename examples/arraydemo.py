#!/usr/bin/env python

import os

import pygame
from pygame import surfarray
from pygame.locals import *

main_dir = os.path.split(os.path.abspath(__file__))[0]

def surfdemo_show(array_img, name):
    "displays a surface, waits for user to continue"
    screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
    surfarray.blit_array(screen, array_img)
    pygame.display.flip()
    pygame.display.set_caption(name)
    while 1:
        e = pygame.event.wait()
        if e.type == MOUSEBUTTONDOWN: break
        elif e.type == KEYDOWN and e.key == K_s:
            #pygame.image.save(screen, name+'.bmp')
            #s = pygame.Surface(screen.get_size(), 0, 32)
            #s = s.convert_alpha()
            #s.fill((0,0,0,255))
            #s.blit(screen, (0,0))
            #s.fill((222,0,0,50), (0,0,40,40))
            #pygame.image.save_extended(s, name+'.png')
            #pygame.image.save(s, name+'.png')
            #pygame.image.save(screen, name+'_screen.png')
            #pygame.image.save(s, name+'.tga')
            pygame.image.save(screen, name+'.png')
        elif e.type == QUIT:
            raise SystemExit()

def main(arraytype=None):
    """show various surfarray effects

    If arraytype is provided then use that array package. Valid
    values are 'numeric' or 'numpy'. Otherwise default to NumPy,
    or fall back on Numeric if NumPy is not installed.

    """
    if arraytype not in ('numpy', None):
        raise ValueError('Array type not supported: %r' % arraytype)

    import numpy as N
    from numpy import int32, uint8, uint

    pygame.init()
    print ('Using %s' % surfarray.get_arraytype().capitalize())
    print ('Press the mouse button to advance image.')
    print ('Press the "s" key to save the current image.')

    #allblack
    allblack = N.zeros((128, 128), int32)
    surfdemo_show(allblack, 'allblack')


    #striped
    #the element type is required for N.zeros in  NumPy else
    #an array of float is returned.
    striped = N.zeros((128, 128, 3), int32)
    striped[:] = (255, 0, 0)
    striped[:,::3] = (0, 255, 255)
    surfdemo_show(striped, 'striped')


    #rgbarray
    imagename = os.path.join(main_dir, 'data', 'arraydemo.bmp')
    imgsurface = pygame.image.load(imagename)
    rgbarray = surfarray.array3d(imgsurface)
    surfdemo_show(rgbarray, 'rgbarray')


    #flipped
    flipped = rgbarray[:,::-1]
    surfdemo_show(flipped, 'flipped')


    #scaledown
    scaledown = rgbarray[::2,::2]
    surfdemo_show(scaledown, 'scaledown')


    #scaleup
    #the element type is required for N.zeros in NumPy else
    #an #array of floats is returned.
    shape = rgbarray.shape
    scaleup = N.zeros((shape[0]*2, shape[1]*2, shape[2]), int32)
    scaleup[::2,::2,:] = rgbarray
    scaleup[1::2,::2,:] = rgbarray
    scaleup[:,1::2] = scaleup[:,::2]
    surfdemo_show(scaleup, 'scaleup')


    #redimg
    redimg = N.array(rgbarray)
    redimg[:,:,1:] = 0
    surfdemo_show(redimg, 'redimg')


    #soften
    #having factor as an array forces integer upgrade during multiplication
    #of rgbarray, even for numpy.
    factor = N.array((8,), int32)
    soften = N.array(rgbarray, int32)
    soften[1:,:]  += rgbarray[:-1,:] * factor
    soften[:-1,:] += rgbarray[1:,:] * factor
    soften[:,1:]  += rgbarray[:,:-1] * factor
    soften[:,:-1] += rgbarray[:,1:] * factor
    soften //= 33
    surfdemo_show(soften, 'soften')


    #crossfade (50%)
    src = N.array(rgbarray)
    dest = N.zeros(rgbarray.shape)     # dest is float64 by default.
    dest[:] = 20, 50, 100
    diff = (dest - src) * 0.50
    xfade = src + diff.astype(uint)
    surfdemo_show(xfade, 'xfade')


    #alldone
    pygame.quit()

if __name__ == '__main__':
    main()



