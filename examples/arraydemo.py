#!/usr/bin/env python

import sys
import os
try:
    import pygame
    from pygame import surfarray
    from pygame.locals import *
except ImportError:
    raise ImportError('Error Importing Pygame/surfarray')

main_dir = os.path.split(os.path.abspath(__file__))[0]

def main(arraytype=None):
    """show various surfarray effects

    If arraytype is provided then use that array package. Valid
    values are 'numeric' or 'numpy'. Otherwise default to NumPy,
    or fall back on Numeric if NumPy is not installed.

    """

    if arraytype is None:
        if 'numpy' in surfarray.get_arraytypes():
            surfarray.use_arraytype('numpy')
        elif 'numeric' in surfarray.get_arraytype():
            surfarray.use_arraytype('numeric')
        else:
            raise ImportError('No array package is installed')
    else:
        surfarray.use_arraytype(arraytype)

    if surfarray.get_arraytype() == 'numpy':
        import numpy as N
        from numpy import int32, uint8, uint
    else:
        import Numeric as N
        from Numeric import Int32 as int32, UInt8 as uint8, UInt as uint

    pygame.init()
    print ('Using %s' % surfarray.get_arraytype().capitalize())
    print ('Press the mouse button to advance image.')
    print ('Press the "s" key to save the current image.')

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


    #imgarray
    imagename = os.path.join(main_dir, 'data', 'arraydemo.bmp')
    imgsurface = pygame.image.load(imagename)
    imgarray = surfarray.array2d(imgsurface)
    surfdemo_show(imgarray, 'imgarray')


    #flipped
    flipped = imgarray[:,::-1]
    surfdemo_show(flipped, 'flipped')


    #scaledown
    scaledown = imgarray[::2,::2]
    surfdemo_show(scaledown, 'scaledown')


    #scaleup
    #the element type is required for N.zeros in NumPy else
    #an #array of floats is returned.
    size = N.array(imgarray.shape)*2
    scaleup = N.zeros(size, int32)
    scaleup[::2,::2] = imgarray
    scaleup[1::2,::2] = imgarray
    scaleup[:,1::2] = scaleup[:,::2]
    surfdemo_show(scaleup, 'scaleup')


    #redimg
    rgbarray = surfarray.array3d(imgsurface)
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
    dest = N.zeros(rgbarray.shape)
    dest[:] = 20, 50, 100
    diff = (dest - src) * 0.50
    xfade = src + diff.astype(uint)
    surfdemo_show(xfade, 'xfade')




    #alldone
    pygame.quit()

def usage():
    print ("Usage: command line option [--numpy|--numeric]")
    print ("  The default is to use NumPy if installed,")
    print ("  otherwise Numeric")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if '--numpy' in sys.argv:
            main('numpy')
        elif '--numeric' in sys.argv:
            main('numeric')
        else:
            usage()
    elif len(sys.argv) == 1:
        main()
    else:
        usage()

        

