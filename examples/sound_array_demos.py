#!/usr/bin/env python
"""
Creates an echo effect an any Sound object.

Uses sndarray and Numeric to create offset faded copies of the
original sound. Currently it just uses hardcoded values for the
number of echos and the delay. Easy for you to recreate as 
needed.

version 2. changes:
- Should work with different sample rates now.
- put into a function.

"""

__author__ = "Pete 'ShredWheat' Shinners, Rene Dudfield"
__copyright__ = "Copyright (C) 2004 Pete Shinners, Copyright (C) 2005 Rene Dudfield"
__license__ = "Public Domain"
__version__ = "2.0"


import os.path
import pygame.mixer, pygame.time, pygame.sndarray, pygame
mixer = pygame.mixer
sndarray = pygame.sndarray
import time
from math import sin
from Numeric import *

#mixer.init(44100, -16, 0)
mixer.init()
#mixer.init(11025, -16, 0)
#mixer.init(11025)






def make_echo(sound, samples_per_second,  mydebug = True):
    """ returns a sound which is echoed of the last one.
    """

    echo_length = 3.5

    a1 = sndarray.array(sound)
    if mydebug:
        print 'SHAPE1:', a1.shape

    length = a1.shape[0]

    #myarr = zeros(length+12000)
    myarr = zeros(a1.shape)

    if len(a1.shape) > 1:
        mult = a1.shape[1]
        size = (a1.shape[0] + int(echo_length * a1.shape[0]), a1.shape[1])
        #size = (a1.shape[0] + int(a1.shape[0] + (echo_length * 3000)), a1.shape[1])
    else:
        mult = 1
        size = (a1.shape[0] + int(echo_length * a1.shape[0]),)
        #size = (a1.shape[0] + int(a1.shape[0] + (echo_length * 3000)),)

    if mydebug:
        print int(echo_length * a1.shape[0])
    myarr = zeros(size)



    if mydebug:
        print "size", size
        print myarr.shape
    myarr[:length] = a1
    #print myarr[3000:length+3000]
    #print a1 >> 1
    #print "a1.shape", a1.shape
    #c = myarr[3000:length+(3000*mult)]
    #print "c.shape", c.shape

    incr = int(samples_per_second / echo_length)
    gap = length


    myarr[incr:gap+incr] += a1>>1
    myarr[incr*2:gap+(incr*2)] += a1>>2
    myarr[incr*3:gap+(incr*3)] += a1>>3
    myarr[incr*4:gap+(incr*4)] += a1>>4

    if mydebug:
        print 'SHAPE2:', myarr.shape


    sound2 = sndarray.make_sound(myarr.astype(Int16))

    return sound2



if __name__ == "__main__":


    print "mixer.get_init", mixer.get_init()
    inited = mixer.get_init()

    samples_per_second = pygame.mixer.get_init()[0]





    sound = mixer.Sound('data/car_door.wav')
    t1 = time.time()
    sound2 = make_echo(sound, samples_per_second)
    print "time to make echo", time.time() - t1


    print "original sound"
    sound.play()
    while mixer.get_busy():
        pygame.time.wait(200)

    print "echoed sound"
    sound2.play()
    while mixer.get_busy():
        pygame.time.wait(200)


    sound = mixer.Sound('data/secosmic_lo.wav')

    t1 = time.time()
    sound3 = make_echo(sound, samples_per_second)
    print "time to make echo", time.time() - t1

    print "original sound"
    sound.play()
    while mixer.get_busy():
        pygame.time.wait(200)


    print "echoed sound"
    sound3.play()
    while mixer.get_busy():
        pygame.time.wait(200)


