#!/usr/bin/env python

'''Test program to check the resolution of the SDL timer on the current
platform.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *

DEFAULT_RESOLUTION = 1

ticks = 0

def ticktock(interval):
    global ticks
    ticks += 1
    return interval

def callback(interval, param):
    print 'Timer %d : param = %d' % (interval, param)
    return interval

if __name__ == '__main__':
    SDL_Init(SDL_INIT_TIMER)

    desired = 0
    if len(sys.argv) > 1:
        desired = int(sys.argv[1])
    if desired == 0:
        desired = DEFAULT_RESOLUTION

    SDL_SetTimer(desired, ticktock)

    print 'Waiting 10 seconds'
    SDL_Delay(10*1000)

    SDL_SetTimer(0, None)
    if ticks:
        print 'Timer resolution: desired = %d ms, actual = %f ms' % \
            (desired, 10*1000.0/ticks)

    print 'Testing multiple timers...'
    t1 = SDL_AddTimer(100, callback, 1)
    t2 = SDL_AddTimer(50, callback, 2)
    t3 = SDL_AddTimer(233, callback, 3)

    print 'Waiting 10 seconds'
    SDL_Delay(10*1000)
    print 'Removing timer 1 and waiting 5 more seconds'
    SDL_RemoveTimer(t1)
    SDL_Delay(5*1000)

    SDL_RemoveTimer(t2)
    SDL_RemoveTimer(t3)

    SDL_Quit()
