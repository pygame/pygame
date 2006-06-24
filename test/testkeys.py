#!/usr/bin/env python

'''Print out all the keysyms we have, just to verify them
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from SDL import *

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    for key in range(SDLK_FIRST, SDLK_LAST):
        print 'Key #%d, "%s"' % (key, SDL_GetKeyName(key))
    SDL_Quit()
