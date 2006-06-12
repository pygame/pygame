#!/usr/bin/env python

'''SDL application focus event handling.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll

SDL_GetAppState = SDL.dll.function('SDL_GetAppState',
    '''Return the current state of the application.

    The return value is a bitwise combination of `SDL_APPMOUSEFOCUS`,
    `SDL_APPINPUTFOCUS`, and `SDL_APPACTIVE`.  The meanings are as follows:
    
        `SDL_APPMOUSEFOCUS`
            The application has mouse coverage.
        `SDL_APPINPUTFOCUS`
            The application has input focus.
        `SDL_APPACTIVATE` 
            The user is able to see your application, otherwise it has been
            iconified or disabled
    
    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)
