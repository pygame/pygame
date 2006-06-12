#!/usr/bin/env python

'''Error detection and error handling functions.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll

class SDL_Exception(Exception):
    '''Exception raised for all SDL errors.  
    
    The message is as returned by `SDL_GetError`.
    '''
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class SDL_NotImplementedError(NotImplementedError):
    '''Exception raised when the available SDL library predates the
    requested function.'''
    pass

SDL_SetError = SDL.dll.function('SDL_SetError',
    '''Set the static error string.

    :Parameters:
        `fmt`
            format string; subsequent integer and string arguments are
            interpreted as in printf().
    ''',
    args=['fmt'],
    arg_types=[c_char_p],
    return_type=None)

SDL_GetError = SDL.dll.function('SDL_GetError',
    '''Return the last error string set.

    :rtype: string
    ''',
    args=[],
    arg_types=[],
    return_type=c_char_p)

SDL_ClearError = SDL.dll.function('SDL_ClearError',
    '''Clear any error string set.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

# SDL_Error not implemented (marked private in SDL_error.h)
