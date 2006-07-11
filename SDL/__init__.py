#!/usr/bin/env python

'''Main module for importing SDL-ctypes.

This module defines the intialization and cleanup functions:
    - `SDL_Init`
    - `SDL_InitSubSystem`
    - `SDL_WasInit`
    - `SDL_QuitSubSystem`
    - `SDL_Quit`

It also imports all functions, constants and classes from the package
submodules.  Typical usage, then, is to just import this module directly
into your namespace::

    from SDL import *

This gives you access to all SDL names exactly as they appear in the C
header files.

:group Non-core modules: ttf, mixer, image, sound
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import ctypes
import sys

import SDL.dll
from SDL.active import *
from SDL.audio import *
from SDL.cdrom import *
from SDL.constants import *
from SDL.endian import *
from SDL.error import *
from SDL.events import *
from SDL.joystick import *
from SDL.keyboard import *
from SDL.mouse import *
from SDL.quit import *
from SDL.rwops import *
from SDL.timer import *
from SDL.version import *
from SDL.video import *

# SDL.h

_SDL_Init = SDL.dll.private_function('SDL_Init',
    arg_types=[ctypes.c_uint],
    return_type=ctypes.c_int)

def SDL_Init(flags):
    '''Initialise the SDL library.

    This function loads the SDL dynamically linked library and initializes 
    the subsystems specified by `flags` (and those satisfying dependencies)
    Unless the `SDL_INIT_NOPARACHUTE` flag is set, it will install cleanup
    signal handlers for some commonly ignored fatal signals (like SIGSEGV).

    The following flags are recognised:

        - `SDL_INIT_TIMER`
        - `SDL_INIT_AUDIO`
        - `SDL_INIT_VIDEO`
        - `SDL_INIT_CDROM`
        - `SDL_INIT_JOYSTICK`
        - `SDL_INIT_NOPARACHUTE`
        - `SDL_INIT_EVENTTHREAD`
        - `SDL_INIT_EVERYTHING`

    :Parameters:
     - `flags`: int

    :rtype: int
    :return: undocumented (FIXME)
    :see: `SDL_Quit`
    '''
    if sys.platform == 'darwin' and flags & SDL_INIT_VIDEO:
        import SDL.darwin
        SDL.darwin.init()
    return _SDL_Init(flags)

_SDL_InitSubSystem = SDL.dll.private_function('SDL_InitSubSystem',
    arg_types=[ctypes.c_uint],
    return_type=ctypes.c_int)

def SDL_InitSubSystem(flags):
    '''Initialize specific SDL subsystems.

    :Parameters:
     - `flags`: int

    :rtype: int
    :return: undocumented (FIXME)
    :see: `SDL_Init`, `SDL_QuitSubSystem`
    '''
    if sys.platform == 'darwin' and flags & SDL_INIT_VIDEO:
        import SDL.darwin
        SDL.darwin.init()
    return _SDL_InitSubSystem(flags)

SDL_QuitSubSystem = SDL.dll.function('SDL_QuitSubSystem',
    '''Clean up specific SDL subsystems.

    :Parameters:
     - `flags`: int

    :see: `SDL_InitSubSystem`
    ''',
    args=['flags'],
    arg_types=[ctypes.c_uint],
    return_type=None)

SDL_WasInit = SDL.dll.function('SDL_WasInit',
    '''Return a mask of the specified subsystems which have been
    initialized.

    If `flags` is 0, return a mask of all initialized subsystems.

    :Parameters:
     - `flags`: int

    :rtype: int
    :return: undocumented (FIXME)
    :see: `SDL_Init`
    ''',
    args=['flags'],
    arg_types=[ctypes.c_uint],
    return_type=ctypes.c_int)

SDL_Quit = SDL.dll.function('SDL_Quit',
    '''Clean up all initialized subsystems.

    You should call this function upon all exit conditions.
    ''',
    args=[],
    arg_types=[],
    return_type=None)
