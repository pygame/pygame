#!/usr/bin/env python

'''Pygame core routines

Contains the core routines that are used by the rest of the
pygame modules. Its routines are merged directly into the pygame
namespace. This mainly includes the auto-initialization `init` and
`quit` routines.

There is a small module named `locals` that also gets merged into
this namespace. This contains all the constants needed by pygame.
Object constructors also get placed into this namespace, you can
call functions like `Rect` and `Surface` to create objects of
that type. As a convenience, you can import the members of
pygame.locals directly into your module's namespace with::

    from pygame.locals import *
    
Most of the pygame examples do this if you'd like to take a look.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import atexit
import sys

import SDL

_quitfunctions = []

class error(RuntimeError):
    pass

def init():
    '''Autoinitialize all imported pygame modules.

    Initialize all imported pygame modules. Includes pygame modules
    that are not part of the base modules (like font and image).

    It does not raise exceptions, but instead silently counts which
    modules have failed to init. The return argument contains a count
    of the number of modules initialized, and the number of modules
    that failed to initialize.

    You can always initialize the modules you want by hand. The
    modules that need it have an `init` and `quit` routine built in,
    which you can call directly. They also have a `get_init` routine
    which you can use to doublecheck the initialization. Note that
    the manual `init` routines will raise an exception on error. Be
    aware that most platforms require the display module to be
    initialized before others. This `init` will handle that for you,
    but if you initialize by hand, be aware of this constraint.

    As with the manual `init` routines. It is safe to call this
    `init` as often as you like. 

    :rtype: int, int
    :return: (count_passed, count_failed)
    '''
    success = 0
    fail = 0

    SDL.SDL_Init(SDL.SDL_INIT_EVENTTHREAD | SDL.SDL_INIT_TIMER)
    for mod in sys.modules.values():
        if hasattr(mod, '__PYGAMEinit__') and callable(mod.__PYGAMEinit__):
            try:
                mod.__PYGAMEinit__()
                success += 1
            except:
                fail += 1
    return success, fail

def register_quit(func):
    '''Routine to call when pygame quits.

    The given callback routine will be called when pygame is
    quitting. Quit callbacks are served on a 'last in, first out'
    basis.
    '''
    _quitfunctions.append(func)

def _video_autoquit():
    if SDL.SDL_WasInit(SDL.SDL_INIT_VIDEO):
        SDL.SDL_QuitSubSystem(SDL.SDL_INIT_VIDEO)

def _video_autoinit():
    if not SDL.SDL_WasInit(SDL.SDL_INIT_VIDEO):
        SDL.SDL_InitSubSystem(SDL.SDL_INIT_VIDEO)
        SDL.SDL_EnableUNICODE(1)

def _atexit_quit():
    while _quitfunctions:
        func = _quitfunctions.pop()
        func()
    _video_autoquit()
    SDL.SDL_Quit()

def get_sdl_version():
    '''Get the version of the linked SDL runtime.

    :rtype: int, int, int
    :return: major, minor, patch
    '''
    v = SDL.SDL_Linked_Version()
    return v.major, v.minor, v.patch

def quit():
    '''Uninitialize all pygame modules.

    Uninitialize all pygame modules that have been initialized. Even
    if you initialized the module by hand, this `quit` will
    uninitialize it for you.

    All the pygame modules are uninitialized automatically when your
    program exits, so you will usually not need this routine. If you
    program plans to keep running after it is done with pygame, then
    would be a good time to make this call.
    '''
    _atexit_quit()

def get_error():
    '''Get current error message.

    SDL maintains an internal current error message. This message is
    usually given to you when an SDL related exception occurs, but
    sometimes you may want to call this directly yourself.

    :rtype: str
    '''
    return SDL.SDL_GetError()

def _rgba_from_obj(obj):
    if not type(obj) in (tuple, list):
        return None

    if len(obj) == 1:
        return _rgba_from_obj(obj[0])
    elif len(obj) == 3:
        return (obj[0], obj[1], obj[2], 255)
    elif len(obj) == 4:
        return obj
    else:
        return None

atexit.register(_atexit_quit)

