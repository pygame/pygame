#!/usr/bin/env python

'''Keyboard event handling.

You should call `SDL_Init` with `SDL_INIT_VIDEO` before using these functions.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll

class SDL_keysym(Structure):
    '''Keysym structure

        * The scancode is hardware dependent, and should not be used by
          general applications.  If no hardware scancode is available, it will
          be 0
        * The `unicode` translated character is only available when character
          translation is enabled by `SDL_EnableUNICODE`.  If non-empty, this is
          unicode string of unit length.

    :Ivariables:
        `scancode` : int
            Hardware specific scancode
        `sym` : int
            SDL virtual keysym (SDLK_*)
        `mod` : int
            Bitwise OR of current key modifiers
        `unicode` : string
            Unicode character represented by keypress, or the empty string
            if translation is not possible.

    '''
    _fields_ = [('scancode', c_ubyte),
                ('sym', c_int),
                ('mod', c_int),
                ('_unicode', c_ushort)]

    def __getattr__(self, name):
        if name == 'unicode':
            return unichr(self._unicode)
        raise AttributeError

SDL_EnableUNICODE = SDL.dll.function('SDL_EnableUNICODE',
    '''Enable or disable Unicode translation of keyboard input.

    This translation has some overhead, so translation defaults off.

    :Parameters:
        `enable` : int
            * if 1, translation is enabled
            * if 0, translation is disabled.  
            * if -1, the translation is not changed.

    :rtype: int
    :return: the previous state of keyboard translation
    ''',
    args=['enable'],
    arg_types=[c_int],
    return_type=c_int)

SDL_EnableKeyRepeat = SDL.dll.function('SDL_EnableKeyRepeat',
    '''Enable keyboard repeat.

    Keyboard repeat defaults to off.

    :Parameters:
        `delay` : int
            the initial delay in milliseconds between the time when a key is
            pressed, and keyboard repeat begins.  If 0, keyboard repeat is
            disabled.
        `interval` : int
            the time in milliseconds between keyboard repeat events.

    :rtype: int
    :return: undocumented (FIXME)
    ''',
    args=['delay', 'interval'],
    arg_types=[c_int, c_int],
    return_type=c_int)

_SDL_GetKeyRepeat = SDL.dll.private_function('SDL_GetKeyRepeat',
    arg_types=[POINTER(c_int), POINTER(c_int)],
    return_type=None,
    since=(1,2,10))

def SDL_GetKeyRepeat():
    '''Get the keyboard repeat parameters.

    :see: `SDL_EnableKeyRepeat`
    :rtype: (int, int)
    :return: tuple (delay, interval), as defined in `SDL_EnableKeyRepeat`
    :since: 1.2.10
    '''
    delay, interval = c_int(), c_int()
    _SDL_GetKeyRepeat(byref(delay), byref(interval))
    return delay.value, interval.value

_SDL_GetKeyState = SDL.dll.private_function('SDL_GetKeyState',
    arg_types=[POINTER(c_int)],
    return_type=POINTER(c_ubyte))

def SDL_GetKeyState():
    '''Get a snapshot of the current state of the keyboard.

    Example::
        
        keystate = SDL_GetKeyState()
        if keystate[SDLK_RETURN]:
            print '<RETURN> is pressed'
    
    :rtype: list
    :return: a list of integers indexed by the SDLK_* symbols
    '''
    numkeys = c_int()
    keystate = _SDL_GetKeyState(byref(numkeys))
    keystate_ar = cast(keystate, POINTER(c_ubyte * numkeys.value)).contents
    return list(keystate_ar)

SDL_GetModState = SDL.dll.function('SDL_GetModState',
    '''Get the current key modifier state.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

SDL_SetModState = SDL.dll.function('SDL_SetModState',
    '''Set the current key modifier state.

    This does not change the keyboard state, only the key modifier flags.

    :Parameters:
     - `modstate`: int
    
    ''',
    args=['modstate'],
    arg_types=[c_int],
    return_type=None)

_SDL_GetKeyName = SDL.dll.private_function('SDL_GetKeyName',
    arg_types=[c_int],
    return_type=c_char_p)

def SDL_GetKeyName(key):
    '''Get the name of an SDL virtual keysym.

    :Parameters:
     - `key`: int

    :rtype: string
    '''
    return _SDL_GetKeyName(key)

