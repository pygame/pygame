#!/usr/bin/env python

'''Mouse event handling.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll
import SDL.video

class SDL_Cursor(Structure):
    _fields_ = [('area', SDL.video.SDL_Rect),
                ('hot_x', c_short),
                ('hot_y', c_short),
                ('data', POINTER(c_ubyte)),
                ('mask', POINTER(c_ubyte)),
                ('save', POINTER(c_ubyte) * 2),
                ('wm_cursor', c_void_p)]

_SDL_GetMouseState = SDL.dll.private_function('SDL_GetMouseState',
    arg_types=[POINTER(c_int), POINTER(c_int)],
    return_type=c_ubyte)

def SDL_GetMouseState():
    '''Retrieve the current state of the mouse.

    Example::

        state, x, y = SDL_GetMouseState()
        if state & SDL_BUTTON_LMASK:
            print 'Left button pressed.'

    :rtype: (int, int, int)
    :return: (state, x, y), where
        * state is a button bitmask, which can be tested using `SDL_BUTTON`
        * x and y are the current mouse cursor position.

    '''
    x, y = c_int(), c_int()
    state = _SDL_GetMouseState(byref(x), byref(y))
    return state, x.value, y.value

_SDL_GetRelativeMouseState = \
    SDL.dll.private_function('SDL_GetRelativeMouseState',
    arg_types=[POINTER(c_int), POINTER(c_int)],
    return_type=c_ubyte)

def SDL_GetRelativeMouseState():
    '''Retrieve the current state of the mouse.

    :rtype: (int, int, int)
    :return: (state, dx, dy), 
        where state is a button bitmask, which can be tested using
        `SDL_BUTTON`; dx and dy are the mouse deltas since the last call to
        `SDL_GetRelativeMouseState`

    '''
    dx, dy = c_int(), c_int()
    state = _SDL_GetRelativeMouseState(byref(dx), byref(dy))
    return state, dx.value, dy.value

SDL_WarpMouse = SDL.dll.function('SDL_WarpMouse',
    '''Set the position of the mouse cursor.

    Generates a mouse motion event.

    :Parameters:
     - `x`: int
     - `y`: int

    ''',
    args=['x', 'y'],
    arg_types=[c_ushort, c_ushort],
    return_type=None)

# TODO: SDL_CreateCursor

SDL_SetCursor = SDL.dll.function('SDL_SetCursor',
    '''Set the currently active cursor to the specified one.

    If the cursor is currently visible, the change will be immediately
    represented on the display.

    :Parameters:
     - `cursor`: `SDL_Cursor`

    ''',
    args=['cursor'],
    arg_types=[POINTER(SDL_Cursor)],
    return_type=None)

SDL_GetCursor = SDL.dll.function('SDL_GetCursor',
    '''Return the currently active cursor.

    :rtype: `SDL_Cursor`
    ''',
    args=[],
    arg_types=[],
    return_type=POINTER(SDL_Cursor),
    dereference_return=True)

SDL_FreeCursor = SDL.dll.function('SDL_FreeCursor',
    '''Deallocate a cursor created with `SDL_CreateCursor`

    :Parameters:
     - `cursor`: `SDL_Cursor`
    ''',
    args=['cursor'],
    arg_types=[POINTER(SDL_Cursor)],
    return_type=None)

SDL_ShowCursor = SDL.dll.function('SDL_ShowCursor',
    '''Toggle whether or not the curosr is shown on the screen.

    The cursor starts off displayed, but can be turned off.

    :Parameters:
        `toggle` : int
            if 1, shows the cursor; if 0, hides the cursor, if -1, returns
            the current state of the cursor.

    :rtype: int
    :return: 1 if the cursor was being displayed before the call, otherwise
        0.
    ''',
    args=['toggle'],
    arg_types=[c_int],
    return_type=c_int)

def SDL_BUTTON(X):
    '''Used to create a mask for a mouse button.
    '''
    return 1 << (X-1)

SDL_BUTTON_LMASK    = SDL_BUTTON(SDL.constants.SDL_BUTTON_LEFT)
SDL_BUTTON_MMASK    = SDL_BUTTON(SDL.constants.SDL_BUTTON_MIDDLE)
SDL_BUTTON_RMASK    = SDL_BUTTON(SDL.constants.SDL_BUTTON_RIGHT)
