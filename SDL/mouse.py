#!/usr/bin/env python

'''Mouse event handling.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.array
import SDL.dll
import SDL.video

class SDL_Cursor(Structure):
    '''Cursor structure.

    :Ivariables:
        `area` : `SDL_Rect`
            Area of the mouse cursor
        `hot_x` : int
            X coordinate of the tip of the cursor
        `hot_y` : int
            Y coordinate of the tip of the cursor
    
    '''
    _fields_ = [('area', SDL.video.SDL_Rect),
                ('hot_x', c_short),
                ('hot_y', c_short),
                ('_data', POINTER(c_ubyte)),
                ('_mask', POINTER(c_ubyte)),
                ('save', POINTER(c_ubyte) * 2),
                ('wm_cursor', c_void_p)]

    def __getattr__(self, name):
        w, h = self.area.w, self.area.h
        if name == 'data':
            return SDL.array.SDL_array(self._data, w * h / 8, c_ubyte)
        elif name == 'mask':
            return SDL.array.SDL_array(self._mask, w * h / 8, c_ubyte)
        raise AttributeError, name

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

_SDL_CreateCursor = SDL.dll.private_function('SDL_CreateCursor',
    arg_types=[POINTER(c_ubyte), POINTER(c_ubyte), c_int, c_int, c_int, c_int],
    return_type=POINTER(SDL_Cursor),
    dereference_return=True,
    require_return=True)

def SDL_CreateCursor(data, mask, w, h, hot_x, hot_y):
    '''Create a cursor using the specified data and mask.

    The cursor width must be a multiple of 8 bits.  Mask and cursor
    data may be given as either SDL_array byte buffers or as a sequence
    of bytes; in either case the data is in MSB order.

    The cursor is created in black and white according to the following:

    ==== ==== =========================================
    data mask resulting pixel on screen
    ==== ==== =========================================
    0    1    White
    1    1    Black
    0    0    Transparent
    1    0    Inverted color if possible, black if not.
    ==== ==== =========================================

    Cursors created with this function must be freed with `SDL_FreeCursor`.

    :Parameters:
     - `data`: `SDL_array`
     - `mask`: `SDL_array`
     - `w`: int
     - `h`: int
     - `hot_x`: int
     - `hot_y`: int

    :rtype: `SDL_Cursor`
    '''
    dataref, data = SDL.array.to_ctypes(data, len(data), c_ubyte)
    maskref, mask = SDL.array.to_ctypes(mask, len(mask), c_ubyte)
    return _SDL_CreateCursor(data, mask, w, h, hot_x, hot_y)

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
