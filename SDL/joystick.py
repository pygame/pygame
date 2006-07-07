#!/usr/bin/env python

'''Joystick event handling.

TODO: This module is completely untested.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.constants
import SDL.dll

class _SDL_Joystick(Structure):
    _fields_ = [('_dummy', c_void_p)]

SDL_Joystick_p = POINTER(_SDL_Joystick)

SDL_NumJoysticks = SDL.dll.function('SDL_NumJoysticks',
    '''Count the number of joysticks attached to the system.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

SDL_JoystickName = SDL.dll.function('SDL_JoystickName',
    '''Get the implementation dependent name of a joystick.

    This can be called before any joysticks are opened.  If no name can be
    found, this function returns None.

    :Parameters:
     - `device_index`: int

    :rtype: str
    ''',
    args=['device_index'],
    arg_types=[c_int],
    return_type=c_char_p)

SDL_JoystickOpen = SDL.dll.function('SDL_JoystickOpen',
    '''Open a joystick for use.
    
    The index passed as an argument refers to the N'th joystick on the
    system.  This index is the value which will identify this joystick in
    future joystick events.

    This function returns an opaque joystick identifier.

    :Parameters:
     - `device_index`: int

    :rtype: `SDL_Joystick_p`
    ''',
    args=['device_index'],
    arg_types=[c_int],
    return_type=SDL_Joystick_p,
    require_return=True)

SDL_JoystickOpened = SDL.dll.function('SDL_JoystickOpened',
    '''Determine if a joystick has been opened.

    :Parameters:
     - `device_index`: int

    :rtype: `int`
    :return: 1 if the joystick has been opened, or 0 if it has not.
    ''',
    args=['device_index'],
    arg_types=[c_int],
    return_type=c_int)

SDL_JoystickIndex = SDL.dll.function('SDL_JoystickIndex',
    '''Get the device index of an opened joystick.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    :rtype: int
    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=c_int)

SDL_JoystickNumAxes = SDL.dll.function('SDL_JoystickNumAxes',
    '''Get the number of general axis controls on a joystick.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    :rtype: int
    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=c_int)

SDL_JoystickNumBalls = SDL.dll.function('SDL_JoystickNumBalls',
    '''Get the number of trackballs on a joystick.

    Joystick trackballs have only relative motion events associated with
    them and their state cannot be polled.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    :rtype: int
    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=c_int)

SDL_JoystickNumHats = SDL.dll.function('SDL_JoystickNumHats',
    '''Get the number of POV hats on a joystick.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    :rtype: int
    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=c_int)

SDL_JoystickNumButtons = SDL.dll.function('SDL_JoystickNumButtons',
    '''Get the number of buttons on a joystick.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    :rtype: int
    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=c_int)

SDL_JoystickUpdate = SDL.dll.function('SDL_JoystickUpdate',
    '''Update the current state of the open joysticks.

    This is called automatically by the event loop if any joystick events
    are enabled.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

SDL_JoystickEventState = SDL.dll.function('SDL_JoystickEventState',
    '''Enable/disable joystick event polling.

    If joystick events are disabled, you must call `SDL_JoystickUpdate`
    yourself and check the state of the joystick when you want joystick
    information.

    :Parameters:
        `state` : int
            one of SDL_QUERY, SDL_ENABLE or SDL_IGNORE.

    :rtype: int
    :return: undocumented
    ''',
    args=['state'],
    arg_types=[c_int],
    return_type=c_int)


SDL_JoystickGetAxis = SDL.dll.function('SDL_JoystickGetAxis',
    '''Get the current state of an axis control on a joystick.

    The axis indices start at index 0.
    
    :Parameters:
     - `joystick`: `SDL_Joystick_p`
     - `axis`: int

    :rtype: int
    :return: a value ranging from -32,768 to 32767.
    ''',
    args=['joystick', 'axis'],
    arg_types=[SDL_Joystick_p, c_int],
    return_type=c_short)

SDL_JoystickGetHat = SDL.dll.function('SDL_JoystickGetHat',
    '''Get the current state of POV hat on a joystick.

    The hat indices start at index 0.
    
    :Parameters:
     - `joystick`: `SDL_Joystick_p`
     - `hat`: int

    :rtype: int
    :return: one of `SDL_HAT_CENTERED`, `SDL_HAT_UP`, `SDL_HAT_LEFT`,
        `SDL_HAT_DOWN`, `SDL_HAT_RIGHT`, `SDL_HAT_RIGHTUP`,
        `SDL_HAT_RIGHTDOWN`, `SDL_HAT_RIGHTUP`, `SDL_HAT_LEFTUP`,
        `SDL_HAT_LEFTDOWN`.
    ''',
    args=['joystick', 'hat'],
    arg_types=[SDL_Joystick_p, c_int],
    return_type=c_ubyte)

_SDL_JoystickGetBall = SDL.dll.private_function('SDL_JoystickGetBall',
    arg_types=[SDL_Joystick_p, c_int, POINTER(c_int), POINTER(c_int)],
    return_type=c_int,
    error_return=-1)

def SDL_JoystickGetBall(joystick, ball):
    '''Get the ball axis change since the last poll.

    The ball indicies start at index 0.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`
     - `ball`: int

    :rtype: (int, int)
    :return: a tuple (dx, dy) of the relative motion of the ball.
    '''
    dx, dy = c_int(), c_int()
    _SDL_JoystickGetBall(joystick, ball, byref(x), byref(y))
    return dx.value, dy.value

SDL_JoystickGetButton = SDL.dll.function('SDL_JoystickGetButton',
    '''Get the current state of a button on a joystick.

    The button indices start at index 0.
    
    :Parameters:
     - `joystick`: `SDL_Joystick_p`
     - `button`: int

    :rtype: int
    :return: undocumented
    ''',
    args=['joystick', 'button'],
    arg_types=[SDL_Joystick_p, c_int],
    return_type=c_ubyte)

SDL_JoystickClose = SDL.dll.function('SDL_JoystickClose',
    '''Close a joystick previously opened with `SDL_JoystickOpen`.

    :Parameters:
     - `joystick`: `SDL_Joystick_p`

    ''',
    args=['joystick'],
    arg_types=[SDL_Joystick_p],
    return_type=None)
