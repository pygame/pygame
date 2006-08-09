#!/usr/bin/env python

'''Time management routines.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll

SDL_GetTicks = SDL.dll.function('SDL_GetTicks',
    '''Get the number of milliseconds since the SDL library initialization.

    Note that this value wraps if the program runs for more than ~49 days.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_uint)

SDL_Delay = SDL.dll.function('SDL_Delay',
    '''Wait a specified number of milliseconds before returning.

    :Parameters:
        `ms` : int
            delay in milliseconds

    ''',
    args=['ms'],
    arg_types=[c_uint],
    return_type=None)

_SDL_TimerCallback = CFUNCTYPE(c_int, c_uint)
_SDL_SetTimer = SDL.dll.private_function('SDL_SetTimer',
    arg_types=[c_uint, _SDL_TimerCallback],
    return_type=c_int)

_timercallback_ref = None   # Keep global to avoid possible GC

def SDL_SetTimer(interval, callback):
    '''Set a callback to run after the specified number of milliseconds has
    elapsed.

    The callback function is passed the current timer interval
    and returns the next timer interval.  If the returned value is the 
    same as the one passed in, the periodic alarm continues, otherwise a
    new alarm is scheduled.  If the callback returns 0, the periodic alarm
    is cancelled.

    To cancel a currently running timer, call ``SDL_SetTimer(0, None)``.

    The timer callback function may run in a different thread than your
    main code, and so shouldn't call any functions from within itself.

    The maximum resolution of this timer is 10 ms, which means that if
    you request a 16 ms timer, your callback will run approximately 20 ms
    later on an unloaded system.  If you wanted to set a flag signaling
    a frame update at 30 frames per second (every 33 ms), you might set a 
    timer for 30 ms::

          SDL_SetTimer((33/10)*10, flag_update)

    If you use this function, you need to pass `SDL_INIT_TIMER` to
    `SDL_Init`.

    Under UNIX, you should not use raise or use SIGALRM and this function
    in the same program, as it is implemented using ``setitimer``.  You
    also should not use this function in multi-threaded applications as
    signals to multi-threaded apps have undefined behavior in some
    implementations.

    :Parameters:
        `interval` : int
            Interval before callback, in milliseconds.
        `callback` : function
            Callback function should accept one argument, the number of
            milliseconds elapsed, and return the next timer interval,
            in milliseconds.
    '''

    # Note SDL_SetTimer actually returns 1 on success, not 0 as documented
    # in SDL_timer.h.
    global _timercallback_ref
    if callback:
        _timercallback_ref = _SDL_TimerCallback(callback)
    else:
        _timercallback_ref = _SDL_TimerCallback()
    
    # XXX if this fails the global ref is incorrect and old one will
    # possibly be collected early.
    if _SDL_SetTimer(interval, _timercallback_ref) == -1:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()


# For the new timer functions, the void *param passed to the callback
# is ignored; using an local function instead.  The SDL_TimerID type
# is not defined, we use c_void_p instead.

_SDL_NewTimerCallback = CFUNCTYPE(c_uint, c_int, c_void_p)
_SDL_AddTimer = SDL.dll.private_function('SDL_AddTimer',
    arg_types=[c_uint, _SDL_NewTimerCallback, c_void_p],
    return_type=c_void_p)

_timer_refs = {}        # Keep global to manage GC

def SDL_AddTimer(interval, callback, param):
    '''Add a new timer to the pool of timers already running.

    :Parameters:
        `interval` : int
            The interval before calling the callback, in milliseconds.
        `callback` : function
            The callback function.  It is passed the current timer
            interval, in millseconds, and returns the next timer interval,
            in milliseconds.  If the returned value is the same as the one
            passed in, the periodic alarm continues, otherwise a new alarm
            is scheduled.  If the callback returns 0, the periodic alarm is
            cancelled.  An example callback function is::

                def timer_callback(interval, param):
                    print 'timer called after %d ms.' % interval
                    return 1000     # call again in 1 second

        `param` : any
            A value passed to the callback function.
    
    :rtype: int
    :return: the timer ID
    '''
    def _callback(interval, _ignored_param):
        return callback(interval, param)
    
    func = _SDL_NewTimerCallback(_callback)
    result = _SDL_AddTimer(interval, func, None)
    if not result:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
    _timer_refs[result] = func
    return result

_SDL_RemoveTimer = SDL.dll.private_function('SDL_RemoveTimer',
    args=['t'],
    arg_types=[c_void_p],
    return_type=c_int,
    error_return=0)

def SDL_RemoveTimer(t):
    '''Remove one of the multiple timers knowing its ID.

    :Parameters:
        `t` : int
            The timer ID, as returned by `SDL_AddTimer`.

    '''
    global _timer_refs
    _SDL_RemoveTimer(t)
    del _timer_refs[t]

