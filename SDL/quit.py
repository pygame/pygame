#!/usr/bin/env python

'''Quit event handling.

An `SDL_QUITEVENT` is generated when the user tries to close the application
window.  If it is ignored or filtered out, the window will remain open.
If it is not ignored or filtered, it is queued normally and the window
is allowed to close.  When the window is closed, screen updates will 
complete, but have no effect.

`SDL_Init` installs signal handlers for SIGINT (keyboard interrupt)
and SIGTERM (system termination request), if handlers do not already
exist, that generate `SDL_QUITEVENT` events as well.  There is no way
to determine the cause of an `SDL_QUITEVENT`, but setting a signal
handler in your application will override the default generation of
quit events for that signal.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import SDL.events

def SDL_QuitRequested():
    '''Return True if there is a quit event in the event queue.

    :rtype: bool
    '''
    SDL.events.SDL_PumpEvents()
    return SDL.events.SDL_HaveEvents(SDL.events.SDL_QUITMASK)
