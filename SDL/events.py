#!/usr/bin/env python

'''Event handling.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.constants
import SDL.dll
import SDL.keyboard

class SDL_ActiveEvent(Structure):
    '''Application visibility event structure.

    :see: `SDL_GetAppState`

    :Ivariables:
        `type` : int
            SDL_ACTIVEEVENT
        `gain` : int
            1 if states were gained, zero otherwise
        `state` : int
            A mask of the focus states.  A bitwise OR combination of:
            SDL_APPMOUSEFOCUS, SDL_APPINPUTFOCUS and SDL_APPACTIVE.

    '''
    _fields_ = [('type', c_ubyte),
                ('gain', c_ubyte),
                ('state', c_ubyte)]

class SDL_KeyboardEvent(Structure):
    '''Keyboard event structure.

    :Ivariables:
        `type` : int
            SDL_KEYDOWN or SDL_KEYUP
        `which` : int
            The keyboard device index
        `state` : int
            SDL_PRESSED or SDL_RELEASED
        `keysym` : `SDL_keysym`
            Decoded key information.

    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('state', c_ubyte),
                ('keysym', SDL.keyboard.SDL_keysym)]

class SDL_MouseMotionEvent(Structure):
    '''Mouse motion event structure.

    :Ivariables:
        `type` : int
            SDL_MOUSEMOTION
        `which` : int
            The mouse device index
        `state` : int
            The current button state
        `x` : int
            The X coordinate of the mouse pointer
        `y` : int
            The Y coordinate of the mouse pointer
        `xrel` : int
            The relative motion in the X direction
        `yrel` : int
            The relative motion in the Y direction

    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('state', c_ubyte),
                ('x', c_ushort),
                ('y', c_ushort),
                ('xrel', c_short),
                ('yrel', c_short)]

class SDL_MouseButtonEvent(Structure):
    '''Mouse button event structure.

    :Ivariables:
        `type` : int
            SDL_MOUSEBUTTONDOWN or SDL_MOUSEBUTTONUP
        `which` : int
            The mouse device index
        `button` : int
            The mouse button index
        `state` : int
            SDL_PRESSED or SDL_RELEASED
        `x` : int
            The X coordinate of the mouse pointer
        `y` : int
            The Y coordinate of the mouse pointer
    
    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('button', c_ubyte),
                ('state', c_ubyte),
                ('x', c_ushort),
                ('y', c_ushort)]

class SDL_JoyAxisEvent(Structure):
    '''Joystick axis motion event structure.

    :Ivariables:
        `type` : int
            SDL_JOYAXISMOTION
        `which` : int
            The joystick device index
        `axis` : int
            The joystick axis index
        `value` : int
            The axis value, in range [-32768, 32767]
    
    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('axis', c_ubyte),
                ('value', c_short)]

class SDL_JoyBallEvent(Structure):
    '''Joystick trackball motion event structure.

    :Ivariables:
        `type` : int
            SDL_JOYBALLMOTION
        `which` : int
            The joystick device index
        `ball` : int
            The joystick trackball index
        `xrel` : int
            The relative motion in the X direction
        `yrel` : int
            The relative motion in the Y direction
    
    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('ball', c_ubyte),
                ('xrel', c_short),
                ('yrel', c_short)]

class SDL_JoyHatEvent(Structure):
    '''Joystick hat position change event structure.

    :Ivariables:
        `type` : int
            SDL_JOYHATMOTION
        `which` : int
            The joystick device index
        `hat` : int
            The joystick hat index
        `value` : int
            The hat position value.  One of: SDL_HAT_LEFTUP, SDL_HAT_UP,
            SDL_HAT_RIGHTUP, SDL_HAT_LEFT, SDL_HAT_CENTERED, SDL_HAT_RIGHT,
            SDL_HAT_LEFTDOWN, SDL_HAT_DOWN, SDL_HAT_RIGHTDOWN.  Note that
            zero means the POV is centered.
    
    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('hat', c_ubyte),
                ('value', c_ubyte)]

class SDL_JoyButtonEvent(Structure):
    '''Joystick button event structure.

    :Ivariables:
        `type` : int
            SDL_JOYBUTTONDOWN or SDL_JOYBUTTONUP
        `which` : int
            The joystick device index
        `button` : int
            The joystick button index
        `state` : int
            SDL_PRESSED or SDL_RELEASED

    '''
    _fields_ = [('type', c_ubyte),
                ('which', c_ubyte),
                ('button', c_ubyte),
                ('state', c_ubyte)]

class SDL_ResizeEvent(Structure):
    '''The window resized event structure.

    :Ivariables:
        `type` : int
            SDL_VIDEORESIZE
        `w` : int
            New width
        `h` : int
            New height
    
    '''
    _fields_ = [('type', c_ubyte),
                ('w', c_int),
                ('h', c_int)]

class SDL_ExposeEvent(Structure):
    '''The screen redraw event structure.

    :Ivariables:
        `type` : int
            SDL_VIDEOEXPOSE
    
    '''
    _fields_ = [('type', c_ubyte)]

class SDL_QuitEvent(Structure):
    '''The quit requested event structure

    :Ivariables:
        `type` : int
            SDL_QUIT
    
    '''
    _fields_ = [('type', c_ubyte)]

class SDL_UserEvent(Structure):
    '''A user-defined event structure.

    :Ivariables:
        `type` : int
            SDL_USEREVENT through SDL_NUMEVENTS - 1
        `code` : int
            User defined event code
    
    '''
    # pygame-ctypes needs data1 and data2 to be c_void_p; POINTER(c_ubyte)
    # or similar will break it (see pygame/event.py) unless another workaround
    # can be found.
    _fields_ = [('type', c_ubyte),
                ('code', c_int),
                ('data1', c_void_p),
                ('data2', c_void_p)]

class SDL_SysWMEvent(Structure):
    '''System window management event structure.

    :Ivariables:
        `type` : int
            SDL_SYSWMEVENT
    
    '''
    _fields_ = [('type', c_ubyte),
                ('msg', c_void_p)]  # TODO with SDL_syswm.h

class SDL_Event(Union):
    '''Union event structure.

    Events returned from functions are always returned as the
    specialised subclass; for example you will receive a
    `SDL_MouseMotionEvent` rather than `SDL_Event`.  This structure
    therefore has limited application use, but is used internally.

    :Ivariables:
        `type` : int
            Type of event
    
    '''
    _fields_ = [('type', c_ubyte),
                ('active', SDL_ActiveEvent),
                ('key', SDL_KeyboardEvent),
                ('motion', SDL_MouseMotionEvent),
                ('button', SDL_MouseButtonEvent),
                ('jaxis', SDL_JoyAxisEvent),
                ('jball', SDL_JoyBallEvent),
                ('jhat', SDL_JoyHatEvent),
                ('jbutton', SDL_JoyButtonEvent),
                ('resize', SDL_ResizeEvent),
                ('expose', SDL_ExposeEvent),
                ('quit', SDL_QuitEvent),
                ('user', SDL_UserEvent),
                ('syswm', SDL_SysWMEvent)]

    types = {
        SDL.constants.SDL_ACTIVEEVENT: (SDL_ActiveEvent, 'active'),
        SDL.constants.SDL_KEYDOWN: (SDL_KeyboardEvent, 'key'),
        SDL.constants.SDL_KEYUP: (SDL_KeyboardEvent, 'key'),
        SDL.constants.SDL_MOUSEMOTION: (SDL_MouseMotionEvent, 'motion'),
        SDL.constants.SDL_MOUSEBUTTONDOWN: (SDL_MouseButtonEvent, 'button'),
        SDL.constants.SDL_MOUSEBUTTONUP: (SDL_MouseButtonEvent, 'button'),
        SDL.constants.SDL_JOYAXISMOTION: (SDL_JoyAxisEvent, 'jaxis'),
        SDL.constants.SDL_JOYBALLMOTION: (SDL_JoyBallEvent, 'jball'),
        SDL.constants.SDL_JOYHATMOTION: (SDL_JoyHatEvent, 'jhat'),
        SDL.constants.SDL_JOYBUTTONDOWN: (SDL_JoyButtonEvent, 'jbutton'),
        SDL.constants.SDL_JOYBUTTONUP: (SDL_JoyButtonEvent, 'jbutton'),
        SDL.constants.SDL_VIDEORESIZE: (SDL_ResizeEvent, 'resize'),
        SDL.constants.SDL_VIDEOEXPOSE: (SDL_ExposeEvent, 'expose'),
        SDL.constants.SDL_QUIT: (SDL_QuitEvent, 'quit'),
        SDL.constants.SDL_SYSWMEVENT: (SDL_SysWMEvent, 'syswm')
    }

    def __init__(self, typecode=0):
        self.type = typecode

    def __repr__(self):
        if self.type in self.types:
            return self.types[self.type][0].__repr__(self)
        elif self.type >= SDL.constants.SDL_USEREVENT: 
            # SDL_MAXEVENTS not defined
            return SDL_UserEvent.__repr__(self)
        return 'SDLEvent(type=%d)' % self.type

    def specialize(self):
        '''Get an instance of the specialized subclass for this event,
        based on `self.type`.

        :rtype: `SDL_Event` subclass
        '''
        if self.type in self.types:
            return getattr(self, self.types[self.type][1])
        elif self.type >= SDL.constants.SDL_USEREVENT:    
            # SDL_MAXEVENTS not defined
            return self.user
        return self

SDL_PumpEvents = SDL.dll.function('SDL_PumpEvents',
    '''Pumps the event loop, gathering events from the input devices.

    This function updates the event queue and internal input device state.
    This should only be run in the thread that sets the video mode.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

_SDL_PeepEvents = SDL.dll.private_function('SDL_PeepEvents',
    arg_types=[POINTER(SDL_Event), c_int, c_int, c_uint],
    return_type=c_int)

def SDL_PeepEvents(numevents, action, mask):
    '''Check the event queue for messages and optionally return them.

    The behaviour depends on `action`:

        `SDL_ADDEVENT`
            Not implemented; will raise an exception.  See `SDL_PushEvent`.
        `SDL_PEEKEVENT`
            Up to `numevents` events at the front of the event queue,
            matching `mask`, will be returned and will not be removed
            from the queue.
        `SDL_GETEVENT`
            Up to `numevents` events at the front of the event queue,
            matching `mask`, will be returned and will be removed from the
            queue.

    :Parameters:
        `numevents` : int
            Maximum number of events to return
        `action` : int
            Either `SDL_PEEKEVENT` or `SDL_GETEVENT`
        `mask` : int
            Mask to match type of returned events with.

    :rtype: list
    :return: list of SDL_Event (or subclass)
    :see: `SDL_PushEvent`, `SDL_HaveEvents`
    '''
    if action == SDL.constants.SDL_ADDEVENT:
        raise NotImplementedError, 'Use SDL_PushEvent to add events'
    ar = (SDL_Event * numevents)()
    num = _SDL_PeepEvents(ar, numevents, action, mask)
    if num == -1:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
    return list([e.specialize() for e in ar[:num]])

def SDL_HaveEvents(mask):
    '''Check the event queue for events matching the given mask.

    :note: This function replaces the C function 
        SDL_PeepEvents(NULL, ...), which had undocumented behaviour.

    :Parameters:
        `mask` : int
            Mask to match type of returned events with.

    :rtype: bool
    :return: True if at least one event matches the mask in the event
        queue.
    '''
    num = _SDL_PeepEvents(None, 1, SDL.constants.SDL_PEEKEVENT, mask)
    return num > 0

_SDL_PollEvent = SDL.dll.private_function('SDL_PollEvent',
    arg_types=[POINTER(SDL_Event)],
    return_type=c_int)

def SDL_PollEvent():
    '''Poll for currently pending events.

    Returns True if there are any pending events, or False if there are none
    available.

    :see: `SDL_PollEventAndReturn`
    :rtype: bool
    '''
    return _SDL_PollEvent(None) == 1
    
def SDL_PollEventAndReturn():
    '''Poll for currently pending events, and return one off the queue
    if possible.

    :see: `SDL_PollEvent`
    :rtype: `SDL_Event` or subclass
    '''
    e = SDL_Event()
    result = _SDL_PollEvent(byref(e))
    if result == 1:
        return e.specialize()
    return None

_SDL_WaitEvent = SDL.dll.private_function('SDL_WaitEvent',
    arg_types=[POINTER(SDL_Event)],
    return_type=c_int)

def SDL_WaitEvent():
    '''Wait indefinitely for an event.

    Returns when an event is available on the queue, or raises an exception
    if an error occurs while waiting.

    :see: `SDL_WaitEventAndReturn`
    '''
    if _SDL_WaitEvent(None) == 0:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
    
def SDL_WaitEventAndReturn():
    '''Wait indefinitely for the next event and return it.

    :see: `SDL_WaitEvent`
    :rtype: `SDL_Event` or subclass
    '''
    ev = SDL_Event()
    result = _SDL_WaitEvent(byref(ev))
    if result == 1:
        return ev.specialize()
    else:
        raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()

SDL_PushEvent = SDL.dll.function('SDL_PushEvent',
    '''Add an event to the event queue.

    :Parameters:
     - `event`: `SDL_Event`

    ''',
    args=['event'],
    arg_types=[POINTER(SDL_Event)],
    return_type=c_int,
    error_return=-1)


_SDL_EventFilter = CFUNCTYPE(c_int, POINTER(SDL_Event))
_SDL_SetEventFilter = SDL.dll.private_function('SDL_SetEventFilter',
    arg_types=[_SDL_EventFilter],
    return_type=None)

_eventfilter_ref = None     # keep global to avoid GC of anon func

def SDL_SetEventFilter(filter):
    '''Set up a filter to process all events before they change internal
    state and are posted to the internal event queue.

    :warning:  Be very careful of what you do in the event filter function,
        as it may run in a different thread.

    There is one caveat when dealing with the `SDL_QUITEVENT` event type.
    The event filter is only called when the window manager desires to
    close the application window.  If the event filter returns 1, then the
    window will be closed, otherwise the window will remain open if
    possible.  If the quit event is generated by an interrupt signal, it
    will bypass the internal queue and be delivered to the application at
    the next event poll.

    :Parameters:
        `filter` : function
            a function that takes an `SDL_Event` as a single argument.  If
            the function returns 1 the event will be added to the internal
            queue.  If it returns 0, the event will be dropped from the
            queue, but the internal state will still be updated.  The event
            instance must not be modified.

    '''
    global _eventfilter_ref
    if filter:
        def f(e):
            return filter(e.contents.specialize())
        _eventfilter_ref = _SDL_EventFilter(f)
    else:
        _eventfilter_ref = _SDL_EventFilter()
    _SDL_SetEventFilter(_eventfilter_ref)

SDL_GetEventFilter = SDL.dll.function('SDL_GetEventFilter',
    '''Return the current event filter.

    This can be used to "chain" filters.  If there is no event filter set,
    this function returns None.

    :rtype: function
    ''',
    args=[],
    arg_types=[],
    return_type=_SDL_EventFilter)

SDL_EventState = SDL.dll.function('SDL_EventState',
    '''Ignore or enable the processing of certain events.

    The behaviour of this function depends on `state`

        `SDL_IGNORE`
            the event will be automatically dropped from the event queue
            and will not be event filtered.
        `SDL_ENABLE`
            the event will be processed normally.
        `SDL_QUERY`
            return the current processing state of the event.

    :Parameters:
        `type` : int
            Type of event, e.g. `SDL_KEYDOWN`, `SDL_MOUSEMOTION`, etc.
            (see `SDL_Event`)
        `state` : int
            One of `SDL_IGNORE`, `SDL_ENABLE` or `SDL_QUERY`

    :rtype: int
    :return: the event processing state: either `SDL_IGNORE or
        `SDL_ENABLE`.
    ''',
    args=['type', 'state'],
    arg_types=[c_ubyte, c_int],
    return_type=c_int)
