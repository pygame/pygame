#!/usr/bin/env python

'''Pygame module for interacting with events and queues.

Pygame handles all it's event messaging through an event queue. The routines
in this module help you manage that event queue. The input queue is heavily
dependent on the pygame display module. If the display has not been
initialized and a video mode not set, the event queue will not really work.
 
The queue is a regular queue of Event objects, there are a variety of ways
to access the events it contains. From simply checking for the existance of
events, to grabbing them directly off the stack.
 
All events have a type identifier. This event type is in between the values
of NOEVENT and NUMEVENTS. All user defined events can have the value of
USEREVENT or higher. It is recommended make sure your event id's follow this
system.
 
To get the state of various input devices, you can forego the event queue
and access the input devices directly with their appropriate modules; mouse,
key, and joystick. If you use this method, remember that pygame requires some
form of communication with the system window manager and other parts of the
platform. To keep pygame in synch with the system, you will need to call
pygame.event.pump() to keep everything current. You'll want to call this
function usually once per game loop.
 
The event queue offers some simple filtering. This can help performance
slightly by blocking certain event types from the queue, use the
pygame.event.set_allowed() and pygame.event.set_blocked() to work with
this filtering. All events default to allowed.
 
Joysticks will not send any events until the device has been initialized.

An Event object contains an event type and a readonly set of member data.
The Event object contains no method functions, just member data. Event
objects are retrieved from the pygame event queue. You can create your
own new events with the pygame.event.Event() function.

Your program must take steps to keep the event queue from overflowing. If the
program is not clearing or getting all events off the queue at regular
intervals, it can overflow.  When the queue overflows an exception is thrown.
 
All Event objects contain an event type identifier in the Event.type member.
You may also get full access to the Event's member data through the Event.dict
method. All other member lookups will be passed through to the Event's
dictionary values.
 
While debugging and experimenting, you can print the Event objects for a
quick display of its type and members. Events that come from the system
will have a guaranteed set of member items based on the type. Here is a
list of the Event members that are defined with each type.

QUIT         
    (none)
ACTIVEEVENT      
    gain, state
KEYDOWN      
    unicode, key, mod
KEYUP        
    key, mod
MOUSEMOTION      
    pos, rel, buttons
MOUSEBUTTONUP    
    pos, button
MOUSEBUTTONDOWN  
    pos, button
JOYAXISMOTION       
    joy, axis, value
JOYBALLMOTION    
    joy, ball, rel
JOYHATMOTION     
    joy, hat, value
JOYBUTTONUP         
    joy, button
JOYBUTTONDOWN       
    joy, button
VIDEORESIZE         
    size, w, h
VIDEOEXPOSE         
    (none)
USEREVENT           
    code
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from copy import copy
from ctypes import *

from SDL import *
import pygame.base
import pygame.constants

def _video_init_check():
    if not SDL_WasInit(SDL_INIT_VIDEO):
        raise pygame.base.error, 'video system not initialized'

def pump():
    '''Internally process pygame event handlers.

    For each frame of your game, you will need to make some sort of call to
    the event queue. This ensures your program can internally interact with
    the rest of the operating system. If you are not using other event
    functions in your game, you should call pygame.event.pump() to allow
    pygame to handle internal actions.

    This function is not necessary if your program is consistently processing
    events on the queue through the other pygame.event functions.
     
    There are important things that must be dealt with internally in the event
    queue. The main window may need to be repainted or respond to the system.
    If you fail to make a call to the event queue for too long, the system may
    decide your program has locked up.
    '''
    _video_init_check()
    SDL_PumpEvents()

def get(typelist=None):
    '''Get events from the queue.
    pygame.event.get(): return Eventlist
    pygame.event.get(type): return Eventlist
    pygame.event.get(typelist): return Eventlist

    This will get all the messages and remove them from the queue. If a type
    or sequence of types is given only those messages will be removed from the
    queue.

    If you are only taking specific events from the queue, be aware that the
    queue could eventually fill up with the events you are not interested.

    :Parameters:
        `typelist` : int or sequence of int
            Event type or list of event types that can be returned.

    :rtype: list of `Event`
    '''
    _video_init_check()
    
    if not typelist:
        mask = SDL_ALLEVENTS
    else:
        if hasattr(typelist, '__len__'):
            mask = reduce(lambda a,b: a | SDL_EVENTMASK(b), typelist)
        else:
            mask = int(typelist)

    SDL_PumpEvents()

    events = []
    new_events = SDL_PeepEvents(1, SDL_GETEVENT, mask)
    while new_events:
        events.append(Event(0, sdl_event=new_events[0]))
        new_events = SDL_PeepEvents(1, SDL_GETEVENT, mask)
    return events


def poll():
    '''Get a single event from the queue.

    Returns a single event from the queue. If the event queue is empty an event
    of type pygame.NOEVENT will be returned immediately. The returned event is
    removed from the queue.

    :rtype: Event
    '''
    _video_init_check()

    event = SDL_PollEventAndReturn()
    if event:
        return Event(0, sdl_event=event, keep_userdata=True)
    else:
        return Event(pygame.constants.NOEVENT)

def wait():
    '''Wait for a single event from the queue.

    Returns a single event from the queue. If the queue is empty this function
    will wait until one is created. While the program is waiting it will sleep
    in an idle state. This is important for programs that want to share the
    system with other applications.

    :rtype: Event
    '''
    _video_init_check()

    return Event(0, sdl_event=SDL_WaitEventAndReturn())

def peek(typelist=None):
    '''Test if event types are waiting on the queue.

    Returns true if there are any events of the given type waiting on the
    queue.  If a sequence of event types is passed, this will return True if
    any of those events are on the queue.

    :Parameters:
        `typelist` : int or sequence of int
            Event type or list of event types to look for.

    :rtype: bool
    '''
    _video_init_check()

    if not typelist:
        mask = SDL_ALLEVENTS
    else:
        if hasattr(typelist, '__len__'):
            mask = reduce(lambda a,b: a | SDL_EVENTMASK(b), typelist)
        else:
            mask = int(typelist)
    
    SDL_PumpEvents()
    events = SDL_PeepEvents(1, SDL_PEEKEVENT, mask)

    if not typelist:
        if events:
            return Event(0, sdl_event=events[0])
        else:
            return Event(pygame.constants.NOEVENT) # XXX deviation from pygame
    return len(events) > 0

def clear(typelist=None):
    '''Remove all events from the queue.

    Remove all events or events of a specific type from the queue. This has the
    same effect as `get` except nothing is returned. This can be slightly more
    effecient when clearing a full event queue.

    :Parameters:
        `typelist` : int or sequence of int
            Event type or list of event types to remove.
    
    '''
    _video_init_check()
    
    if not typelist:
        mask = SDL_ALLEVENTS
    else:
        if hasattr(typelist, '__len__'):
            mask = reduce(lambda a,b: a | SDL_EVENTMASK(b), typelist)
        else:
            mask = int(typelist)

    SDL_PumpEvents()
    
    events = []
    new_events = SDL_PeepEvents(1, SDL_GETEVENT, mask)
    while new_events:
        new_events = SDL_PeepEvents(1, SDL_GETEVENT, mask)

_event_names = {
    SDL_ACTIVEEVENT:    'ActiveEvent',
    SDL_KEYDOWN:        'KeyDown',
    SDL_KEYUP:          'KeyUp',
    SDL_MOUSEMOTION:    'MouseMotion',
    SDL_MOUSEBUTTONDOWN:'MouseButtonDown',
    SDL_MOUSEBUTTONUP:  'MouseButtonUp',
    SDL_JOYAXISMOTION:  'JoyAxisMotion',
    SDL_JOYBALLMOTION:  'JoyBallMotion',
    SDL_JOYHATMOTION:   'JoyHatMotion',
    SDL_JOYBUTTONUP:    'JoyButtonUp',
    SDL_JOYBUTTONDOWN:  'JoyButtonDown',
    SDL_QUIT:           'Quit',
    SDL_SYSWMEVENT:     'SysWMEvent',
    SDL_VIDEORESIZE:    'VideoResize',
    SDL_VIDEOEXPOSE:    'VideoExpose',
    SDL_NOEVENT:        'NoEvent'
}

def event_name(event_type):
    '''Get the string name from an event id.

    Pygame uses integer ids to represent the event types. If you want to
    report these types to the user they should be converted to strings. This
    will return a the simple name for an event type. The string is in the
    CamelCase style.

    :Parameters:
     - `event_type`: int
          
    :rtype: str
    '''
    if event_type >= SDL_USEREVENT and event_type < SDL_NUMEVENTS:
        return 'UserEvent'
    return _event_names.get(event_type, 'Unknown')

def set_blocked(typelist):
    '''Control which events are allowed on the queue.

    The given event types are not allowed to appear on the event queue. By
    default all events can be placed on the queue. It is safe to disable an
    event type multiple times.

    If None is passed as the argument, this has the opposite effect and none of
    the event types are allowed to be placed on the queue.

    :Parameters:
        `typelist` : int or sequence of int or None
            Event type or list of event types to disallow.

    '''
    _video_init_check()

    if typelist == None:
        SDL_EventState(SDL_ALLEVENTS, SDL_IGNORE)
    elif hasattr(typelist, '__len__'):
        for val in typelist:
            SDL_EventState(val, SDL_IGNORE)
    else:
        SDL_EventState(typelist, SDL_IGNORE)

def set_allowed(typelist):
    '''Control which events are allowed on the queue.

    The given event types are allowed to appear on the event queue. By default
    all events can be placed on the queue. It is safe to enable an event type
    multiple times.

    If None is passed as the argument, this has the opposite effect and all of
    the event types are allowed to be placed on the queue.

    :Parameters:
        `typelist` : int or sequence of int or None
            Event type or list of event types to disallow.

    '''
    _video_init_check()

    if typelist == None:
        SDL_EventState(SDL_ALLEVENTS, SDL_ENABLE)
    elif hasattr(typelist, '__len__'):
        for val in typelist:
            SDL_EventState(val, SDL_ENABLE)
    else:
        SDL_EventState(typelist, SDL_ENABLE)

def get_blocked(event_type):
    '''Test if a type of event is blocked from the queue.

    Returns true if the given event type is blocked from the queue.

    :Parameters:
     - `event_type`: int

    :rtype: int
    '''
    _video_init_check()

    if typelist == None:
        return SDL_EventState(SDL_ALLEVENTS, SDL_QUERY) == SDL_ENABLE
    elif hasattr(typelist, '__len__'):  # XXX undocumented behaviour
        for val in typelist:
            if SDL_EventState(val, SDL_QUERY) == SDL_ENABLE:
                return True
        return False
    else:
        return SDL_EventState(typelist, SDL_QUERY) == SDL_ENABLE
    
def set_grab(grab):
    '''Control the sharing of input devices with other applications.

    When your program runs in a windowed environment, it will share the mouse
    and keyboard devices with other applications that have focus. If your 
    program sets the event grab to True, it will lock all input into your
    program. 

    It is best to not always grab the input, since it prevents the user from
    doing other things on their system.

    :Parameters:
     - `grab`: bool

    '''
    _video_init_check()

    if grab:
        SDL_WM_GrabInput(SDL_GRAB_ON)
    else:
        SDL_WM_GrabInput(SDL_GRAB_OFF)
    
def get_grab():
    '''Test if the program is sharing input devices.

    Returns true when the input events are grabbed for this application. Use
    `set_grab` to control this state.

    :rtype: bool
    '''
    _video_init_check()

    return SDL_WM_GrabInput(SDL_GRAB_QUERY) == SDL_GRAB_ON

_USEROBJECT_CHECK1 = 0xdeadbeef
_USEROBJECT_CHECK2 = 0xfeedf00d
_user_event_objects = {}
_user_event_nextid = 0
    
def post(event):
    '''Place a new event on the queue.

    This places a new event at the end of the event queue. These Events will
    later be retrieved from the other queue functions.

    This is usually used for placing pygame.USEREVENT events on the queue.
    Although any type of event can be placed, if using the sytem event types
    your program should be sure to create the standard attributes with
    appropriate values.

    :Parameters:
        `event` : Event
            Event to add to the queue.

    '''
    global _user_event_nextid

    _video_init_check()

    sdl_event = SDL_Event(pygame.constants.USEREVENT)
    sdl_event.code = _USEROBJECT_CHECK1
    sdl_event.data1 = c_void_p(_USEROBJECT_CHECK2)
    sdl_event.data2 = c_void_p(_user_event_nextid)
    _user_event_objects[_user_event_nextid] = event
    _user_event_nextid += 1

    SDL_PushEvent(sdl_event)

class Event:
    def __init__(self, event_type, event_dict=None, sdl_event=None, 
                 keep_userdata=False, **attributes):
        '''Create a new event object.

        Creates a new event with the given type. The event is created with the
        given attributes and values. The attributes can come from a dictionary
        argument, or as string keys from a dictionary. 

        The given attributes will be readonly attributes on the new event
        object itself. These are the only attributes on the Event object,
        there are no methods attached to Event objects.

        :Parameters:
            `event_type` : int
                Event type to create
            `event_dict` : dict
                Dictionary of attributes to assign.
            `sdl_event` : `SDL_Event`
                Construct a Pygame event from the given SDL_Event; used
                internally.
            `keep_userdata` : bool
                Used internally.
            `attributes` : additional keyword arguments
                Additional attributes to assign to the event.

        '''
        if sdl_event:
            if sdl_event.type == pygame.constants.USEREVENT and \
               sdl_event.code == _USEROBJECT_CHECK1 and \
               sdl_event.data1.value == _USEROBJECT_CHECK2 and \
               sdl_event.data2.value in _user_event_objects:
                # An event that was posted; grab dict from local store.
                id = sdl_event.data2.value
                for key, value in _user_event_objects[id].items():
                    setattr(self, key, value)
                # Free memory unless just peeking
                if not keep_userdata:
                    del _user_event_objects[id]
            else:    
                # Standard SDL event
                self.type = sdl_event.type
                if self.type == SDL_QUIT:
                    pass
                elif self.type == SDL_ACTIVEEVENT:
                    self.gain = sdl_event.gain
                    self.state = sdl_event.state
                elif self.type == SDL_KEYDOWN:
                    self.unicode = sdl_event.keysym.unicode
                    self.key = sdl_event.keysym.sym
                    self.mod = sdl_event.keysym.mod
                elif self.type == SDL_KEYUP:
                    self.key = sdl_event.keysym.sym
                    self.mod = sdl_event.keysym.mod
                elif self.type == SDL_MOUSEMOTION:
                    self.pos = (sdl_event.x, sdl_event.y)
                    self.rel = (sdl_event.xrel, sdl_event.yrel)
                    self.buttons = (sdl_event.state & SDL_BUTTON(1) != 0,
                                    sdl_event.state & SDL_BUTTON(2) != 0,
                                    sdl_event.state & SDL_BUTTON(3) != 0)
                elif self.type in (SDL_MOUSEBUTTONDOWN, SDL_MOUSEBUTTONUP):
                    self.pos = (sdl_event.x, sdl_event.y)
                    self.button = sdl_event.button
                elif self.type == SDL_JOYAXISMOTION:
                    self.joy = sdl_event.which
                    self.axis = sdl_event.axis
                    self.value = sdl_event.value / 32767.0
                elif self.type == SDL_JOYBALLMOTION:
                    self.joy = sdl_event.which
                    self.ball = sdl_event.ball
                    self.rel = (sdl_event.xrel, sdl_event.yrel)
                elif self.type == SDL_JOYHATMOTION:
                    self.joy = sdl_event.which
                    self.hat = sdl_event.hat
                    hx = hy = 0
                    if sdl_event.value & SDL_HAT_UP:
                        hy = 1
                    if sdl_event.value & SDL_HAT_DOWN:
                        hy = -1
                    if sdl_event.value & SDL_HAT_RIGHT:
                        hx = 1
                    if sdl_event.value & SDL_HAT_LEFT:
                        hx = -1
                    self.value = (hx, hy)
                elif self.type in (SDL_JOYBUTTONUP, SDL_JOYBUTTONDOWN):
                    self.joy = sdl_event.which
                    self.button = sdl_event.button
                elif self.type == SDL_VIDEORESIZE:
                    self.size = (sdl_event.w, sdl_event.h)
                    self.w = sdl_event.w
                    self.h = sdl_event.h
                elif self.type == SDL_VIDEOEXPOSE:
                    pass
                elif self.type == SDL_SYSWMEVENT:
                    pass ### XXX: not implemented
                elif self.type >= SDL_USEREVENT and self.type < SDL_NUMEVENTS:
                    self.code = sdl_event.code
        else:
            # Create an event (not from event queue)
            self.type = event_type
            for key, value in event_dict.items():
                setattr(self, key, value)
            for key, value in attributes:
                setattr(self, key, value)

    def __repr__(self):
        d = copy(self.__dict__)
        del d['type']
        return '<Event(%d-%s %r)>' % \
            (self.type, event_name(self.type), d)

    def __nonzero__(self):
        return self.type != SDL_NOEVENT

EventType = Event
