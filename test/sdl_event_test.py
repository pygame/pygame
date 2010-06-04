import unittest
import pygame2
import pygame2.sdl.event as event
import pygame2.sdl.constants as constants

class SDLEventTest (unittest.TestCase):

    def test_pygame2_sdl_event_Event (self):
        # check argument handling
        self.assertRaises (TypeError, event.Event)
        self.assertRaises (TypeError, event.Event, None)
        self.assertRaises (ValueError, event.Event, -5)
        self.assertRaises (ValueError, event.Event, 256)
        
    def test_pygame2_sdl_event_Event_name(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.Event.name:

        # Gets the name of the Event.

        ev = event.Event (200)
        self.assertEqual (ev.name, "Unknown")
        
        ev = event.Event (0)
        self.assertEqual (ev.name, "NoEvent")
        ev = event.Event (constants.NOEVENT)
        self.assertEqual (ev.name, "NoEvent")
        
        ev = event.Event (1)
        self.assertEqual (ev.name, "ActiveEvent")
        ev = event.Event (constants.ACTIVEEVENT)
        self.assertEqual (ev.name, "ActiveEvent")

        ev = event.Event (2)
        self.assertEqual (ev.name, "KeyDown")
        ev = event.Event (constants.KEYDOWN)
        self.assertEqual (ev.name, "KeyDown")
        
        ev = event.Event (3)
        self.assertEqual (ev.name, "KeyUp")
        ev = event.Event (constants.KEYUP)
        self.assertEqual (ev.name, "KeyUp")
        
        ev = event.Event (4)
        self.assertEqual (ev.name, "MouseMotion")
        ev = event.Event (constants.MOUSEMOTION)
        self.assertEqual (ev.name, "MouseMotion")
        
        ev = event.Event (5)
        self.assertEqual (ev.name, "MouseButtonDown")
        ev = event.Event (constants.MOUSEBUTTONDOWN)
        self.assertEqual (ev.name, "MouseButtonDown")
        
        ev = event.Event (6)
        self.assertEqual (ev.name, "MouseButtonUp")
        ev = event.Event (constants.MOUSEBUTTONUP)
        self.assertEqual (ev.name, "MouseButtonUp")

        ev = event.Event (7)
        self.assertEqual (ev.name, "JoyAxisMotion")
        ev = event.Event (constants.JOYAXISMOTION)
        self.assertEqual (ev.name, "JoyAxisMotion")

        ev = event.Event (8)
        self.assertEqual (ev.name, "JoyBallMotion")
        ev = event.Event (constants.JOYBALLMOTION)
        self.assertEqual (ev.name, "JoyBallMotion")

        ev = event.Event (9)
        self.assertEqual (ev.name, "JoyHatMotion")
        ev = event.Event (constants.JOYHATMOTION)
        self.assertEqual (ev.name, "JoyHatMotion")

        ev = event.Event (10)
        self.assertEqual (ev.name, "JoyButtonDown")
        ev = event.Event (constants.JOYBUTTONDOWN)
        self.assertEqual (ev.name, "JoyButtonDown")

        ev = event.Event (11)
        self.assertEqual (ev.name, "JoyButtonUp")
        ev = event.Event (constants.JOYBUTTONUP)
        self.assertEqual (ev.name, "JoyButtonUp")

        ev = event.Event (12)
        self.assertEqual (ev.name, "Quit")
        ev = event.Event (constants.QUIT)
        self.assertEqual (ev.name, "Quit")

        ev = event.Event (13)
        self.assertEqual (ev.name, "SysWMEvent")
        ev = event.Event (constants.SYSWMEVENT)
        self.assertEqual (ev.name, "SysWMEvent")

        ev = event.Event (16)
        self.assertEqual (ev.name, "VideoResize")
        ev = event.Event (constants.VIDEORESIZE)
        self.assertEqual (ev.name, "VideoResize")

        ev = event.Event (17)
        self.assertEqual (ev.name, "VideoExpose")
        ev = event.Event (constants.VIDEOEXPOSE)
        self.assertEqual (ev.name, "VideoExpose")

        ev = event.Event (24)
        self.assertEqual (ev.name, "UserEvent")
        ev = event.Event (constants.USEREVENT)
        self.assertEqual (ev.name, "UserEvent")

    def test_pygame2_sdl_event_Event_type(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.Event.type:

        # Gets the type id of the Event.
        ev = event.Event (constants.USEREVENT)
        self.assertEqual (ev.type, constants.USEREVENT)
        ev = event.Event (constants.KEYUP)
        self.assertEqual (ev.type, constants.KEYUP)

        for i in range (0, 255):
            ev = event.Event (i)
            self.assertEqual (ev.type, i)
            
    def todo_test_pygame2_sdl_event_clear(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.clear:

        # clear ([events]) -> None
        # 
        # Clears the event queue from certain event types.
        # 
        # Clears the event queue from certain event types. If no argument is
        # passed, all current events are removed from the queue. Otherwise
        # the argument can be a sequence or a bitmask combination of event
        # types to clear.

        self.fail() 

    def todo_test_pygame2_sdl_event_get(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.get:

        # get ([events]) -> [Event, Event, ... ]
        # 
        # Gets events from the event queue.
        # 
        # Gets the current events from the event queue. If no argument is
        # passed, all currently available events are received from the event
        # queue and returned as list. Otherwise, the argument can be a
        # sequence or a bitmask combination of event types to receive from
        # the queue.
        # 
        # If no matching events are found on the queue, an empty list will be
        # returned.

        self.fail() 

    def todo_test_pygame2_sdl_event_get_app_state(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.get_app_state:

        # get_app_state () -> int
        # 
        # Gets the current application state.
        # 
        # Gets the current application state. This will be a bitmask
        # combination of the APPMOUSEFOCUS, APPINPUTFOCUS or
        # APPACTIVE masks, indicating whether the application currently is
        # active and has the mouse and keyboard input focus.

        self.fail() 

    def todo_test_pygame2_sdl_event_get_blocked(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.get_blocked:

        # get_blocked () -> [int, int, ...]
        # 
        # Gets a list of currently blocked event types.
        # 
        # Gets a list of currently blocked event types. Events having the
        # matching type will not be processed by the event queue.

        self.fail() 

    def todo_test_pygame2_sdl_event_get_filter(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.get_filter:

        # get_filter () -> object
        # 
        # Gets the currently set filter hook method.
        # 
        # Gets the filter hook method set previously by :func:set_filter.

        self.fail() 

    def todo_test_pygame2_sdl_event_peek(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.peek:

        # peek ([events]) -> bool
        # 
        # Checks, whether certain event types are currently on the queue.
        # 
        # Checks, whether certain event types are currently on the queue. If
        # no argument is passed, this method simply checks, if there is any
        # event on the event queue. Otherwise, the argument can be a
        # sequence or a bitmask combination of event types to check for. In
        # case one event is found, which corresponds to the requested
        # type(s), True is returned.

        self.fail() 

    def todo_test_pygame2_sdl_event_peep(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.peep:

        # peep (num, action, mask[, events]) -> int or [Event, ...]
        # 
        # Checks the event queue for events and optionally returns them.
        # 
        # This is an advanced event queue querying and manipulation
        # method. It allows to inspect the event queue, to receive events
        # from it or to add events.
        # 
        # TODO

        self.fail() 

    def todo_test_pygame2_sdl_event_poll(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.poll:

        # poll () -> Event
        # 
        # Gets a single event from the event queue.
        # 
        # Returns a single event from the queue. If the event queue is
        # empty, None will be returned. If an event is available and
        # returned, it will be removed from the queue.

        self.fail() 

    def todo_test_pygame2_sdl_event_pump(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.pump:

        # pump () -> None
        # 
        # Pumps the event queue, forcing it to gather pending events from
        # devices.
        # 
        # This gathers all pending events and input information from
        # devices and places them on the event queue.
        # 
        # It only has to be called, if you use :func:peep or a filter hook
        # without using another event function or no other event function at
        # all.

        self.fail() 

    def todo_test_pygame2_sdl_event_push(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.push:

        # push (event) -> None
        # 
        # Places a new event at the end of the event queue.
        # 
        # This is usually used for placing user defined events on the event
        # queue. You also can push user created device events on the queue,
        # but this will not change the state of the device itself.

        self.fail() 

    def todo_test_pygame2_sdl_event_set_blocked(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.set_blocked:

        # set_blocked (type) -> None
        # 
        # Blocks a single or multiple event types.
        # 
        # This will block a single or multiple event types from being
        # processed by the event system and thus will exactly behave like
        # :func:state(type, IGNORE).
        # 
        # In case other event types are already blocked, the block for them
        # will be reset.
        # 
        # To remove the block for all events, call set_blocked(None).

        self.fail() 

    def todo_test_pygame2_sdl_event_set_filter(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.set_filter:

        # set_filter (filterhook) -> None
        # 
        # Sets up a filter hook for events.
        # 
        # This sets up a filter to process all events before they are posted
        # to the event queue. In order to process events correctly, the
        # *filterhook* must return True or False, indicating whether the event
        # processd by it, is allowed to be placed on the event queue or not.
        # It has to take a single argument, which will be the event to process. 
        # 
        # def example_filter_hook (event):
        # if event.type == ... and ...:
        # # The event matches a certain scheme, do not allow it.
        # return False
        # # Any other event may pass.
        # return True
        # 
        # In case the QUITEVENT is processed by the event filter, returning
        # True will cause the SDL window to be closed, otherwise, the window
        # will remain open, if possible.
        # 
        # .. note
        # 
        # Events pushed onto the event queue using :func:push or :func:peep do
        # not get passed through the filter.

        self.fail() 

    def todo_test_pygame2_sdl_event_state(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.state:

        # state (type, state) -> int
        # 
        # Sets the processing state of an event type.
        # 
        # This allows you to set the processing state of an event *type*.
        # If the *state* is set to IGNORE, events matching *type* will be
        # automatically dropped from the event queue and not be filtered. This will
        # behave similar to :func:set_blocked. If *state* is set to
        # ENABLE, events matching *type* will be processed normally.
        # If *state* is set to QUERY, this will return the current
        # processing state of the specified event type (ENABLE or
        # IGNORE).

        self.fail() 

    def todo_test_pygame2_sdl_event_wait(self):

        # __doc__ (as of 2009-12-12) for pygame2.sdl.event.wait:

        # wait () -> Event
        # 
        # Waits indefinitely for the next available event.
        # 
        # This is a blocking method, that only returns, if an event occurs
        # on the event queue. Once an event occurs, it will be returned to
        # the caller and removed from the queue. While the program is
        # waiting it will sleep in an idle state.

        self.fail()

if __name__ == "__main__":
    unittest.main ()
