#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class EventTypeTest(unittest.TestCase):
    def test_Event(self):
        
        # __doc__ (as of 2008-06-25) for pygame.event.Event:
        
          # pygame.event.Event(type, dict): return Event
          # pygame.event.Event(type, **attributes): return Event
          # create a new event object
        
        self.assert_(test_not_implemented()) 

class EventModuleTest(unittest.TestCase):
    def test_clear(self):

        # __doc__ (as of 2008-06-25) for pygame.event.clear:

          # pygame.event.clear(): return None
          # pygame.event.clear(type): return None
          # pygame.event.clear(typelist): return None
          # remove all events from the queue

        self.assert_(test_not_implemented()) 

    def test_event_name(self):

        # __doc__ (as of 2008-06-25) for pygame.event.event_name:

          # pygame.event.event_name(type): return string
          # get the string name from and event id

        self.assert_(test_not_implemented()) 

    def test_get(self):

        # __doc__ (as of 2008-06-25) for pygame.event.get:

          # pygame.event.get(): return Eventlist
          # pygame.event.get(type): return Eventlist
          # pygame.event.get(typelist): return Eventlist
          # get events from the queue

        self.assert_(test_not_implemented()) 

    def test_get_blocked(self):

        # __doc__ (as of 2008-06-25) for pygame.event.get_blocked:

          # pygame.event.get_blocked(type): return bool
          # test if a type of event is blocked from the queue

        self.assert_(test_not_implemented()) 

    def test_get_grab(self):

        # __doc__ (as of 2008-06-25) for pygame.event.get_grab:

          # pygame.event.get_grab(): return bool
          # test if the program is sharing input devices

        self.assert_(test_not_implemented()) 

    def test_peek(self):

        # __doc__ (as of 2008-06-25) for pygame.event.peek:

          # pygame.event.peek(type): return bool
          # pygame.event.peek(typelist): return bool
          # test if event types are waiting on the queue

        self.assert_(test_not_implemented()) 

    def test_poll(self):

        # __doc__ (as of 2008-06-25) for pygame.event.poll:

          # pygame.event.poll(): return Event
          # get a single event from the queue

        self.assert_(test_not_implemented()) 

    def test_post(self):

        # __doc__ (as of 2008-06-25) for pygame.event.post:

          # pygame.event.post(Event): return None
          # place a new event on the queue

        self.assert_(test_not_implemented()) 

    def test_pump(self):

        # __doc__ (as of 2008-06-25) for pygame.event.pump:

          # pygame.event.pump(): return None
          # internally process pygame event handlers

        self.assert_(test_not_implemented()) 

    def test_set_allowed(self):

        # __doc__ (as of 2008-06-25) for pygame.event.set_allowed:

          # pygame.event.set_allowed(type): return None
          # pygame.event.set_allowed(typelist): return None
          # pygame.event.set_allowed(None): return None
          # control which events are allowed on the queue

        self.assert_(test_not_implemented()) 

    def test_set_blocked(self):

        # __doc__ (as of 2008-06-25) for pygame.event.set_blocked:

          # pygame.event.set_blocked(type): return None
          # pygame.event.set_blocked(typelist): return None
          # pygame.event.set_blocked(None): return None
          # control which events are allowed on the queue

        self.assert_(test_not_implemented()) 

    def test_set_grab(self):

        # __doc__ (as of 2008-06-25) for pygame.event.set_grab:

          # pygame.event.set_grab(bool): return None
          # control the sharing of input devices with other applications

        self.assert_(test_not_implemented()) 

    def test_wait(self):

        # __doc__ (as of 2008-06-25) for pygame.event.wait:

          # pygame.event.wait(): return Event
          # wait for a single event from the queue

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
