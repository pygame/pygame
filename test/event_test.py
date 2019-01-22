import os
import unittest

import pygame
from pygame.compat import as_unicode

################################################################################

events = (
#   pygame.NOEVENT,
#   pygame.ACTIVEEVENT,
    pygame.KEYDOWN,
    pygame.KEYUP,
    pygame.MOUSEMOTION,
    pygame.MOUSEBUTTONDOWN,
    pygame.MOUSEBUTTONUP,
    pygame.JOYAXISMOTION,
    pygame.JOYBALLMOTION,
    pygame.JOYHATMOTION,
    pygame.JOYBUTTONDOWN,
    pygame.JOYBUTTONUP,
    pygame.VIDEORESIZE,
    pygame.VIDEOEXPOSE,
    pygame.QUIT,
    pygame.SYSWMEVENT,
    pygame.USEREVENT,
#   pygame.NUMEVENTS,
)

class EventTypeTest(unittest.TestCase):
    def test_Event(self):
        # __doc__ (as of 2008-08-02) for pygame.event.Event:

          # pygame.event.Event(type, dict): return Event
          # pygame.event.Event(type, **attributes): return Event
          # create a new event object
          #
          # Creates a new event with the given type. The event is created with
          # the given attributes and values. The attributes can come from a
          # dictionary argument, or as string keys from a dictionary.
          #
          # The given attributes will be readonly attributes on the new event
          # object itself. These are the only attributes on the Event object,
          # there are no methods attached to Event objects.

        e = pygame.event.Event(pygame.USEREVENT, some_attr=1, other_attr='1')

        self.assertEqual(e.some_attr, 1)
        self.assertEqual(e.other_attr, "1")

        # Event now uses tp_dictoffset and tp_members: request 62
        # on Motherhamster Bugzilla.
        self.assertEqual(e.type, pygame.USEREVENT)
        self.assert_(e.dict is e.__dict__)
        e.some_attr = 12
        self.assertEqual(e.some_attr, 12)
        e.new_attr = 15
        self.assertEqual(e.new_attr, 15)

        # For Python 2.x a TypeError is raised for a readonly member;
        # for Python 3.x it is an AttributeError.
        self.assertRaises((TypeError, AttributeError), setattr, e, 'type', 0)
        self.assertRaises((TypeError, AttributeError), setattr, e, 'dict', None)

        # Ensure attributes are visible to dir(), part of the original
        # posted request.
        d = dir(e)
        self.assert_('type' in d)
        self.assert_('dict' in d)
        self.assert_('__dict__' in d)
        self.assert_('some_attr' in d)
        self.assert_('other_attr' in d)
        self.assert_('new_attr' in d)

    def test_as_str(self):
        # Bug reported on Pygame mailing list July 24, 2011:
        # For Python 3.x str(event) to raises an UnicodeEncodeError when
        # an event attribute is a string with a non-ascii character.
        try:
            str(pygame.event.Event(events[0], a=as_unicode(r"\xed")))
        except UnicodeEncodeError:
            self.fail("Event object raised exception for non-ascii character")
        # Passed.


race_condition_notification = """
This test is dependent on timing. The event queue is cleared in preparation for
tests. There is a small window where outside events from the OS may have effected
results. Try running the test again.
"""

class EventModuleTest(unittest.TestCase):
    def setUp(self):
        # flush events
        pygame.display.init()
        pygame.event.clear()
        self.assert_(not pygame.event.get())

    def tearDown(self):
        pygame.display.quit()

    def test_event_attribute(self):
        e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
        self.assertEqual(e1.attr1, 'attr1')

    def test_set_blocked(self):
        # __doc__ (as of 2008-06-25) for pygame.event.set_blocked:

          # pygame.event.set_blocked(type): return None
          # pygame.event.set_blocked(typelist): return None
          # pygame.event.set_blocked(None): return None
          # control which events are allowed on the queue

        event = events[0]

        pygame.event.set_blocked(event)

        self.assert_(pygame.event.get_blocked(event))

        pygame.event.post(pygame.event.Event(event))

        ret = pygame.event.get()
        should_be_blocked = [e for e in ret if e.type == event]

        self.assertEqual(should_be_blocked, [])

    def test_set_blocked_all(self):
        pygame.event.set_blocked(None)
        for e in events:
            self.assert_(pygame.event.get_blocked(e))

    def test_post__and_poll(self):
        # __doc__ (as of 2008-06-25) for pygame.event.post:

          # pygame.event.post(Event): return None
          # place a new event on the queue

        e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
        pygame.event.post(e1)
        posted_event = pygame.event.poll()

        self.assertEqual(e1.attr1, posted_event.attr1,
                         race_condition_notification)

        # fuzzing event types
        for i in range(1, 11):
            pygame.event.post(pygame.event.Event(events[i]))

            self.assertEqual(pygame.event.poll().type, events[i],
                             race_condition_notification)

    def test_post_large_user_event(self):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'a': "a" * 1024}))
        e = pygame.event.poll()

        self.assertEqual(e.type, pygame.USEREVENT)
        self.assertEqual(e.a, "a" * 1024)

    def test_get(self):
        # __doc__ (as of 2008-06-25) for pygame.event.get:

          # pygame.event.get(): return Eventlist
          # pygame.event.get(type): return Eventlist
          # pygame.event.get(typelist): return Eventlist
          # get events from the queue

        # Put 10 events on the queue
        for _ in range(1, 11):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT))

        queue = pygame.event.get()
        self.assert_ ( len(queue) >= 10 )
        self.assert_ ( all(e.type == pygame.USEREVENT for e in queue) )

    def test_clear(self):

        # __doc__ (as of 2008-06-25) for pygame.event.clear:

          # pygame.event.clear(): return None
          # pygame.event.clear(type): return None
          # pygame.event.clear(typelist): return None
          # remove all events from the queue

        for e in events:
            pygame.event.post(pygame.event.Event(e))

        self.assert_(pygame.event.poll())  # there are some events on queue

        pygame.event.clear()

        self.assert_(not pygame.event.poll(), race_condition_notification)

    def test_event_name(self):

        # __doc__ (as of 2008-06-25) for pygame.event.event_name:

          # pygame.event.event_name(type): return string
          # get the string name from and event id

        self.assertEqual(pygame.event.event_name(pygame.KEYDOWN), "KeyDown")
        self.assertEqual(pygame.event.event_name(pygame.USEREVENT),
                         "UserEvent")

    def test_wait(self):
        # __doc__ (as of 2008-06-25) for pygame.event.wait:

          # pygame.event.wait(): return Event
          # wait for a single event from the queue

        pygame.event.post ( pygame.event.Event(events[0]) )
        self.assert_(pygame.event.wait())

    def test_peek(self):

        # __doc__ (as of 2008-06-25) for pygame.event.peek:

          # pygame.event.peek(type): return bool
          # pygame.event.peek(typelist): return bool
          # test if event types are waiting on the queue

        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]

        for event_type in event_types:
            pygame.event.post (
                pygame.event.Event(event_type)
            )
            self.assert_(pygame.event.peek(event_type))

        self.assert_(pygame.event.peek(event_types))

    def test_set_allowed(self):
        # __doc__ (as of 2008-06-25) for pygame.event.set_allowed:

          # pygame.event.set_allowed(type): return None
          # pygame.event.set_allowed(typelist): return None
          # pygame.event.set_allowed(None): return None
          # control which events are allowed on the queue

        event = events[0]
        pygame.event.set_blocked(event)
        self.assert_(pygame.event.get_blocked(event))
        pygame.event.set_allowed(event)
        self.assert_(not pygame.event.get_blocked(event))

    def test_set_allowed_all(self):
        pygame.event.set_blocked(None)
        for e in events:
            self.assert_(pygame.event.get_blocked(e))
        pygame.event.set_allowed(None)
        for e in events:
            self.assert_(not pygame.event.get_blocked(e))

    def test_pump(self):
        # __doc__ (as of 2008-06-25) for pygame.event.pump:

          # pygame.event.pump(): return None
          # internally process pygame event handlers

        # see it doesn't cause an error
        pygame.event.pump()

    def test_set_grab__and_get_symmetric(self):

        # __doc__ (as of 2008-06-25) for pygame.event.set_grab:

          # pygame.event.set_grab(bool): return None
          # control the sharing of input devices with other applications

        # If we don't have a real display, don't do the test.
        if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
            return

        surf = pygame.display.set_mode((10,10))
        pygame.event.set_grab(True)
        self.assert_(pygame.event.get_grab())
        pygame.event.set_grab(False)
        self.assert_(not pygame.event.get_grab())

    def test_event_equality(self):
        a = pygame.event.Event(events[0], a=1)
        b = pygame.event.Event(events[0], a=1)
        c = pygame.event.Event(events[1], a=1)
        d = pygame.event.Event(events[0], a=2)

        self.failUnless(a == a)
        self.assertFalse(a != a)
        self.failUnless(a == b)
        self.assertFalse(a != b)
        self.failUnless(a !=  c)
        self.assertFalse(a == c)
        self.failUnless(a != d)
        self.assertFalse(a == d)

    def todo_test_get_blocked(self):

        # __doc__ (as of 2008-08-02) for pygame.event.get_blocked:

          # pygame.event.get_blocked(type): return bool
          # test if a type of event is blocked from the queue
          #
          # Returns true if the given event type is blocked from the queue.

        self.fail()

    def todo_test_get_grab(self):

        # __doc__ (as of 2008-08-02) for pygame.event.get_grab:

          # pygame.event.get_grab(): return bool
          # test if the program is sharing input devices
          #
          # Returns true when the input events are grabbed for this application.
          # Use pygame.event.set_grab() to control this state.
          #

        self.fail()

    def todo_test_poll(self):

        # __doc__ (as of 2008-08-02) for pygame.event.poll:

          # pygame.event.poll(): return Event
          # get a single event from the queue
          #
          # Returns a single event from the queue. If the event queue is empty
          # an event of type pygame.NOEVENT will be returned immediately. The
          # returned event is removed from the queue.
          #

        self.fail()

################################################################################

if __name__ == '__main__':
    unittest.main()
