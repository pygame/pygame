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
        """Ensure an Event object can be created."""
        e = pygame.event.Event(pygame.USEREVENT, some_attr=1, other_attr='1')

        self.assertEqual(e.some_attr, 1)
        self.assertEqual(e.other_attr, "1")

        # Event now uses tp_dictoffset and tp_members: request 62
        # on Motherhamster Bugzilla.
        self.assertEqual(e.type, pygame.USEREVENT)
        self.assertIs(e.dict, e.__dict__)

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
        attrs = ('type', 'dict', '__dict__', 'some_attr', 'other_attr',
                 'new_attr')

        for attr in attrs:
            self.assertIn(attr, d)

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
        pygame.display.init()
        pygame.event.clear()  # flush events

    def tearDown(self):
        pygame.event.clear()  # flush events
        pygame.display.quit()

    def test_event_attribute(self):
        e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
        self.assertEqual(e1.attr1, 'attr1')

    def test_set_blocked(self):
        """Ensure events can be blocked from the queue."""
        event = events[0]
        pygame.event.set_blocked(event)

        self.assertTrue(pygame.event.get_blocked(event))

        pygame.event.post(pygame.event.Event(event))
        ret = pygame.event.get()
        should_be_blocked = [e for e in ret if e.type == event]

        self.assertEqual(should_be_blocked, [])

    def test_set_blocked_all(self):
        """Ensure all events can be unblocked at once."""
        pygame.event.set_blocked(None)

        for e in events:
            self.assertTrue(pygame.event.get_blocked(e))

    def test_post__and_poll(self):
        """Ensure events can be posted to the queue."""
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
        """Ensure get() retrieves all the events on the queue."""
        event_cnt = 10
        for _ in range(event_cnt):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT))

        queue = pygame.event.get()

        self.assertEqual(len(queue), event_cnt)
        self.assertTrue(all(e.type == pygame.USEREVENT for e in queue))

    def test_clear(self):
        """Ensure clear() removes all the events on the queue."""
        for e in events:
            pygame.event.post(pygame.event.Event(e))

        poll_event = pygame.event.poll()

        self.assertNotEqual(poll_event.type, pygame.NOEVENT)

        pygame.event.clear()
        poll_event = pygame.event.poll()

        self.assertEqual(poll_event.type, pygame.NOEVENT,
                         race_condition_notification)

    def test_event_name(self):
        """Ensure event_name() returns the correct event name."""
        self.assertEqual(pygame.event.event_name(pygame.KEYDOWN), "KeyDown")
        self.assertEqual(pygame.event.event_name(pygame.USEREVENT),
                         "UserEvent")

    def test_wait(self):
        """Ensure wait() waits for an event on the queue."""
        event = pygame.event.Event(events[0])
        pygame.event.post(event)
        wait_event = pygame.event.wait()

        self.assertEqual(wait_event.type, event.type)

    def test_peek(self):
        """Ensure queued events can be peeked at."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]

        for event_type in event_types:
            pygame.event.post(pygame.event.Event(event_type))

        for event_type in event_types:
            self.assertTrue(pygame.event.peek(event_type))

        self.assertTrue(pygame.event.peek(event_types))

    def test_peek_empty(self):
        pygame.event.clear()
        self.assertFalse(pygame.event.peek())

    def test_set_allowed(self):
        """Ensure a blocked event type can be unblocked/allowed."""
        event = events[0]
        pygame.event.set_blocked(event)

        self.assertTrue(pygame.event.get_blocked(event))

        pygame.event.set_allowed(event)

        self.assertFalse(pygame.event.get_blocked(event))

    def test_set_allowed_all(self):
        """Ensure all events can be unblocked/allowed at once."""
        pygame.event.set_blocked(None)

        for e in events:
            self.assertTrue(pygame.event.get_blocked(e))

        pygame.event.set_allowed(None)

        for e in events:
            self.assertFalse(pygame.event.get_blocked(e))

    def test_pump(self):
        """Ensure pump() functions properly."""
        pygame.event.pump()

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy',
                     'requires the SDL_VIDEODRIVER to be a non "dummy" value')
    def test_set_grab__and_get_symmetric(self):
        """Ensure event grabbing can be enabled and disabled."""
        surf = pygame.display.set_mode((10,10))
        pygame.event.set_grab(True)

        self.assertTrue(pygame.event.get_grab())

        pygame.event.set_grab(False)

        self.assertFalse(pygame.event.get_grab())

    def test_event_equality(self):
        a = pygame.event.Event(events[0], a=1)
        b = pygame.event.Event(events[0], a=1)
        c = pygame.event.Event(events[1], a=1)
        d = pygame.event.Event(events[0], a=2)

        self.assertTrue(a == a)
        self.assertFalse(a != a)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a !=  c)
        self.assertFalse(a == c)
        self.assertTrue(a != d)
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
