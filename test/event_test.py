import os
import sys
import unittest
import collections

import pygame
from pygame.compat import as_unicode


PY3 = sys.version_info >= (3, 0, 0)
SDL1 = pygame.get_sdl_version()[0] < 2

################################################################################

EVENT_TYPES = (
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

EVENT_TEST_PARAMS = collections.defaultdict(dict)
EVENT_TEST_PARAMS.update({
    pygame.KEYDOWN:{'key': pygame.K_SPACE},
    pygame.KEYUP:{'key': pygame.K_SPACE},
    pygame.MOUSEMOTION:dict(),
    pygame.MOUSEBUTTONDOWN:dict(button=1),
    pygame.MOUSEBUTTONUP:dict(button=1),
})


NAMES_AND_EVENTS = (
    ("NoEvent", pygame.NOEVENT),
    ("ActiveEvent", pygame.ACTIVEEVENT),
    ("KeyDown", pygame.KEYDOWN),
    ("KeyUp", pygame.KEYUP),
    ("MouseMotion", pygame.MOUSEMOTION),
    ("MouseButtonDown", pygame.MOUSEBUTTONDOWN),
    ("MouseButtonUp", pygame.MOUSEBUTTONUP),
    ("JoyAxisMotion", pygame.JOYAXISMOTION),
    ("JoyBallMotion", pygame.JOYBALLMOTION),
    ("JoyHatMotion", pygame.JOYHATMOTION),
    ("JoyButtonDown", pygame.JOYBUTTONDOWN),
    ("JoyButtonUp", pygame.JOYBUTTONUP),
    ("VideoResize", pygame.VIDEORESIZE),
    ("VideoExpose", pygame.VIDEOEXPOSE),
    ("Quit", pygame.QUIT),
    ("SysWMEvent", pygame.SYSWMEVENT),
    ("MidiIn", pygame.MIDIIN),
    ("MidiOut", pygame.MIDIOUT),
    ("UserEvent", pygame.USEREVENT),
    ("Unknown", 0xFFFF),
)

# Add in any SDL 2 specific events.
if pygame.get_sdl_version()[0] >= 2:
    NAMES_AND_EVENTS += (
        ("FingerMotion", pygame.FINGERMOTION),
        ("FingerDown", pygame.FINGERDOWN),
        ("FingerUp", pygame.FINGERUP),
        ("MultiGesture", pygame.MULTIGESTURE),
        ("MouseWheel", pygame.MOUSEWHEEL),
        ("TextInput", pygame.TEXTINPUT),
        ("TextEditing", pygame.TEXTEDITING),
        ("WindowEvent", pygame.WINDOWEVENT),
        ("ControllerAxisMotion", pygame.CONTROLLERAXISMOTION),
        ("ControllerButtonDown", pygame.CONTROLLERBUTTONDOWN),
        ("ControllerButtonUp", pygame.CONTROLLERBUTTONUP),
        ("ControllerDeviceAdded", pygame.CONTROLLERDEVICEADDED),
        ("ControllerDeviceRemoved", pygame.CONTROLLERDEVICEREMOVED),
        ("ControllerDeviceMapped", pygame.CONTROLLERDEVICEREMAPPED),
        ("DropFile", pygame.DROPFILE),
    )

    # Add in any SDL 2.0.4 specific events.
    if pygame.get_sdl_version() >= (2, 0, 4):
        NAMES_AND_EVENTS += (
            ("AudioDeviceAdded", pygame.AUDIODEVICEADDED),
            ("AudioDeviceRemoved", pygame.AUDIODEVICEREMOVED),
        )

    # Add in any SDL 2.0.5 specific events.
    if pygame.get_sdl_version() >= (2, 0, 5):
        NAMES_AND_EVENTS += (
            ("DropText", pygame.DROPTEXT),
            ("DropBegin", pygame.DROPBEGIN),
            ("DropComplete", pygame.DROPCOMPLETE),
        )


class EventTypeTest(unittest.TestCase):
    def test_Event(self):
        """Ensure an Event object can be created."""
        e = pygame.event.Event(pygame.USEREVENT, some_attr=1, other_attr="1")

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
        self.assertRaises((TypeError, AttributeError), setattr, e, "type", 0)
        self.assertRaises((TypeError, AttributeError), setattr, e, "dict", None)

        # Ensure attributes are visible to dir(), part of the original
        # posted request.
        d = dir(e)
        attrs = ("type", "dict", "__dict__", "some_attr", "other_attr", "new_attr")

        for attr in attrs:
            self.assertIn(attr, d)

    def test_as_str(self):
        # Bug reported on Pygame mailing list July 24, 2011:
        # For Python 3.x str(event) to raises an UnicodeEncodeError when
        # an event attribute is a string with a non-ascii character.
        try:
            str(pygame.event.Event(EVENT_TYPES[0], a=as_unicode(r"\xed")))
        except UnicodeEncodeError:
            self.fail("Event object raised exception for non-ascii character")
        # Passed.


race_condition_notification = """
This test is dependent on timing. The event queue is cleared in preparation for
tests. There is a small window where outside events from the OS may have effected
results. Try running the test again.
"""


class EventModuleArgsTest(unittest.TestCase):
    def setUp(self):
        pygame.display.init()
        pygame.event.clear()

    def tearDown(self):
        pygame.display.quit()

    def test_get(self):
        pygame.event.get()
        pygame.event.get(None)
        pygame.event.get(None, True)

        pygame.event.get(pump=False)
        pygame.event.get(pump=True)
        pygame.event.get(eventtype=None)
        pygame.event.get(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.get(eventtype=pygame.USEREVENT, pump=False)

    def test_clear(self):
        pygame.event.clear()
        pygame.event.clear(None)
        pygame.event.clear(None, True)

        pygame.event.clear(pump=False)
        pygame.event.clear(pump=True)
        pygame.event.clear(eventtype=None)
        pygame.event.clear(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.clear(eventtype=pygame.USEREVENT, pump=False)

    def test_peek(self):
        pygame.event.peek()
        pygame.event.peek(None)
        pygame.event.peek(None, True)

        pygame.event.peek(pump=False)
        pygame.event.peek(pump=True)
        pygame.event.peek(eventtype=None)
        pygame.event.peek(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.peek(eventtype=pygame.USEREVENT, pump=False)


class EventCustomTypeTest(unittest.TestCase):
    """Those tests are special in that they need the _custom_event counter to
    be reset before and/or after being run."""
    def setUp(self):
        pygame.quit()
        pygame.init()
        pygame.display.init()

    def tearDown(self):
        pygame.quit()

    def test_custom_type(self):
        self.assertEqual(pygame.event.custom_type(), pygame.USEREVENT + 1)
        atype = pygame.event.custom_type()
        atype2 = pygame.event.custom_type()

        self.assertEqual(atype, atype2 - 1)

        ev = pygame.event.Event(atype)
        pygame.event.post(ev)
        queue = pygame.event.get(atype)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, atype)

    def test_custom_type__end_boundary(self):
        """Ensure custom_type() raises error when no more custom types.

        The last allowed custom type number should be (pygame.NUMEVENTS - 1).
        """
        start = pygame.event.custom_type() + 1
        for i in range(start, pygame.NUMEVENTS):
            last = pygame.event.custom_type()
        self.assertEqual(last, pygame.NUMEVENTS - 1)
        with self.assertRaises(pygame.error):
            pygame.event.custom_type()

    def test_custom_type__reset(self):
        """Ensure custom events get 'deregistered' by quit().
        """
        before = pygame.event.custom_type()
        self.assertEqual(before, pygame.event.custom_type() - 1)
        pygame.quit()
        pygame.init()
        pygame.display.init()
        self.assertEqual(before, pygame.event.custom_type())


class EventModuleTest(unittest.TestCase):
    def _assertCountEqual(self, *args, **kwargs):
        # Handle method name differences between Python versions.
        if PY3:
            self.assertCountEqual(*args, **kwargs)
        else:
            self.assertItemsEqual(*args, **kwargs)

    def _assertExpectedEvents(self, expected, got):
        """Find events like expected events, raise on unexpected or missing,
        ignore additional event properties if expected properties are present."""

        # This does greedy matching, don't encode an NP-hard problem
        # into your input data, *please*
        items_left=got[:]
        for expected_element in expected:
            for item in items_left:
                for key in expected_element.__dict__:
                    if item.__dict__[key]!=expected_element.__dict__[key]:
                        break
                else:
                    #found item!
                    items_left.remove(item)
                    break
            else:
                raise AssertionError("Expected "+str(expected_element)+" among remaining events "+str(items_left)+" out of "+str(got))
        if len(items_left)>0:
            raise AssertionError("Unexpected Events: "+str(items_left))

    def setUp(self):
        pygame.display.init()
        pygame.event.clear()  # flush events

    def tearDown(self):
        pygame.event.clear()  # flush events
        pygame.display.quit()

    def test_event_numevents(self):
        """Ensures NUMEVENTS does not exceed the maximum SDL number of events.
        """
        # Ref: https://www.libsdl.org/tmp/SDL/include/SDL_events.h
        MAX_SDL_EVENTS = 0xFFFF + 1  # SDL_LASTEVENT = 0xFFFF

        self.assertLessEqual(pygame.NUMEVENTS, MAX_SDL_EVENTS)

    def test_event_attribute(self):
        e1 = pygame.event.Event(pygame.USEREVENT, attr1="attr1")
        self.assertEqual(e1.attr1, "attr1")

    def test_set_blocked(self):
        """Ensure events can be blocked from the queue."""
        event = EVENT_TYPES[0]
        pygame.event.set_blocked(event)

        self.assertTrue(pygame.event.get_blocked(event))

        pygame.event.post(pygame.event.Event(event, **EVENT_TEST_PARAMS[EVENT_TYPES[0]]))
        ret = pygame.event.get()
        should_be_blocked = [e for e in ret if e.type == event]

        self.assertEqual(should_be_blocked, [])

    def test_set_blocked__event_sequence(self):
        """Ensure a sequence of event types can be blocked."""
        event_types = [
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
        ]

        pygame.event.set_blocked(event_types)

        for etype in event_types:
            self.assertTrue(pygame.event.get_blocked(etype))

    def test_set_blocked_all(self):
        """Ensure all events can be unblocked at once."""
        pygame.event.set_blocked(None)

        for e in EVENT_TYPES:
            self.assertTrue(pygame.event.get_blocked(e))

    def test_post__and_poll(self):
        """Ensure events can be posted to the queue."""
        e1 = pygame.event.Event(pygame.USEREVENT, attr1="attr1")
        pygame.event.post(e1)
        posted_event = pygame.event.poll()

        self.assertEqual(e1.attr1, posted_event.attr1, race_condition_notification)

        # fuzzing event types
        for i in range(1, 13):
            pygame.event.post(pygame.event.Event(EVENT_TYPES[i], **EVENT_TEST_PARAMS[EVENT_TYPES[i]]))

            self.assertEqual(
                pygame.event.poll().type, EVENT_TYPES[i], race_condition_notification
            )
    
    # @unittest.skip("At the moment, this test seems to fail on all platforms")
    def test_post_and_get_keydown(self):
        """Ensure keydown events can be posted to the queue."""
        surf = pygame.display.set_mode((10, 10))
        pygame.event.get()
        activemodkeys = pygame.key.get_mods()
        
        events = []
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_y, mod=activemodkeys))
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, unicode="g"))
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode=None))
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=None, window=None))
        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=activemodkeys, unicode="e"))
        
        for e in events:
            pygame.event.clear()
            pygame.event.post(e)
            posted_event = pygame.event.poll()
            self.assertEqual(e.type, posted_event.type, race_condition_notification)
            self.assertEqual(e.type, pygame.KEYDOWN, race_condition_notification)
            self.assertEqual(e.key, posted_event.key, race_condition_notification)

    def test_post_large_user_event(self):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {"a": "a" * 1024}))
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

    def test_get_type(self):
        ev = pygame.event.Event(pygame.USEREVENT)
        pygame.event.post(ev)
        queue = pygame.event.get(pygame.USEREVENT)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, pygame.USEREVENT)

    def test_get__empty_queue(self):
        """Ensure get() works correctly on an empty queue."""
        expected_events = []
        pygame.event.clear()

        # Ensure all events can be checked.
        retrieved_events = pygame.event.get()

        self.assertListEqual(retrieved_events, expected_events)

        # Ensure events can be checked individually.
        for event_type in EVENT_TYPES:
            retrieved_events = pygame.event.get(event_type)

            self.assertListEqual(retrieved_events, expected_events)

        # Ensure events can be checked as a sequence.
        retrieved_events = pygame.event.get(EVENT_TYPES)

        self.assertListEqual(retrieved_events, expected_events)

    def test_get__event_sequence(self):
        """Ensure get() can handle a sequence of event types."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
        other_event_type = pygame.MOUSEBUTTONUP

        # Test when no events in the queue.
        expected_events = []
        pygame.event.clear()
        retrieved_events = pygame.event.get(event_types)

        # don't use self._assertCountEqual here. This checks for
        # expected properties in events, and ignores unexpected ones, for
        # forward compatibility with SDL2.
        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)

        # Test when an event type not in the list is in the queue.
        expected_events = []
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))

        retrieved_events = pygame.event.get(event_types)

        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)

        # Test when 1 event type in the list is in the queue.
        expected_events = [pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]])]
        pygame.event.clear()
        pygame.event.post(expected_events[0])

        retrieved_events = pygame.event.get(event_types)

        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)

        # Test all events in the list are in the queue.
        pygame.event.clear()
        expected_events = []

        for etype in event_types:
            expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
            pygame.event.post(expected_events[-1])

        retrieved_events = pygame.event.get(event_types)

        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)

    def test_clear(self):
        """Ensure clear() removes all the events on the queue."""
        for e in EVENT_TYPES:
            pygame.event.post(pygame.event.Event(e, **EVENT_TEST_PARAMS[e]))
        poll_event = pygame.event.poll()

        self.assertNotEqual(poll_event.type, pygame.NOEVENT)

        pygame.event.clear()
        poll_event = pygame.event.poll()

        self.assertEqual(poll_event.type, pygame.NOEVENT, race_condition_notification)

    def test_clear__empty_queue(self):
        """Ensure clear() works correctly on an empty queue."""
        expected_events = []
        pygame.event.clear()

        # Test calling clear() on an already empty queue.
        pygame.event.clear()

        retrieved_events = pygame.event.get()

        self.assertListEqual(retrieved_events, expected_events)

    def test_clear__event_sequence(self):
        """Ensure a sequence of event types can be cleared from the queue."""
        cleared_event_types = EVENT_TYPES[:5]
        expected_event_types = EVENT_TYPES[5:10]
        expected_events = []

        # Add the events to the queue.
        for etype in cleared_event_types:
            pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))

        for etype in expected_events:
            expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
            pygame.event.post(expected_events[-1])

        # Clear the cleared_events from the queue.
        pygame.event.clear(cleared_event_types)

        # Check the rest of the events in the queue.
        remaining_events = pygame.event.get()

        self._assertCountEqual(remaining_events, expected_events)

    def test_event_name(self):
        """Ensure event_name() returns the correct event name."""
        for expected_name, event in NAMES_AND_EVENTS:
            self.assertEqual(
                pygame.event.event_name(event), expected_name, "0x{:X}".format(event)
            )

    def test_event_name__userevent_range(self):
        """Ensures event_name() returns the correct name for user events.

        Tests the full range of user events.
        """
        expected_name = "UserEvent"

        for event in range(pygame.USEREVENT, pygame.NUMEVENTS):
            self.assertEqual(
                pygame.event.event_name(event), expected_name, "0x{:X}".format(event)
            )

    def test_event_name__userevent_boundary(self):
        """Ensures event_name() does not return 'UserEvent' for events
        just outside the user event range.
        """
        unexpected_name = "UserEvent"

        for event in (pygame.USEREVENT - 1, pygame.NUMEVENTS):
            self.assertNotEqual(
                pygame.event.event_name(event), unexpected_name, "0x{:X}".format(event)
            )

    def test_wait(self):
        """Ensure wait() waits for an event on the queue."""
        # Test case without timeout.
        event = pygame.event.Event(EVENT_TYPES[0], **EVENT_TEST_PARAMS[EVENT_TYPES[0]])
        pygame.event.post(event)
        wait_event = pygame.event.wait()

        self.assertEqual(wait_event.type, event.type)

        # Test case with timeout and no event in the queue.
        if SDL1:
            with self.assertRaises(TypeError):
                pygame.event.wait(250)
        else:
            wait_event = pygame.event.wait(250)
            self.assertEqual(wait_event.type, pygame.NOEVENT)

        # Test case with timeout and an event in the queue.
        if SDL1:
            with self.assertRaises(TypeError):
                pygame.event.wait(250)
        else:
            event = pygame.event.Event(EVENT_TYPES[0], **EVENT_TEST_PARAMS[EVENT_TYPES[0]])
            pygame.event.post(event)
            wait_event = pygame.event.wait(250)

            self.assertEqual(wait_event.type, event.type)

    def test_peek(self):
        """Ensure queued events can be peeked at."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]

        for event_type in event_types:
            pygame.event.post(pygame.event.Event(event_type, **EVENT_TEST_PARAMS[event_type]))

        # Ensure events can be checked individually.
        for event_type in event_types:
            self.assertTrue(pygame.event.peek(event_type))

        # Ensure events can be checked as a sequence.
        self.assertTrue(pygame.event.peek(event_types))

    def test_peek__event_sequence(self):
        """Ensure peek() can handle a sequence of event types."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
        other_event_type = pygame.MOUSEBUTTONUP

        # Test when no events in the queue.
        pygame.event.clear()
        peeked = pygame.event.peek(event_types)

        self.assertFalse(peeked)

        # Test when an event type not in the list is in the queue.
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))

        peeked = pygame.event.peek(event_types)

        self.assertFalse(peeked)

        # Test when 1 event type in the list is in the queue.
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]]))

        peeked = pygame.event.peek(event_types)

        self.assertTrue(peeked)

        # Test all events in the list are in the queue.
        pygame.event.clear()
        for etype in event_types:
            pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))

        peeked = pygame.event.peek(event_types)

        self.assertTrue(peeked)

    def test_peek__empty_queue(self):
        """Ensure peek() works correctly on an empty queue."""
        pygame.event.clear()

        # Ensure all events can be checked.
        peeked = pygame.event.peek()

        self.assertFalse(peeked)

        # Ensure events can be checked individually.
        for event_type in EVENT_TYPES:
            peeked = pygame.event.peek(event_type)
            self.assertFalse(peeked)

        # Ensure events can be checked as a sequence.
        peeked = pygame.event.peek(EVENT_TYPES)

        self.assertFalse(peeked)

    def test_set_allowed(self):
        """Ensure a blocked event type can be unblocked/allowed."""
        event = EVENT_TYPES[0]
        pygame.event.set_blocked(event)

        self.assertTrue(pygame.event.get_blocked(event))

        pygame.event.set_allowed(event)

        self.assertFalse(pygame.event.get_blocked(event))

    def test_set_allowed__event_sequence(self):
        """Ensure a sequence of blocked event types can be unblocked/allowed.
        """
        event_types = [
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
        ]
        pygame.event.set_blocked(event_types)

        pygame.event.set_allowed(event_types)

        for etype in event_types:
            self.assertFalse(pygame.event.get_blocked(etype))

    def test_set_allowed_all(self):
        """Ensure all events can be unblocked/allowed at once."""
        pygame.event.set_blocked(None)

        for e in EVENT_TYPES:
            self.assertTrue(pygame.event.get_blocked(e))

        pygame.event.set_allowed(None)

        for e in EVENT_TYPES:
            self.assertFalse(pygame.event.get_blocked(e))

    def test_pump(self):
        """Ensure pump() functions properly."""
        pygame.event.pump()

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'requires the SDL_VIDEODRIVER to be a non "dummy" value',
    )
    def test_set_grab__and_get_symmetric(self):
        """Ensure event grabbing can be enabled and disabled.

        WARNING: Moving the mouse off the display during this test can cause it
                 to fail.
        """
        surf = pygame.display.set_mode((10, 10))
        pygame.event.set_grab(True)

        self.assertTrue(pygame.event.get_grab())

        pygame.event.set_grab(False)

        self.assertFalse(pygame.event.get_grab())

    def test_event_equality(self):
        a = pygame.event.Event(EVENT_TYPES[0], a=1)
        b = pygame.event.Event(EVENT_TYPES[0], a=1)
        c = pygame.event.Event(EVENT_TYPES[1], a=1)
        d = pygame.event.Event(EVENT_TYPES[0], a=2)

        self.assertTrue(a == a)
        self.assertFalse(a != a)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)
        self.assertFalse(a == c)
        self.assertTrue(a != d)
        self.assertFalse(a == d)

    def test_get_blocked(self):
        """Ensure an event's blocked state can be retrieved."""
        # Test each event is not blocked.
        pygame.event.set_allowed(None)

        for etype in EVENT_TYPES:
            blocked = pygame.event.get_blocked(etype)

            self.assertFalse(blocked)

        # Test each event type is blocked.
        pygame.event.set_blocked(None)

        for etype in EVENT_TYPES:
            blocked = pygame.event.get_blocked(etype)

            self.assertTrue(blocked)

    def test_get_blocked__event_sequence(self):
        """Ensure get_blocked() can handle a sequence of event types."""
        event_types = [
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
        ]

        # Test no event types in the list are blocked.
        blocked = pygame.event.get_blocked(event_types)

        self.assertFalse(blocked)

        # Test when 1 event type in the list is blocked.
        pygame.event.set_blocked(event_types[2])

        blocked = pygame.event.get_blocked(event_types)

        self.assertTrue(blocked)

        # Test all event types in the list are blocked.
        pygame.event.set_blocked(event_types)

        blocked = pygame.event.get_blocked(event_types)

        self.assertTrue(blocked)

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

if __name__ == "__main__":
    unittest.main()
