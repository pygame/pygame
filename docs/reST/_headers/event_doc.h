/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMEEVENT "pygame module for interacting with events and queues"
#define DOC_PYGAMEEVENTPUMP "pump() -> None\ninternally process pygame event handlers"
#define DOC_PYGAMEEVENTGET "get(eventtype=None) -> Eventlist\nget(eventtype=None, pump=True) -> Eventlist\nget(eventtype=None, pump=True, exclude=None) -> Eventlist\nget events from the queue"
#define DOC_PYGAMEEVENTPOLL "poll() -> Event instance\nget a single event from the queue"
#define DOC_PYGAMEEVENTWAIT "wait() -> Event instance\nwait(timeout) -> Event instance\nwait for a single event from the queue"
#define DOC_PYGAMEEVENTPEEK "peek(eventtype=None) -> bool\npeek(eventtype=None, pump=True) -> bool\ntest if event types are waiting on the queue"
#define DOC_PYGAMEEVENTCLEAR "clear(eventtype=None) -> None\nclear(eventtype=None, pump=True) -> None\nremove all events from the queue"
#define DOC_PYGAMEEVENTEVENTNAME "event_name(type) -> string\nget the string name from an event id"
#define DOC_PYGAMEEVENTSETBLOCKED "set_blocked(type) -> None\nset_blocked(typelist) -> None\nset_blocked(None) -> None\ncontrol which events are allowed on the queue"
#define DOC_PYGAMEEVENTSETALLOWED "set_allowed(type) -> None\nset_allowed(typelist) -> None\nset_allowed(None) -> None\ncontrol which events are allowed on the queue"
#define DOC_PYGAMEEVENTGETBLOCKED "get_blocked(type) -> bool\nget_blocked(typelist) -> bool\ntest if a type of event is blocked from the queue"
#define DOC_PYGAMEEVENTSETGRAB "set_grab(bool) -> None\ncontrol the sharing of input devices with other applications"
#define DOC_PYGAMEEVENTGETGRAB "get_grab() -> bool\ntest if the program is sharing input devices"
#define DOC_PYGAMEEVENTSETKEYBOARDGRAB "set_keyboard_grab(bool) -> None\ngrab enables capture of system keyboard shortcuts like Alt+Tab or the Meta/Super key."
#define DOC_PYGAMEEVENTGETKEYBOARDGRAB "get_keyboard_grab() -> bool\nget the current keyboard grab state"
#define DOC_PYGAMEEVENTPOST "post(Event) -> bool\nplace a new event on the queue"
#define DOC_PYGAMEEVENTCUSTOMTYPE "custom_type() -> int\nmake custom user event type"
#define DOC_PYGAMEEVENTEVENT "Event(type, dict) -> Event\nEvent(type, **attributes) -> Event\npygame object for representing events"
#define DOC_EVENTTYPE "type -> int\nevent type identifier."
#define DOC_EVENTDICT "__dict__ -> dict\nevent attribute dictionary"


/* Docs in a comment... slightly easier to read. */

/*

pygame.event
pygame module for interacting with events and queues

pygame.event.pump
 pump() -> None
internally process pygame event handlers

pygame.event.get
 get(eventtype=None) -> Eventlist
 get(eventtype=None, pump=True) -> Eventlist
 get(eventtype=None, pump=True, exclude=None) -> Eventlist
get events from the queue

pygame.event.poll
 poll() -> Event instance
get a single event from the queue

pygame.event.wait
 wait() -> Event instance
 wait(timeout) -> Event instance
wait for a single event from the queue

pygame.event.peek
 peek(eventtype=None) -> bool
 peek(eventtype=None, pump=True) -> bool
test if event types are waiting on the queue

pygame.event.clear
 clear(eventtype=None) -> None
 clear(eventtype=None, pump=True) -> None
remove all events from the queue

pygame.event.event_name
 event_name(type) -> string
get the string name from an event id

pygame.event.set_blocked
 set_blocked(type) -> None
 set_blocked(typelist) -> None
 set_blocked(None) -> None
control which events are allowed on the queue

pygame.event.set_allowed
 set_allowed(type) -> None
 set_allowed(typelist) -> None
 set_allowed(None) -> None
control which events are allowed on the queue

pygame.event.get_blocked
 get_blocked(type) -> bool
 get_blocked(typelist) -> bool
test if a type of event is blocked from the queue

pygame.event.set_grab
 set_grab(bool) -> None
control the sharing of input devices with other applications

pygame.event.get_grab
 get_grab() -> bool
test if the program is sharing input devices

pygame.event.set_keyboard_grab
 set_keyboard_grab(bool) -> None
grab enables capture of system keyboard shortcuts like Alt+Tab or the Meta/Super key.

pygame.event.get_keyboard_grab
 get_keyboard_grab() -> bool
get the current keyboard grab state

pygame.event.post
 post(Event) -> bool
place a new event on the queue

pygame.event.custom_type
 custom_type() -> int
make custom user event type

pygame.event.Event
 Event(type, dict) -> Event
 Event(type, **attributes) -> Event
pygame object for representing events

pygame.event.Event.type
 type -> int
event type identifier.

pygame.event.Event.__dict__
 __dict__ -> dict
event attribute dictionary

*/