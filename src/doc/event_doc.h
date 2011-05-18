/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEEVENT "pygame module for interacting with events and queues"

#define DOC_PYGAMEEVENTPUMP "pump() -> None\ninternally process pygame event handlers"

#define DOC_PYGAMEEVENTGET "get() -> Eventlist\nget(type) -> Eventlist\nget(typelist) -> Eventlist\nget events from the queue"

#define DOC_PYGAMEEVENTPOLL "poll() -> Event\nget a single event from the queue"

#define DOC_PYGAMEEVENTWAIT "wait() -> Event\nwait for a single event from the queue"

#define DOC_PYGAMEEVENTPEEK "peek(type) -> bool\npeek(typelist) -> bool\ntest if event types are waiting on the queue"

#define DOC_PYGAMEEVENTCLEAR "clear() -> None\nclear(type) -> None\nclear(typelist) -> None\nremove all events from the queue"

#define DOC_PYGAMEEVENTEVENTNAME "event_name(type) -> string\nget the string name from and event id"

#define DOC_PYGAMEEVENTSETBLOCKED "set_blocked(type) -> None\nset_blocked(typelist) -> None\nset_blocked(None) -> None\ncontrol which events are allowed on the queue"

#define DOC_PYGAMEEVENTSETALLOWED "set_allowed(type) -> None\nset_allowed(typelist) -> None\nset_allowed(None) -> None\ncontrol which events are allowed on the queue"

#define DOC_PYGAMEEVENTGETBLOCKED "get_blocked(type) -> bool\ntest if a type of event is blocked from the queue"

#define DOC_PYGAMEEVENTSETGRAB "set_grab(bool) -> None\ncontrol the sharing of input devices with other applications"

#define DOC_PYGAMEEVENTGETGRAB "get_grab() -> bool\ntest if the program is sharing input devices"

#define DOC_PYGAMEEVENTPOST "post(Event) -> None\nplace a new event on the queue"

#define DOC_PYGAMEEVENTEVENT "Event(type, dict) -> Event\nEvent(type, **attributes) -> Event\ncreate a new event object"



/* Docs in a comment... slightly easier to read. */

/*

pygame.event
pygame module for interacting with events and queues

pygame.event.pump
 pump() -> None
internally process pygame event handlers

pygame.event.get
 get() -> Eventlist
 get(type) -> Eventlist
 get(typelist) -> Eventlist
get events from the queue

pygame.event.poll
 poll() -> Event
get a single event from the queue

pygame.event.wait
 wait() -> Event
wait for a single event from the queue

pygame.event.peek
 peek(type) -> bool
 peek(typelist) -> bool
test if event types are waiting on the queue

pygame.event.clear
 clear() -> None
 clear(type) -> None
 clear(typelist) -> None
remove all events from the queue

pygame.event.event_name
 event_name(type) -> string
get the string name from and event id

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
test if a type of event is blocked from the queue

pygame.event.set_grab
 set_grab(bool) -> None
control the sharing of input devices with other applications

pygame.event.get_grab
 get_grab() -> bool
test if the program is sharing input devices

pygame.event.post
 post(Event) -> None
place a new event on the queue

pygame.event.Event
 Event(type, dict) -> Event
 Event(type, **attributes) -> Event
create a new event object

*/