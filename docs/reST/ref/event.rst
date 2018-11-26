.. include:: common.txt

:mod:`pygame.event`
===================

.. module:: pygame.event
   :synopsis: pygame module for interacting with events and queues

| :sl:`pygame module for interacting with events and queues`

Pygame handles all its event messaging through an event queue. The routines in
this module help you manage that event queue. The input queue is heavily
dependent on the pygame display module. If the display has not been initialized
and a video mode not set, the event queue will not really work.

The queue is a regular queue of :class:`pygame.event.EventType` event objects,
there are a variety of ways to access the events it contains. From simply
checking for the existence of events, to grabbing them directly off the stack.

All events have a type identifier. This event type is in between the values of
``NOEVENT`` and ``NUMEVENTS``. All user defined events can have the value of
``USEREVENT`` or higher. It is recommended make sure your event id's follow
this system.

To get the state of various input devices, you can forego the event queue and
access the input devices directly with their appropriate modules; mouse, key,
and joystick. If you use this method, remember that pygame requires some form
of communication with the system window manager and other parts of the
platform. To keep pygame in synch with the system, you will need to call
``pygame.event.pump()`` to keep everything current. You'll want to call this
function usually once per game loop.

The event queue offers some simple filtering. This can help performance
slightly by blocking certain event types from the queue, use the
``pygame.event.set_allowed()`` and ``pygame.event.set_blocked()`` to work with
this filtering. All events default to allowed.

The event subsystem should be called from the main thread.  If you want to post
events into the queue from other threads, please use the fastevent package.

Joysticks will not send any events until the device has been initialized.

An ``EventType`` event object contains an event type identifier and a set of
member data. The event object contains no method functions, just member data.
EventType objects are retrieved from the pygame event queue. You can create
your own new events with the ``pygame.event.Event()`` function.

The SDL event queue has an upper limit on the number of events it can hold
(128  for standard SDL 1.2).
When the queue becomes full new events are quietly dropped.
To prevent lost events, especially input events which signal a quit
command, your program must regularly check for events and process them.
To speed up queue processing use :func:`pygame.event.set_blocked` to
limit which events get queued.

All EventType instances have an event type identifier, accessible as the
``EventType.type`` property. You may also get full access to the event object's
attributes through the ``EventType.__dict__`` attribute. All other member
lookups will be passed through to the object's dictionary values.

While debugging and experimenting, you can print an event object for a quick
display of its type and members. Events that come from the system will have a
guaranteed set of member items based on the type. Here is a list of the
event attributes defined with each event type.

::

    QUIT	            none
    ACTIVEEVENT	      gain, state
    KEYDOWN	          unicode, key, mod
    KEYUP	            key, mod
    MOUSEMOTION	      pos, rel, buttons
    MOUSEBUTTONUP     pos, button
    MOUSEBUTTONDOWN   pos, button
    JOYAXISMOTION     joy, axis, value
    JOYBALLMOTION     joy, ball, rel
    JOYHATMOTION      joy, hat, value
    JOYBUTTONUP       joy, button
    JOYBUTTONDOWN     joy, button
    VIDEORESIZE       size, w, h
    VIDEOEXPOSE       none
    USEREVENT         code

Events support equality comparison. Two events are equal if they are the same
type and have identical attribute values. Inequality checks also work.

.. versionadded:: 1.9.2

    On MacOSX, USEREVENT can have `code = pygame.USEREVENT_DROPFILE`. That
    means the user is trying to open a file with your application. The filename
    can be found at `event.filename`

.. versionadded:: 1.9.5

    When compiled with SDL2, pygame has these events.

    AUDIODEVICEADDED   which, iscapture
    AUDIODEVICEREMOVED which, iscapture

.. function:: pump

   | :sl:`internally process pygame event handlers`
   | :sg:`pump() -> None`

   For each frame of your game, you will need to make some sort of call to the
   event queue. This ensures your program can internally interact with the rest
   of the operating system. If you are not using other event functions in your
   game, you should call ``pygame.event.pump()`` to allow pygame to handle
   internal actions.

   This function is not necessary if your program is consistently processing
   events on the queue through the other :mod:`pygame.event` functions.

   There are important things that must be dealt with internally in the event
   queue. The main window may need to be repainted or respond to the system. If
   you fail to make a call to the event queue for too long, the system may
   decide your program has locked up.

   .. ## pygame.event.pump ##

.. function:: get

   | :sl:`get events from the queue`
   | :sg:`get() -> Eventlist`
   | :sg:`get(type) -> Eventlist`
   | :sg:`get(typelist) -> Eventlist`

   This will get all the messages and remove them from the queue. If a type or
   sequence of types is given only those messages will be removed from the
   queue.

   If you are only taking specific events from the queue, be aware that the
   queue could eventually fill up with the events you are not interested.

   .. ## pygame.event.get ##

.. function:: poll

   | :sl:`get a single event from the queue`
   | :sg:`poll() -> EventType instance`

   Returns a single event from the queue. If the event queue is empty an event
   of type ``pygame.NOEVENT`` will be returned immediately. The returned event
   is removed from the queue.

   .. ## pygame.event.poll ##

.. function:: wait

   | :sl:`wait for a single event from the queue`
   | :sg:`wait() -> EventType instance`

   Returns a single event from the queue. If the queue is empty this function
   will wait until one is created. The event is removed from the queue once it
   has been returned. While the program is waiting it will sleep in an idle
   state. This is important for programs that want to share the system with
   other applications.

   .. ## pygame.event.wait ##

.. function:: peek

   | :sl:`test if event types are waiting on the queue`
   | :sg:`peek(type) -> bool`
   | :sg:`peek(typelist) -> bool`

   Returns true if there are any events of the given type waiting on the queue.
   If a sequence of event types is passed, this will return True if any of
   those events are on the queue.

   .. ## pygame.event.peek ##

.. function:: clear

   | :sl:`remove all events from the queue`
   | :sg:`clear() -> None`
   | :sg:`clear(type) -> None`
   | :sg:`clear(typelist) -> None`

   Remove all events or events of a specific type from the queue. This has the
   same effect as ``pygame.event.get()`` except nothing is returned. This can
   be slightly more efficient when clearing a full event queue.

   .. ## pygame.event.clear ##

.. function:: event_name

   | :sl:`get the string name from and event id`
   | :sg:`event_name(type) -> string`

   Pygame uses integer ids to represent the event types. If you want to report
   these types to the user they should be converted to strings. This will
   return a the simple name for an event type. The string is in the WordCap
   style.

   .. ## pygame.event.event_name ##

.. function:: set_blocked

   | :sl:`control which events are allowed on the queue`
   | :sg:`set_blocked(type) -> None`
   | :sg:`set_blocked(typelist) -> None`
   | :sg:`set_blocked(None) -> None`

   The given event types are not allowed to appear on the event queue. By
   default all events can be placed on the queue. It is safe to disable an
   event type multiple times.

   If None is passed as the argument, this has the opposite effect and ``ALL``
   of the event types are allowed to be placed on the queue.

   .. ## pygame.event.set_blocked ##

.. function:: set_allowed

   | :sl:`control which events are allowed on the queue`
   | :sg:`set_allowed(type) -> None`
   | :sg:`set_allowed(typelist) -> None`
   | :sg:`set_allowed(None) -> None`

   The given event types are allowed to appear on the event queue. By default
   all events can be placed on the queue. It is safe to enable an event type
   multiple times.

   If None is passed as the argument, ``NONE`` of the event types are allowed
   to be placed on the queue.

   .. ## pygame.event.set_allowed ##

.. function:: get_blocked

   | :sl:`test if a type of event is blocked from the queue`
   | :sg:`get_blocked(type) -> bool`

   Returns true if the given event type is blocked from the queue.

   .. ## pygame.event.get_blocked ##

.. function:: set_grab

   | :sl:`control the sharing of input devices with other applications`
   | :sg:`set_grab(bool) -> None`

   When your program runs in a windowed environment, it will share the mouse
   and keyboard devices with other applications that have focus. If your
   program sets the event grab to True, it will lock all input into your
   program.

   It is best to not always grab the input, since it prevents the user from
   doing other things on their system.

   .. ## pygame.event.set_grab ##

.. function:: get_grab

   | :sl:`test if the program is sharing input devices`
   | :sg:`get_grab() -> bool`

   Returns true when the input events are grabbed for this application. Use
   ``pygame.event.set_grab()`` to control this state.

   .. ## pygame.event.get_grab ##

.. function:: post

   | :sl:`place a new event on the queue`
   | :sg:`post(Event) -> None`

   This places a new event at the end of the event queue. These Events will
   later be retrieved from the other queue functions.

   This is usually used for placing ``pygame.USEREVENT`` events on the queue.
   Although any type of event can be placed, if using the system event types
   your program should be sure to create the standard attributes with
   appropriate values.

   If the SDL event queue is full a :exc:`pygame.error` is raised.

   .. ## pygame.event.post ##

.. function:: Event

   | :sl:`create a new event object`
   | :sg:`Event(type, dict) -> EventType instance`
   | :sg:`Event(type, **attributes) -> EventType instance`

   Creates a new event with the given type. The event is created with the given
   attributes and values. The attributes can come from a dictionary argument
   with string keys, or from keyword arguments.

.. class:: EventType

   | :sl:`pygame object for representing SDL events`

   A Python object that represents an SDL event. User event instances are
   created with an `Event` function call. The `EventType` type is not directly
   callable. `EventType` instances support attribute assignment and deletion.

   .. attribute:: type

      | :sl:`SDL event type identifier.`
      | :sg:`type -> int`

      Read only. Predefined event identifiers are `QUIT` and `MOUSEMOTION`, for
      example. For user created event objects, this is the `type` argument
      passed to :func:`pygame.event.Event`.

   .. attribute:: __dict__

      | :sl:`Event object attribute dictionary`
      | :sg:`__dict__ -> dict`

      Read only. The event type specific attributes of an event. As an example,
      this would contain the `unicode`, `key`, and `mod` attributes of a
      `KEYDOWN` event. The `dict` attribute is a synonym, for backward
      compatibility.

   Mutable attributes are new to pygame 1.9.2.

   .. ## pygame.event.Event ##

.. ## pygame.event ##
