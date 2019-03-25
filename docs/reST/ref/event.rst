.. include:: common.txt

:mod:`pygame.event`
===================

.. module:: pygame.event
   :synopsis: pygame module for interacting with events and queues

| :sl:`pygame module for interacting with events and queues`

Pygame handles all its event messaging through an event queue. The routines in
this module help you manage that event queue. The input queue is heavily
dependent on the :mod:`pygame.display` module. If the display has not been
initialized and a video mode not set, the event queue may not work properly.
The event subsystem should be called from the main thread. If you want to post
events into the queue from other threads, please use the
:mod:`pygame.fastevent` module.

The event queue has an upper limit on the number of events it can hold
(128 for standard SDL 1.2). When the queue becomes full new events are quietly
dropped. To prevent lost events, especially input events which signal a quit
command, your program must regularly check for events and process them.
To speed up queue processing use :func:`pygame.event.set_blocked()` to
limit which events get queued.

To get the state of various input devices, you can forego the event queue and
access the input devices directly with their appropriate modules:
:mod:`pygame.mouse`, :mod:`pygame.key`, and :mod:`pygame.joystick`. If you use
this method, remember that pygame requires some form of communication with the
system window manager and other parts of the platform. To keep pygame in sync
with the system, you will need to call :func:`pygame.event.pump()` to keep
everything current. Usually, this should be called once per game loop.
Note: Joysticks will not send any events until the device has been initialized.

The event queue contains :class:`pygame.event.EventType` event objects.
There are a variety of ways to access the queued events, from simply
checking for the existence of events, to grabbing them directly off the stack.
The event queue also offers some simple filtering which can slightly help
performance by blocking certain event types from the queue. Use
:func:`pygame.event.set_allowed()` and :func:`pygame.event.set_blocked()` to
change this filtering. By default, all event types can be placed on the queue.

All :class:`pygame.event.EventType` instances contain an event type identifier
and attributes specific to that event type. The event type identifier is
accessible as the :attr:`pygame.event.EventType.type` property. Any of the
event specific attributes can be accessed through the
:attr:`pygame.event.EventType.__dict__` attribute or directly as an attribute
of the event object (as member lookups are passed through to the object's
dictionary values). The event object has no method functions. Users can create
their own new events with the :func:`pygame.event.Event()` function.

The event type identifier is in between the values of ``NOEVENT`` and
``NUMEVENTS``. User defined events should have a value in the inclusive range
of ``USEREVENT`` to ``NUMEVENTS - 1``. It is recommended all user events follow
this system.

Events support equality and inequality comparisons. Two events are equal if
they are the same type and have identical attribute values.

While debugging and experimenting, you can print an event object for a quick
display of its type and members. The function :func:`pygame.event.event_name()`
can be used to get a string representing the name of the event type.

Events that come from the system will have a guaranteed set of member
attributes based on the type. The following is a list event types with their
specific attributes.

::

    QUIT	      none
    ACTIVEEVENT	      gain, state
    KEYDOWN	      unicode, key, mod
    KEYUP	      key, mod
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

|

.. versionadded:: 1.9.2

On MacOSX when a file is opened using a pygame application, a ``USEREVENT``
with its ``code`` attribute set to ``pygame.USEREVENT_DROPFILE`` is generated.
There is an additional attribute called ``filename`` where the name of the file
being accessed is stored.

::

    USEREVENT         code=pygame.USEREVENT_DROPFILE, filename

|

.. versionadded:: 1.9.5

When compiled with SDL2, pygame has these additional events and their
attributes.

::

    AUDIODEVICEADDED   which, iscapture
    AUDIODEVICEREMOVED which, iscapture
    FINGERMOTION       touch_id, finger_id, x, y, dx, dy
    FINGERDOWN         touch_id, finger_id, x, y, dx, dy
    FINGERUP           touch_id, finger_id, x, y, dx, dy
    MULTIGESTURE       touch_id, x, y, pinched, rotated, num_fingers
    TEXTEDITING        text, start, length
    TEXTINPUT          text

|

.. versionadded:: 1.9.5

When compiled with SDL2, pygame can recognize text or files dropped
into the window. If a file is dropped, ``file`` will be its path.

::

   SDL_DROPBEGIN
   SDL_DROPEND
   SDL_DROPFILE        file
   SDL_DROPTEXT        text

|

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

   .. caution::
      This function should only be called in the thread that initialized :mod:`pygame.display`.

   .. ## pygame.event.pump ##

.. function:: get

   | :sl:`get events from the queue`
   | :sg:`get(eventtype=None) -> Eventlist`
   | :sg:`get(eventtype=None, pump=True) -> Eventlist`

   This will get all the messages and remove them from the queue. If a type or
   sequence of types is given only those messages will be removed from the
   queue.

   If you are only taking specific events from the queue, be aware that the
   queue could eventually fill up with the events you are not interested.

   If ``pump`` is ``True`` (the default), then :func:`pygame.event.pump()` will be called.

   .. versionadded:: 1.9.5 ``pump``

   .. ## pygame.event.get ##

.. function:: poll

   | :sl:`get a single event from the queue`
   | :sg:`poll() -> EventType instance`

   Returns a single event from the queue. If the event queue is empty an event
   of type ``pygame.NOEVENT`` will be returned immediately. The returned event
   is removed from the queue.

   .. caution::
      This function should only be called in the thread that initialized :mod:`pygame.display`.

   .. ## pygame.event.poll ##

.. function:: wait

   | :sl:`wait for a single event from the queue`
   | :sg:`wait() -> EventType instance`

   Returns a single event from the queue. If the queue is empty this function
   will wait until one is created. The event is removed from the queue once it
   has been returned. While the program is waiting it will sleep in an idle
   state. This is important for programs that want to share the system with
   other applications.

   .. caution::
      This function should only be called in the thread that initialized :mod:`pygame.display`.

   .. ## pygame.event.wait ##

.. function:: peek

   | :sl:`test if event types are waiting on the queue`
   | :sg:`peek(eventtype=None) -> bool`
   | :sg:`peek(eventtype=None, pump=True) -> bool`

   Returns ``True`` if there are any events of the given type waiting on the
   queue. If a sequence of event types is passed, this will return ``True`` if
   any of those events are on the queue.

   If ``pump`` is ``True`` (the default), then :func:`pygame.event.pump()` will be called.

   .. versionadded:: 1.9.5 ``pump``

   .. ## pygame.event.peek ##

.. function:: clear

   | :sl:`remove all events from the queue`
   | :sg:`clear(eventtype=None) -> None`
   | :sg:`clear(eventtype=None, pump=True) -> None`

   Removes all events from the queue. If ``eventtype`` is given, removes the given event
   or sequence of events. This has the same effect as :func:`pygame.event.get()` except ``None``
   is returned. It can be slightly more efficient when clearing a full event queue.

   If ``pump`` is ``True`` (the default), then :func:`pygame.event.pump()` will be called.

   .. versionadded:: 1.9.5 ``pump``

   .. ## pygame.event.clear ##

.. function:: event_name

   | :sl:`get the string name from an event id`
   | :sg:`event_name(type) -> string`

   Returns a string representing the name (in CapWords style) of the given
   event type.

   "UserEvent" is returned for all values in the user event id range.
   "Unknown" is returned when the event type does not exist.

   .. ## pygame.event.event_name ##

.. function:: set_blocked

   | :sl:`control which events are allowed on the queue`
   | :sg:`set_blocked(type) -> None`
   | :sg:`set_blocked(typelist) -> None`
   | :sg:`set_blocked(None) -> None`

   The given event types are not allowed to appear on the event queue. By
   default all events can be placed on the queue. It is safe to disable an
   event type multiple times.

   If ``None`` is passed as the argument, ALL of the event types are blocked
   from being placed on the queue.

   .. ## pygame.event.set_blocked ##

.. function:: set_allowed

   | :sl:`control which events are allowed on the queue`
   | :sg:`set_allowed(type) -> None`
   | :sg:`set_allowed(typelist) -> None`
   | :sg:`set_allowed(None) -> None`

   The given event types are allowed to appear on the event queue. By default,
   all event types can be placed on the queue. It is safe to enable an event
   type multiple times.

   If ``None`` is passed as the argument, ALL of the event types are allowed
   to be placed on the queue.

   .. ## pygame.event.set_allowed ##

.. function:: get_blocked

   | :sl:`test if a type of event is blocked from the queue`
   | :sg:`get_blocked(type) -> bool`

   Returns ``True`` if the given event type is blocked from the queue.

   .. ## pygame.event.get_blocked ##

.. function:: set_grab

   | :sl:`control the sharing of input devices with other applications`
   | :sg:`set_grab(bool) -> None`

   When your program runs in a windowed environment, it will share the mouse
   and keyboard devices with other applications that have focus. If your
   program sets the event grab to ``True``, it will lock all input into your
   program.

   It is best to not always grab the input, since it prevents the user from
   doing other things on their system.

   .. ## pygame.event.set_grab ##

.. function:: get_grab

   | :sl:`test if the program is sharing input devices`
   | :sg:`get_grab() -> bool`

   Returns ``True`` when the input events are grabbed for this application.

   .. ## pygame.event.get_grab ##

.. function:: post

   | :sl:`place a new event on the queue`
   | :sg:`post(Event) -> None`

   Places the given event at the end of the event queue.

   This is usually used for placing ``pygame.USEREVENT`` events on the queue.
   Although any type of event can be placed, if using the system event types
   your program should be sure to create the standard attributes with
   appropriate values.

   If the event queue is full a :exc:`pygame.error` is raised.

   .. ## pygame.event.post ##

.. function:: Event

   | :sl:`create a new event object`
   | :sg:`Event(type, dict) -> EventType instance`
   | :sg:`Event(type, \**attributes) -> EventType instance`

   Creates a new event with the given type and attributes. The attributes can
   come from a dictionary argument with string keys or from keyword arguments.

   .. ## pygame.event.Event ##

.. class:: EventType

   | :sl:`pygame object for representing events`

   A pygame object that represents an event. User event instances are created
   with an :func:`pygame.event.Event()` function call. The ``EventType`` type
   is not directly callable. ``EventType`` instances support attribute
   assignment and deletion.

   .. attribute:: type

      | :sl:`event type identifier.`
      | :sg:`type -> int`

      Read-only. The event type identifier. For user created event
      objects, this is the ``type`` argument passed to
      :func:`pygame.event.Event()`.

      For example, some predefined event identifiers are ``QUIT`` and
      ``MOUSEMOTION``.

      .. ## pygame.event.EventType.type ##

   .. attribute:: __dict__

      | :sl:`event attribute dictionary`
      | :sg:`__dict__ -> dict`

      Read-only. The event type specific attributes of an event. The
      ``dict`` attribute is a synonym for backward compatibility.

      For example, the attributes of a ``KEYDOWN`` event would be ``unicode``,
      ``key``, and ``mod``

      .. ## pygame.event.EventType.__dict__ ##

   .. versionadded:: 1.9.2 Mutable attributes.

   .. ## pygame.event.EventType ##

.. ## pygame.event ##
