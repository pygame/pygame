=======================
pygame2.sdl.event C API
=======================

The :mod:`pygame2.sdl.event` C API contains objects and functions for
accessing and manipulating the SDL event system queue.

TODO: describe user event encapsulation!

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_event (void)

  Imports the :mod:`pygame2.sdl.event` module. This returns 0 on success
  and -1 on failure.

Macros
------
.. cmacro:: PYGAME_USEREVENT

  Constant for SDL user event types. This is usually placed into the
  :cmember:`SDL_Event.data1` field for :cmacro:`SDL_USEREVENT` events.

.. cmacro:: PYGAME_USEREVENT_CODE

  Constant for SDL user event types. This is usually placed into the
  :cmember:`SDL_Event.code` field for :cmacro:`SDL_USEREVENT` events.

PyEvent
-------
.. ctype:: PyEvent
.. ctype:: PyEvent_Type

The PyEvent object is used for accessing and manipulating events that
occur on the SDL event queue.

Members
^^^^^^^
.. cmember:: Uint8 PyEvent.type

  The SDL event type.

.. cmember:: PyObject* PyEvent.dict

  The dictionary containing the data carried by the event.

Functions
^^^^^^^^^^
.. cfunction:: int PyEvent_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyEvent` or a subclass of
  :ctype:`PyEvent`.

.. cfunction:: PyObject* PyEvent_New (SDL_Event* event)

  Creates a new :ctype:`PyEvent` object from the passed
  :ctype:`SDL_Event`. Once created, the :class:`SDL_Event` is not
  required to be hold in memory anymore. On failure, this returns NULL.

.. cfunction:: int PyEvent_SDLEventFromEvent (PyObject *obj, SDL_Event *event)

  Fills the passed :ctype:`SDL_Event` *event* with the information of
  the :ctype:`PyEvent`. On success, this will return 1. If the passed
  *event* is NULL, 0 will be returned and a :exc:`ValueError` be set, if
  the passed *obj* is not a :ctype:`PyEvent` a :exc:`TypeError` be set.
  On any other error an exception will be raised and 0 returned.
