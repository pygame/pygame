=================
pygame2.sdl.event
=================

The :mod:`pygame2.sdl.event` C API contains objects and functions for
accessing and manipulating the SDL event system queue.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_event (void)

  Imports the :mod:`pygame2.sdl.event` module. This returns 0 on success
  and -1 on failure.

Macros
------
.. data:: PYGAME_USEREVENT

  Constant for SDL user event types. This is usually placed into the
  :cmember:`SDL_Event.data1` field for :cmacro:`SDL_USEREVENT` events.

.. data:: PYGAME_USEREVENT_CODE

  Constant for SDL user event types. This is usually placed into the
  :cmember:`SDL_Event.code` field for :cmacro:`SDL_USEREVENT` events.

.. note::

   If you handle events programmatically from C code or interoperate with other
   SDL-based applications or libraries, which exchange events, you might have to
   treat Python-based SDL events in a special way.

   To check, whether the event comes from Pygame, you can check the
   :cmember:`SDL_Event.data1` and :cmember:`SDL_Event.code` for the
   :cdata:`PYGAME_USEREVENT` and :cdata:`PYGAME_USEREVENT_CODE` values.
   If both are set, this indicates that the event comes from Pygame and carries
   a Python dictionary in the :cmember:`SDL_Event.data2` field. ::
   
     if (event->user.code == PYGAME_USEREVENT_CODE &&
         event->user.data1 == (void*)PYGAME_USEREVENT)
     {
         /* It comes from pygame*/
         PyObject *dict = (PyObject*) event->user.data2;
         ...
     }

  
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
  the :ctype:`PyEvent`. This returns 1 on success and 0 on failure.
