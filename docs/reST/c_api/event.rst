.. include:: ../common.txt

.. highlight:: c

********************************
  API exported by pygame.event
********************************

src_c/event.c
=============

The extension module :py:mod:`pygame.event`.

Header file: src_c/include/pygame.h


.. c:type:: pgEventObject

   The :py:class:`pygame.event.EventType` object C struct.

   .. c:member:: int type

      The event type code.

.. c:type:: pgEvent_Type

   The pygame event object type :py:class:`pygame.event.EventType`.

.. c:function:: int pgEvent_Check(PyObject *x)

   Return true if *x* is a pygame event instance

   Will return false if *x* is a subclass of event.
   This is a macro. No check is made that *x* is not ``NULL``.

.. c:function:: PyObject* pgEvent_New(SDL_Event *event)

   Return a new pygame event instance for the SDL *event*.
   If *event* is ``NULL`` then create an empty event object.
   On failure raise a Python exception and return ``NULL``.

.. c:function:: int pg_post_event(Uint32 type, PyObject *dict)

   Posts a pygame event that is an ``SDL_USEREVENT`` on the SDL side. This
   function takes a python dict, which can be NULL too.
   This function does not need GIL to be held if dict is NULL, but needs GIL
   otherwise. Just like the SDL ``SDL_PushEvent`` function, returns 1 on
   success, 0 if the event was not posted due to it being blocked, and -1 on
   failure.
