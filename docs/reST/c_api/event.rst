.. include:: ../common.txt

.. highlight:: c

********************************
  API exported by pygame.event
********************************

src_c/event.c
=============

The extsion module :py:mod:`pygame.event`.

Header file: src_c/include/pygame.h


.. c:type:: pgEventObject

   The :py:class:`pygame.event.EventType` object C struct.

   .. c:member:: int type

      The event type code.

.. c:var:: pgEvent_Type

   The pygame event object type :py:class:`pygame.event.EventType`.

.. c:function:: int pgEvent_Check(PyObject *x)

   Return true if *x* is a pygame event instance

   Will return false if *x* is a subclass of event.
   This is a macro. No check is made that *x* is not ``NULL``.

.. c:function:: PyObject* pgEvent_New(SDL_Event *event)

   Return a new pygame event instance for the SDL *event*.
   If *event* is ``NULL`` then create an empty event object.
   On failure raise a Python exception and return ``NULL``.

.. c:function:: PyObject* pgEvent_New2(int type, PyObject *dict)

   Return a new pygame event instance of SDL *type* and with
   attribute dictionary *dict*.
   If `dict` is ``NULL`` an empty attribute dictionary is created.
   On failure raise a Python exception and return ``NULL``.

.. c:function:: int pgEvent_FillUserEvent(pgEventObject *e, SDL_Event *event)

   Fill SDL event *event* with information from pygame user event instance *e*.
   Return ``0`` on success, ``-1`` otherwise.
