.. include:: ../common.txt

.. highlight:: c

********************************************************
  Class BufferProxy API exported by pgyame.bufferproxy
********************************************************

src_c/bufferproxy.c
===================

This extension module defines Python type :py:class:`pygame.BufferProxy`.

Header file: src_c/include/pygame_bufferproxy.h


.. c:var:: PyTypeObject *pgBufproxy_Type

   The pygame buffer proxy object type pygame.BufferProxy.

.. c:function:: int pgBufproxy_Check(PyObject *x)

   Return true if Python object *x* is a :py:class:`pygame.BufferProxy` instance,
   false otherwise.
   This will return false on :py:class:`pygame.BufferProxy` subclass instances as well.

.. c:function:: PyObject* pgBufproxy_New(PyObject *obj, getbufferproc get_buffer)

   Return a new :py:class:`pygame.BufferProxy` instance.
   Argument *obj* is the Python object that has its data exposed.
   It may be ``NULL``.
   Argument *get_buffer* is the :c:type:`pg_buffer` get callback.
   It must not be ``NULL``.
   On failure raise a Python error and return ``NULL``.

.. c:function:: PyObject* pgBufproxy_GetParent(PyObject *obj)

   Return the Python object wrapped by buffer proxy *obj*.
   Argument *obj* must not be ``NULL``.
   On failure, raise a Python error and return ``NULL``.

.. c:function:: int pgBufproxy_Trip(PyObject *obj)

   Cause the buffer proxy object *obj* to create a :c:type:`pg_buffer` view of its parent.
   Argument *obj* must not be ``NULL``.
   Return ``0`` on success, otherwise raise a Python error and return ``-1``.
