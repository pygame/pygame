.. include:: ../common.txt

.. highlight:: c

******************************************
  High level API exported by pygame.base
******************************************

src_c/base.c
============

This extension module defines general purpose routines for starting and stopping
SDL as well as various conversion routines uses elsewhere in pygame.

C header: src_c/include/pygame.h

.. c:var:: PyObject* pgExc_SDLError

   This is :py:exc:`pygame.error`, the exception type used to raise SDL errors.

.. c:function:: int pg_mod_autoinit(const char* modname)

   Inits a pygame module, which has the name ``modname``
   Return ``1`` on success, ``0`` on error, with python
   error set.

.. c:function:: void pg_mod_autoquit(const char* modname)

   Quits a pygame module, which has the name ``modname``

.. c:function:: void pg_RegisterQuit(void (*f)(void))

   Register function *f* as a callback on Pygame termination.
   Multiple functions can be registered.
   Functions are called in the reverse order they were registered.

.. c:function:: int pg_IntFromObj(PyObject *obj, int *val)

   Convert number like object *obj* to C int and place in argument *val*.
   Return ``1`` on success, else ``0``.
   No Python exceptions are raised.

.. c:function:: int pg_IntFromObjIndex(PyObject *obj, int index, int *val)

   Convert number like object at position *i* in sequence *obj*
   to C int and place in argument *val*.
   Return ``1`` on success, ``0`` on failure.
   No Python exceptions are raised.

.. c:function:: int pg_TwoIntsFromObj(PyObject *obj, int *val1, int *v2)

   Convert the two number like objects in length 2 sequence *obj*
   to C int and place in arguments *val1* and *val2* respectively.
   Return ``1`` on success, ``0`` on failure.
   No Python exceptions are raised.

.. c:function:: int pg_FloatFromObj(PyObject *obj, float *val)

   Convert number like object *obj* to C float and place in argument *val*.
   Returns ``1`` on success, ``0`` on failure.
   No Python exceptions are raised.


.. c:function:: int pg_FloatFromObjIndex(PyObject *obj, int index, float *val)

   Convert number like object at position *i* in sequence *obj*
   to C float and place in argument *val*.
   Return ``1`` on success, else ``0``.
   No Python exceptions are raised.

.. c:function:: int pg_TwoFloatsFromObj(PyObject *obj, float *val1, float *val2)

   Convert the two number like objects in length 2 sequence *obj*
   to C float and place in arguments *val1* and *val2* respectively.
   Return ``1`` on success, else ``0``.
   No Python exceptions are raised.

.. c:function:: int pg_UintFromObj(PyObject *obj, Uint32 *val)

   Convert number like object *obj* to unsigned 32 bit integer and place
   in argument *val*.
   Return ``1`` on success, else ``0``.
   No Python exceptions are raised.

.. c:function:: int pg_UintFromObjIndex(PyObject *obj, int _index, Uint32 *val)

   Convert number like object at position *i* in sequence *obj*
   to unsigned 32 bit integer and place in argument *val*.
   Return ``1`` on success, else ``0``.
   No Python exceptions are raised.

.. c:function:: int pg_RGBAFromObj(PyObject *obj, Uint8 *RGBA)

   Convert the color represented by object *obj* into a red, green, blue, alpha
   length 4 C array *RGBA*.
   The object must be a length 3 or 4 sequence of numbers having values
   between 0 and 255 inclusive.
   For a length 3 sequence an alpha value of 255 is assumed.
   Return ``1`` on success, ``0`` otherwise.
   No Python exceptions are raised.

.. c:type:: pg_buffer

   .. c:member:: Py_buffer view

      A standard buffer description

   .. c:member:: PyObject* consumer

      The object holding the buffer

   .. c:member:: pybuffer_releaseproc release_buffer

      A buffer release callback.

.. c:var:: PyObject *pgExc_BufferError

   Python exception type raised for any pg_buffer related errors.

.. c:function:: PyObject* pgBuffer_AsArrayInterface(Py_buffer *view_p)

   Return a Python array interface object representation of buffer *view_p*.
   On failure raise a Python exception and return *NULL*.

.. c:function:: PyObject* pgBuffer_AsArrayStruct(Py_buffer *view_p)

   Return a Python array struct object representation of buffer *view_p*.
   On failure raise a Python exception and return *NULL*.

.. c:function:: int pgObject_GetBuffer(PyObject *obj, pg_buffer *pg_view_p, int flags)

   Request a buffer for object *obj*.
   Argument *flags* are PyBUF options.
   Return the buffer description in *pg_view_p*.
   An object may support the Python buffer interface, the NumPy array interface,
   or the NumPy array struct interface.
   Return ``0`` on success, raise a Python exception and return ``-1`` on failure.

.. c:function:: void pgBuffer_Release(Pg_buffer *pg_view_p)

   Release the Pygame *pg_view_p* buffer.

.. c:function:: int pgDict_AsBuffer(Pg_buffer *pg_view_p, PyObject *dict, int flags)

   Write the array interface dictionary buffer description *dict* into a Pygame
   buffer description struct *pg_view_p*.
   The *flags* PyBUF options describe the view type requested.
   Return ``0`` on success, or raise a Python exception and return ``-1`` on failure.

.. c:function:: void import_pygame_base()

   Import the pygame.base module C API into an extension module.
   On failure raise a Python exception.

.. c:function:: SDL_Window* pg_GetDefaultWindow(void)

   Return the Pygame default SDL window created by a
   pygame.display.set_mode() call, or *NULL*.

.. c:function:: void pg_SetDefaultWindow(SDL_Window *win)

   Replace the Pygame default window with *win*.
   The previous window, if any, is destroyed.
   Argument *win* may be *NULL*.
   This function is called by pygame.display.set_mode().

.. c:function:: pgSurfaceObject* pg_GetDefaultWindowSurface(void)

   Return a borrowed reference to the Pygame default window display surface,
   or *NULL* if no default window is open.

.. c:function:: void pg_SetDefaultWindowSurface(pgSurfaceObject *screen)

   Replace the Pygame default display surface with object *screen*.
   The previous surface object, if any, is invalidated.
   Argument *screen* may be *NULL*.
   This functions is called by pygame.display.set_mode().
