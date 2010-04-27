================
pygame2.sdl.base
================

The :mod:`pygame2.sdl.base` C API contains fundamental core functions
used throughout nearly all :mod:`pygame2.sdl` related modules. As such,
it should be considered to be imported in the first place in all C
extensions that require parts of :mod:`pygame2.sdl`.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_base (void)

  Imports the :mod:`pygame2.sdl.base` module. This returns 0 on success
  and -1 on failure.

Macros
------
The following macros are used in places within the :mod:`pygame2.sdl`
and related modules. They are used to check whether certain parts of the
SDL subsystems are properly set and initialised.

.. cmacro:: ASSERT_TIME_INIT (retval)

  Checks, whether the time subsystem was properly initialised. If not,
  this will set a :exc:`PyExc_PyGameError` and return *retval*.

Functions
---------

.. cfunction:: int Uint8FromObj (PyObject *obj, Uint8 *val)

  Tries to convert the PyObject to a Uint8 and stores the result in
  *val*, if successful. This returns 1 on success and 0 on failure.

.. cfunction:: int Uint16FromObj (PyObject *obj, Uint16 *val)

  Tries to convert the PyObject to a Uint16 and stores the result in
  *val*, if successful. This returns 1 on success and 0 on failure.

.. cfunction:: int Sint16FromObj (PyObject *obj, Sint16 *val)

  Tries to convert the PyObject to a Sint16 and stores the result in
  *val*, if successful. This returns 1 on success and 0 on failure.

.. cfunction:: int Uint32FromObj (PyObject *obj, Uint32 *val)

  Tries to convert the PyObject to a Uint32 and stores the result in
  *val*, if successful. This returns 1 on success and 0 on failure.

.. cfunction:: int Uint8FromSeqIndex (PyObject *obj, Py_ssize_t index, Uint8 *val)

  Tries to get the item at the desired *index* from the passed sequence
  object and converts it to a Uint8, which will be stored in *val*. This
  returns 1 on success and 0 on failure.

.. cfunction:: int Uint16FromSeqIndex (PyObject *obj, Py_ssize_t index, Uint16 *val)

  Tries to get the item at the desired *index* from the passed sequence
  object and converts it to a Uint16, which will be stored in *val*. This
  returns 1 on success and 0 on failure.

.. cfunction:: int Sint16FromSeqIndex (PyObject *obj, Py_ssize_t index, Sint16 *val)

  Tries to get the item at the desired *index* from the passed sequence
  object and converts it to a Sint16, which will be stored in *val*. This
  returns 1 on success and 0 on failure.

.. cfunction:: int Uint32FromSeqIndex (PyObject *obj, Py_ssize_t index, Uint32 *val)

  Tries to get the item at the desired *index* from the passed sequence
  object and converts it to a Uint32, which will be stored in *val*. This
  returns 1 on success and 0 on failure.

.. cfunction:: int IsValidRect (PyObject *obj)

  Checks, if the passed object is a valid rectangle object. That is the
  case if, the object is either a :ctype:`PyRect` or :ctype:`PyFRect`
  instance or a 4-value sequence that carries two Sint16-compatible
  values two Uint16-compatible values in the order (Sint16, Sint16,
  Uint16, Uint16). This returns 1 on success and 0 on failure.

.. cfunction:: int SDLRectFromRect (PyObject *obj, SDL_Rect *rect)

   Tries to convert the passed object to a :ctype:`SDL_Rect` and stores
   the result in the passed *rect*'s members. The object must be a valid
   rectangle object (as for :cfunc:`IsValidRect`). This returns 1 on success
   and 0 on failure.
