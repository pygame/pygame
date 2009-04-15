==================
pygame2.base C API
==================

The :mod:`pygame2.base` C API contains fundamental core objects and function
used throughout nearly all :mod:`pygame2` modules. As such, it should be
considered to be imported in the first place in all C extensions for
:mod:`pygame2`.

Import
------
.. cfunction:: int import_pygame2_base (void)

  Imports the :mod:`pygame2.base` module. This returns 0 on success and -1 on
  failure.

Basic Types
-----------
For platform and system environment compatibility, :mod:`pygame2.base` defines
a small set of own types to be used for most of its functions.

.. ctype:: pgbyte

  An unsigned 8-bit integer type. It is guaranteed to have at least 8 bits.

.. ctype:: pguint16
           pgint16

  Signed and unsigned 16-bit integer types. They are guaranteed to have at
  least 16 bits.

.. ctype:: pguint32
           pgint32
  
  Signed and unsigned 32-bit integer types. They are guaranteed to have at
  least 32 bits.

.. ctype:: CRect

  A simple structure defining a rectangular area. It carries the following
  members:

  .. cmember:: pgint16 CRect.x

    The topleft x coordinate of the CRect.

  .. cmember:: pgint16 CRect.y

    The topleft y coordinate of the CRect.

  .. cmember:: pguint16 CRect.w

    The width of the CRect.

  .. cmember:: pguint16 CRect.h

    The height of the CRect.

Macros
------
In addition to the types, a set of helpful macros was established, which are
used heavily and should be relied on wherever possible.

.. cfunction:: MIN(x,y)

  Gets the smaller of two values. The own implementation will only be used,
  if no system-specific one was found.

.. cfunction:: MAX(x,y)

  Gets the larger of two values. The own implementation will only be used,
  if no system-specific one was found.

.. cfunction:: ABS(x)

  Gets the absolute value. The own implementation will be only used,
  if no system-specific one was found.

.. cfunction:: trunc(x)

  Truncates a floating point value. The own implementation will only be used,
  if no system-specific one was found.

.. cfunction:: round(x)

  Rounds a floating point value to the nearest integer. The own implementation
  will only be used, if no system-specific one was found.

.. cmacro:: M_PI

  The pi constant with 31 digits. The own definition will only be used, if no
  system-specific one was found.

.. cfunction:: DEG2RAD(x)

  Converts degrees to radians. The own implementation will only be used, if no
  system-specific one was found.

.. cfunction:: RAD2DEG(x)

  Converts radians to degrees. The own implementation will only be used, if no
  system-specific one was found.

.. cfunction:: ADD_LIMIT(x,y,lower,upper)
               SUB_LIMIT(x,y,lower,upper)

   Adds and subtracts two values, but guarantees that the result will not be 
   smaller or larger than the *lower* and *upper* limits.

.. cfunction:: INT_ADD_LIMIT(x,y)
               INT_SUB_LIMIT(x,y)
               INT16_ADD_LIMIT(x,y)
               INT16_SUB_LIMIT(x,y)

   Adds and subtracts two integer values, but guarantees that the result will
   not be smaller or larger than the INT_MIN and INT_MAX limits.

.. cfunction:: UINT_ADD_LIMIT(x,y)
               UINT_SUB_LIMIT(x,y)
               UINT16_ADD_LIMIT(x,y)
               UINT16_SUB_LIMIT(x,y)

   Adds and subtracts two unsigned integer values, but guarantees that the
   result will not be smaller or larger than zero and UINT_MAX.

.. cfunction:: LONG_ADD_LIMIT(x,y)
               LONG_SUB_LIMIT(x,y)
               INT32_ADD_LIMIT(x,y)
               INT32_SUB_LIMIT(x,y)

   Adds and subtracts two long integer values, but guarantees that the result
   will not be smaller or larger than the LONG_MIN and LONG_MAX limits.

.. cfunction:: ULONG_ADD_LIMIT(x,y)
               ULONG_SUB_LIMIT(x,y)
               UINT32_ADD_LIMIT(x,y)
               UINT32_SUB_LIMIT(x,y)

   Adds and subtracts two unsigned long integer values, but guarantees that the
   result will not be smaller or larger than zero and ULONG_MAX.

.. cfunction:: DBL_ADD_LIMIT(x,y)
               DBL_SUB_LIMIT(x,y)

   Adds and subtracts two floating point values, but guarantees that the result
   will not be smaller or larger than the DBL_MIN and DBL_MAX limits.

.. cfunction:: INT_ADD_UINT_LIMIT(x,y,z)
               INT_SUB_UINT_LIMIT(x,y,z)
               INT16_ADD_UINT16_LIMIT(x,y,z)
               INT16_SUB_UINT16_LIMIT(x,y,z)

    Adds and subtracts an unsigned integer *y* to an integer *x* and stores the
    result in the integer *z*. If the operation will exceed the INT_MIN and
    INT_MAX limits, *z* will be set to INT_MIN or INT_MAX.

Errors
------

.. cvar:: PyObject* PyExc_PyGameError

  The internally used :class:`pygame2.base.Error` exception class.

Functions
---------

.. cfunction:: int DoubleFromObj (PyObject* obj, double *val)

  Tries to convert the PyObject to a double and stores the result in *val*, if
  successful. If it does not succeed, 0 will be returned and an exception be
  set, otherwise it will return 1.

.. cfunction:: int IntFromObj (PyObject* obj, int *val)

  Tries to convert the PyObject to an int and stores the result in *val*, if
  successful. If it does not succeed, 0 will be returned and an exception be
  set, otherwise it will return 1.

.. cfunction:: int UintFromObj (PyObject* obj, unsigned int *val)

  Tries to convert the PyObject to an unsigned int and stores the result in
  *val*, if successful. If it does not succeed, 0 will be returned and an
  exception be set, otherwise it will return 1.

.. cfunction:: int DoubleFromSeqIndex (PyObject *seq, Py_ssize_t index, double *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to a double, which will be stored in *val*. If it does not
  succeed, 0 will be returned and an exception be set, otherwise it will return
  1.

.. cfunction:: int IntFromSeqIndex (PyObject *seq, Py_ssize_t index, int *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to an int, which will be stored in *val*. If it does not
  succeed, 0 will be returned and an exception be set, otherwise it will return
  1.

.. cfunction:: int UintFromSeqIndex (PyObject *seq, Py_ssize_t index, unsigned int *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to an unsigned int, which will be stored in *val*. If it does
  not succeed, 0 will be returned and an exception be set, otherwise it will
  return 1.

.. cfunction:: int PointFromObject (PyObject *obj, int *x, int *y)

  Tries to get two int values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the topleft x and y values are taken,
  if the object is a sequence type, the first two items are used.  If it does
  not succeed, 0 will be returned and an exception be set, otherwise it will
  return 1.

.. cfunction:: int SizeFromObject (PyObject *obj, pgint32 *x, pgint32 *y)

  Tries to get two pgint32 values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the width and height values are taken,
  if the object is a sequence type, the first two items are used. If it does
  not succeed, 0 will be returned and an exception be set, otherwise it will
  return 1.

.. cfunction:: int FPointFromObject (PyObject *obj, double *x, double *y)

  Tries to get two double values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the topleft x and y values are taken,
  if the object is a sequence type, the first two items are used.  If it does
  not succeed, 0 will be returned and an exception be set, otherwise it will
  return 1.

.. cfunction:: int FSizeFromObject (PyObject *obj, double *x, double *y)

  Tries to get two double values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the width and height values are taken,
  if the object is a sequence type, the first two items are used. If it does
  not succeed, 0 will be returned and an exception be set, otherwise it will
  return 1.

.. cfunction:: int ASCIIFromObject (PyObject *obj, char** text, PyObject **convobj)

  Tries to get ASCII text from the passed object and stores the result in
  *text*. If the object has to be converted, the conversion result will be
  stored in *convobj* and needs to be freed by the caller, once *text* is not
  required anymore.

.. cfunction:: int UTF8FromObject (PyObject *obj, char** text, PyObject **convobj)

  Tries to get UTF-8 encoded text from the passed object and stores the result
  in *text*. If the object has to be converted, the conversion result will be
  stored in *convobj* and needs to be freed by the caller, once *text* is not
  required anymore.

Python Object Types
-------------------

.. toctree::

  baseobjects.rst