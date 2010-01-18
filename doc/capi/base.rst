============
pygame2.base
============

The :mod:`pygame2.base` C API contains fundamental core objects and
functions used throughout nearly all :mod:`pygame2` modules. As such, it
should be considered to be imported in the first place in all C
extensions for :mod:`pygame2`.

Import
------
Include headers::

  pygame2/pgbase.h

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

    The topleft x coordinate of the :ctype:`CRect`.

  .. cmember:: pgint16 CRect.y

    The topleft y coordinate of the :ctype:`CRect`.

  .. cmember:: pguint16 CRect.w

    The width of the :ctype:`CRect`.

  .. cmember:: pguint16 CRect.h

    The height of the :ctype:`CRect`.

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

.. cfunction:: CLAMP(x,low,high)
  
  Checks, whether *x* is within the boundaries of *low* and *high* and returns
  it. If *x* is not within the boundaries, either *low* or *high* will be
  returned, depending on which of them is larger. The own implementation will
  only be used, if no system-specific one was found.

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
   not be smaller or larger than the *INT_MIN* and *INT_MAX* limits.

.. cfunction:: UINT_ADD_LIMIT(x,y)
               UINT_SUB_LIMIT(x,y)
               UINT16_ADD_LIMIT(x,y)
               UINT16_SUB_LIMIT(x,y)

   Adds and subtracts two unsigned integer values, but guarantees that the
   result will not be smaller or larger than zero and *UINT_MAX*.

.. cfunction:: LONG_ADD_LIMIT(x,y)
               LONG_SUB_LIMIT(x,y)
               INT32_ADD_LIMIT(x,y)
               INT32_SUB_LIMIT(x,y)

   Adds and subtracts two long integer values, but guarantees that the result
   will not be smaller or larger than the *LONG_MIN* and *LONG_MAX* limits.

.. cfunction:: ULONG_ADD_LIMIT(x,y)
               ULONG_SUB_LIMIT(x,y)
               UINT32_ADD_LIMIT(x,y)
               UINT32_SUB_LIMIT(x,y)

   Adds and subtracts two unsigned long integer values, but guarantees that the
   result will not be smaller or larger than zero and *ULONG_MAX*.

.. cfunction:: DBL_ADD_LIMIT(x,y)
               DBL_SUB_LIMIT(x,y)

   Adds and subtracts two floating point values, but guarantees that the result
   will not be smaller or larger than the *DBL_MIN* and *DBL_MAX* limits.

.. cfunction:: INT_ADD_UINT_LIMIT(x,y,z)
               INT_SUB_UINT_LIMIT(x,y,z)
               INT16_ADD_UINT16_LIMIT(x,y,z)
               INT16_SUB_UINT16_LIMIT(x,y,z)

    Adds and subtracts an unsigned integer *y* to an integer *x* and stores the
    result in the integer *z*. If the operation will exceed the *INT_MIN* and
    *INT_MAX* limits, *z* will be set to *INT_MIN or *INT_MAX*.

Errors
------

.. cvar:: PyObject* PyExc_PyGameError

  The internally used :class:`pygame2.base.Error` exception class.

Functions
---------

.. cfunction:: int DoubleFromObj (PyObject* obj, double *val)

  Tries to convert the PyObject to a double and stores the result in *val*, if
  successful. This returns 1 on success and 0 on failure.

.. cfunction:: int IntFromObj (PyObject* obj, int *val)

  Tries to convert the PyObject to an int and stores the result in *val*, if
  successful. This returns 1 on success and 0 on failure.

.. cfunction:: int UintFromObj (PyObject* obj, unsigned int *val)

  Tries to convert the PyObject to an unsigned int and stores the result in
  *val*, if successful. This returns 1 on success and 0 on failure.

.. cfunction:: int DoubleFromSeqIndex (PyObject *seq, Py_ssize_t index, double *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to a double, which will be stored in *val*. This returns 1 on
  success and 0 on failure.

.. cfunction:: int IntFromSeqIndex (PyObject *seq, Py_ssize_t index, int *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to an int, which will be stored in *val*. This returns 1 on
  success and 0 on failure.

.. cfunction:: int UintFromSeqIndex (PyObject *seq, Py_ssize_t index, unsigned int *val)

  Tries to get the item at the desired *index* from the passed sequence object
  and converts it to an unsigned int, which will be stored in *val*. This
  returns 1 on success and 0 on failure.

.. cfunction:: int PointFromObject (PyObject *obj, int *x, int *y)

  Tries to get two int values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the topleft x and y values are taken,
  if the object is a sequence type, the first two items are used. This returns
  1 on success and 0 on failure.

.. cfunction:: int SizeFromObject (PyObject *obj, pgint32 *x, pgint32 *y)

  Tries to get two pgint32 values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the width and height values are taken,
  if the object is a sequence type, the first two items are used. This returns
  1 on success and 0 on failure.

.. cfunction:: int FPointFromObject (PyObject *obj, double *x, double *y)

  Tries to get two double values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the topleft x and y values are taken,
  if the object is a sequence type, the first two items are used. This returns
  1 on success and 0 on failure.

.. cfunction:: int FSizeFromObject (PyObject *obj, double *x, double *y)

  Tries to get two double values from the passed object. If the object is a
  :ctype:`PyRect` or :ctype:`PyFRect`, the width and height values are taken,
  if the object is a sequence type, the first two items are used. This returns
  1 on success and 0 on failure.

.. cfunction:: int ASCIIFromObject (PyObject *obj, char** text, PyObject **convobj)

  Tries to get ASCII text from the passed object and stores the result in
  *text*. If the object has to be converted, the conversion result will be
  stored in *convobj* and needs to be freed by the caller, once *text* is not
  required anymore. This returns 1 on success and 0 on failure.

.. cfunction:: int UTF8FromObject (PyObject *obj, char** text, PyObject **convobj)

  Tries to get UTF-8 encoded text from the passed object and stores the result
  in *text*. If the object has to be converted, the conversion result will be
  stored in *convobj* and needs to be freed by the caller, once *text* is not
  required anymore. This returns 1 on success and 0 on failure.

.. cfunction:: int IsReadableStreamObj (PyObject *obj)

  Checks, whether the passed object supports the most important stream
  operation for reading data, such as ``read``, ``seek`` and ``tell``.
  This returns 1 on success and 0 on failure.

.. cfunction:: int IsWriteableStreamObj (PyObject *obj)

  Checks, whether the passed object supports the most important stream
  operation for writing data, such as ``write``, ``seek`` and ``tell``.
  This returns 1 on success and 0 on failure.

.. cfunction:: int IsReadWriteableStreamObj (PyObject *obj)

  Checks, whether the passed object supports the most important stream
  operation for reading and writing data, such as ``read``, ``write``,
  ``seek`` and ``tell``. This returns 1 on success and 0 on failure.


PyColor
-------
.. ctype:: PyColor
.. ctype:: PyColor_Type

The PyColor object is suitable for storing RGBA color values that feature a
8-bit resolution range for each channel (allowing it to represent a 24/32-bit
color depth).

Members
^^^^^^^
.. cmember:: pgbyte PyColor.r

  The red color part value.

.. cmember:: pgbyte PyColor.g

  The green color part value.

.. cmember:: pgbyte PyColor.b

  The blue color part value.

.. cmember:: pgbyte PyColor.a

  The alpha transparency value.

Functions
^^^^^^^^^
.. cfunction:: int PyColor_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyColor` or a subclass of
  :ctype:`PyColor`.

.. cfunction:: PyObject* PyColor_New (pgbyte rgba[])

  Creates a new :ctype:`PyColor` object from the passed 4-value RGBA array. On
  failure, this returns NULL.

.. cfunction:: PyObject* PyColor_NewFromNumber (pguint32 rgba)

  Creates a new :ctype:`PyColor` object from the passed pguint32. On failure,
  this returns NULL.

.. cfunction:: PyObject* PyColor_NewFromRGBA (pgbyte r, pgbyte g, pgbyte b, pgbyte a)

  Creates a new :ctype:`PyColor` object from the passed four RGBA values. On
  failure, this returns NULL.

.. cfunction:: pguint32 PyColor_AsNumber (PyObject *color)

  Returns the 32-bit ARGB integer representation of the :ctype:`PyColor` object.
  On failure, this returns 0. As 0 might be a valid color, you should check
  for an error explicitly using :cfunc:`PyErr_Occured`.

PyRect
------
.. ctype:: PyRect
.. ctype:: PyRect_Type

The PyRect object defines a rectangular area for arbitrary usage. It features
the most typical operations, but is - due to its integer resolution - limited
in some usage scenarios.

Members
^^^^^^^
.. cmember:: pgint16 PyRect.x

  The topleft x coordinate of the PyRect.

.. cmember:: pgint16 PyRect.y

  The topleft y coordinate of the PyRect.

.. cmember:: pguint16 PyRect.w

  The width of the PyRect.

.. cmember:: pguint16 PyRect.h

  The height of the PyRect.

Functions
^^^^^^^^^
.. cfunction:: int PyRect_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyRect` or a subclass of
  :ctype:`PyRect`.

.. cfunction:: PyObject* PyRect_New (pgint16 x, pgint16 y, pguint16 w, pguint16 h)

  Creates a new :ctype:`PyRect` object from the passed four values. On failure,
  this returns NULL.

PyFRect
-------
.. ctype:: PyFRect
.. ctype:: PyFRect_Type

The PyFRect object defines a rectangular area for arbitrary usage and a high
floating point resolution. It features the most typical operations required by
most applications.

Members
^^^^^^^
.. cmember:: double PyFRect.x

  The topleft x coordinate of the PyFRect.

.. cmember:: double PyFRect.y

  The topleft y coordinate of the PyFRect.

.. cmember:: double PyFRect.w

  The width of the PyFRect.

.. cmember:: double PyFRect.h

  The height of the PyFRect.

Functions
^^^^^^^^^
.. cfunction:: int PyFRect_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyFRect` or a subclass of
  :ctype:`PyFRect`.

.. cfunction:: PyObject* PyFRect_New (double x, double y, double w, double h)

  Creates a new :ctype:`PyFRect` object from the passed four values. On failure,
  this returns NULL.

PyBufferProxy
-------------
.. ctype:: PyBufferProxy
.. ctype:: PyBufferProxy_Type

The PyBufferProxy object is a transparent proxy class for buffer-like access.
It supports the Python 2.x and 3.x buffer APIs, automatic unlock hooks for
the buffer object and read/write access to the buffer contents.

Members
^^^^^^^
.. cmember:: void* PyBufferProxy.buffer

  A pointer to the underlying C buffer contents.

.. cmember:: Py_ssize_t PyBufferProxy.length

  The length of the buffer in bytes

.. cmember:: bufferunlock_func PyBufferProxy.unlock_func

  The unlock function callback hook. bufferunlock_func is defined as::

    int (*bufferunlock_func)(PyObject* object, PyObject* buffer)

Functions
^^^^^^^^^
.. cfunction:: int PyBufferProxy_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyBufferProxy` or a subclass of
  :ctype:`PyBufferProxy`.

.. cfunction:: void* PyBufferProxy_AsBuffer (PyObject *obj)

  Macro for accessing the *buffer* member of the :ctype:`PyBufferProxy`.
  
  This does not perform any type or argument checks.

.. cfunction:: PyObject* PyBufferProxy_New (PyObject *object, void *buffer, Py_ssize_t length, bufferunlock_func func)

  Creates a new :ctype:`PyBufferProxy` object from the passed PyObject.
  *buffer* must be the buffer to refer to for read and write operations,
  *length* the maximum length in bytes that is safe to write to the buffer.
  *func* is the unlock func to release any pending locks and references on the
  buffered object. On failure, this returns NULL.

PyFont
------
.. ctype:: PyFont
.. ctype:: PyFont_Type

The PyFont object an abstract base class, to be used by inheriting classes
and other interfaces, so it is guaranteed that font-like objects contain a
set of same attributes and methods.

Members
^^^^^^^
PyFont only defines a set of function pointer bindings to access and set by
inheriting classes and interfaces. Those are

.. cfunction:: PyObject* (*get_height) (PyObject *self, void *closure)

  Gets the height of the :ctype:`PyFont` instance. *self* is the
  :ctype:`PyFont` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*get_name) (PyObject *self, void *closure)

  Gets the name of the :ctype:`PyFont` instance. *self* is the :ctype:`PyFont`
  itself, the *closure* argument is the same as for the Python C API
  getter definition.

.. cfunction:: PyObject* (*get_style) (PyObject *self, void *closure)

  Gets the currently applied style of the :ctype:`PyFont`
  instance. *self* is the :ctype:`PyFont` itself, the *closure* argument
  is the same as for the Python C API getter definition.

.. cfunction:: int (*set_style) (PyObject *self, PyObject *attr, void *closure)

  Applies a style to the :ctype:`PyFont` instance. *self* is the
  :ctype:`PyFont` itself, *attr* the style to apply, the *closure*
  argument is the same as for the Python C API getter definition.

.. cfunction:: PyObject* (*get_size) (PyObject *self, PyObject *args, PyObject *kwds)

  Gets the size of the :ctype:`PyFont` instance. *self* is the
  :ctype:`PyFont` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*render) (PyObject *self, PyObject *args, PyObject *kwds)

  Renders the :ctype:`PyFont` onto some :ctype:`PySurface` or whatever
  is appropriate for the concrete implementation. *self* is the
  :ctype:`PyFont` itself, the *args* and *kwds* arguments are the same as for
  the Python C API method definition.

.. cfunction:: PyObject* (*copy) (PyObject *self);

  Creates an exact copy of the :ctype:`PyFont`. *self* is the
  :ctype:`PyFont` itself.

Functions
^^^^^^^^^

.. cfunction:: int PyFont_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyFont` or a subclass of
  :ctype:`PyFont`.

.. cfunction:: PyObject* PyFont_New (void)

  Creates a new, empty :ctype:`PyFont` object, which's members are set to
  NULL. On failure, this returns NULL.


PySurface
---------
.. ctype:: PySurface
.. ctype:: PySurface_Type

The PySurface object is an abstract base class, to be used by inheriting
classes and other interfaces, so it is guaranteed that surface-like
objects contain a set of same attributes and methods.

Members
^^^^^^^
PySurface only defines a set of function pointer bindings to access and set by
inheriting classes and interfaces. Those are

.. cfunction:: PyObject* (*get_width) (PyObject *self, void *closure)

  Gets the width of the :ctype:`PySurface` instance. *self* is the
  :ctype:`PySurface` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*get_height) (PyObject *self, void *closure)

  Gets the height of the :ctype:`PySurface` instance. *self* is the
  :ctype:`PySurface` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*get_size) (PyObject *self, void *closure)

  Gets the size of the :ctype:`PySurface` instance. *self* is the
  :ctype:`PySurface` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*get_pixels) (PyObject *self, void *closure)

  Gets the raw pixels of the :ctype:`PySurface` instance. *self* is the
  :ctype:`PySurface` itself, the *closure* argument is the same as for the
  Python C API getter definition.

.. cfunction:: PyObject* (*blit)(PyObject *self, PyObject *args, PyObject *kwds)

  Blits the :ctype:`PySurface` onto some other :ctype:`PySurface` or whatever
  is appropriate for the concrete implementation. *self* is the
  :ctype:`PySurface` itself, the *args* and *kwds* arguments are the same as for
  the Python C API method definition.

.. cfunction:: PyObject* (*copy)(PyObject *self)

  Creates an exact copy of the :ctype:`PySurface`. *self* is the
  :ctype:`PySurface` itself.

Functions
^^^^^^^^^
.. cfunction:: int PySurface_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PySurface` or a subclass of
  :ctype:`PySurface`.

.. cfunction:: PyObject* PySurface_New (void)

  Creates a new, empty :ctype:`PySurface` object, which's members are set to
  NULL. On failure, this returns NULL.

CPyStreamWrapper
----------------
.. ctype:: CPyStreamWrapper

CPyStreamWrapper is a C API only class type for reading and writing
Python stream objects in a threaded or non-threaded manner. It
encapsules the important underlying stream methods.

Members
^^^^^^^
.. cmember:: PyObject* CPyStreamWrapper.read

  The method pointer to the underlying Python object's ``read`` method.
  This will be NULL, if the Python object does not support read access.

.. cmember:: PyObject* CPyStreamWrapper.write

  The method pointer to the underlying Python object's ``write`` method.
  This will be NULL, if the Python object does not support write access.

.. cmember:: PyObject* CPyStreamWrapper.seek

  The method pointer to the underlying Python object's ``seek`` method.
  This will be NULL, if the Python object does not support seeking the
  stream.

.. cmember:: PyObject* CPyStreamWrapper.tell

  The method pointer to the underlying Python object's ``tell`` method.
  This will be NULL, if the Python object does not support seeking the
  stream.

.. cmember:: PyObject* CPyStreamWrapper.close

  The method pointer to the underlying Python object's ``close`` method.
  This will be NULL, if the Python object does not support closing the
  stream.

.. cmember:: PyThreadState* CPyStreamWrapper.thread

  If Python was built with thread support, this will contain the
  preserved Python thread state to allow concurrent external threads
  to access the interpreter state and perform stream operations on the
  underlying Python object.

Functions
^^^^^^^^^
.. cfunction:: CPyStreamWrapper* CPyStreamWrapper_New (PyObject *obj)

  Creates a new :ctype:`CPyStreamWrapper` object encapsuling the passed
  *obj*. This will not perform any checks, whether the *obj* actually is
  a stream-like object and supports all required methods. If it is not a
  stream-like object or does not implement the methods, the according
  :ctype:`CPyStreamWrapper` members will be NULL.
   
  Use :cfunc:`IsReadableStreamObj`, :cfunc:`IsWriteableStreamObj` or
  :cfunc:`IsReadWritebbleStreamObj` beforehand, to check whether *obj*
  implements all wanted methods.

  On failure, this returns NULL.

.. cfunction:: CPyStreamWrapper_Free (CPyStreamWrapper *wrapper)

  Releases all resources hold by the passed :ctype:`CPyStreamWrapper`
  instance.

.. cfunction:: int CPyStreamWrapper_Read_Threaded (CPyStreamWrapper *wrapper, void *buf, pguint32 offset, pguint32 count, pguint32 *read_)

  Reads a maximum of *count* bytes from the passed
  :ctype:`CPyStreamWrapper`, starting at *offset*. The read data will be
  stored in *buf*, which must be large enough to hold the data. The
  amount of bytes actually written to *buf* will be stored in *read_*.
  If *offset* is 0, the stream will not be repositioned. Otherwise,
  *offset* denotes a position relative to the start of the stream.
  Returns 1 on succes and 0 on failure.

  This will swap the Python interpreter thread state to gain access to
  the underlying Python stream object. It **should not** be called from
  within the same interpreter thread, as it locks the interpreter stat
  (and thus itself).

.. cfunction:: int CPyStreamWrapper_Read (CPyStreamWrapper *wrapper, void *buf, pguint32 offset, pguint32 count, pguint32 *read_)

  Same as :cfunc:`CPyStreamWrapper_Read_Threaded`, but this will not
  swap the thread state of the Python interpreter and thus is safe
  to be called from within the interpreter thread.

.. cfunction:: int CPyStreamWrapper_Write_Threaded (CPyStreamWrapper *wrapper, const void *buf, pguint32 num, pguint32 size, pguint32 *written)

  Writes at least *num* elements of size *size* from the passed *buf* to
  the stream of the passed :ctype:`CPyStreamWrapper`. The actual amount
  of written elements will be stored in *written*.

  This will swap the Python interpreter thread state to gain access to
  the underlying Python stream object. It **should not** be called from
  within the same interpreter thread, as it locks the interpreter stat
  (and thus itself).

.. cfunction:: int CPyStreamWrapper_Write (CPyStreamWrapper *wrapper, const void *buf, pguint32 num, pguint32 size, pguint32 *written)

  Same as :cfunc:`CPyStreamWrapper_Write_Threaded`, but this will not
  swap the thread state of the Python interpreter and thus is safe
  to be called from within the interpreter thread.

.. cfunction:: int CPyStreamWrapper_Seek_Threaded (CPyStreamWrapper *wrapper, pgint32 offset, int whence)

  Moves to a new stream position. *offset* is the position in
  bytes. *whence* indicates, how the movement should be performed and
  can be a valid value of

    * SEEK_SET - *offset* is relative to the start of the stream
    * SEEK_CUR - *offset* is relative to the current stream position
    * SEEK_END - *offset* is relative to the end of the stream.

  .. note:: 

    Seeking beyond the end of the stream boundaries might result in an
    undefined behaviour.

  Returns 1 on succes and 0 on failure.

  This will swap the Python interpreter thread state to gain access to
  the underlying Python stream object. It **should not** be called from
  within the same interpreter thread, as it locks the interpreter stat
  (and thus itself).

.. cfunction:: int CPyStreamWrapper_Seek (CPyStreamWrapper *wrapper, pgint32 offset, int whence)

  Same as :cfunc:`CPyStreamWrapper_Seek_Threaded`, but this will not
  swap the thread state of the Python interpreter and thus is safe
  to be called from within the interpreter thread.

.. cfunction:: pgint32 CPyStreamWrapper_Tell_Threaded (CPyStreamWrapper *wrapper)

  Returns the current stream position or -1 if an error occured.

  This will swap the Python interpreter thread state to gain access to
  the underlying Python stream object. It **should not** be called from
  within the same interpreter thread, as it locks the interpreter stat
  (and thus itself).

.. cfunction:: pgint32 CPyStreamWrapper_Tell (CPyStreamWrapper *wrapper)

  Same as :cfunc:`CPyStreamWrapper_Tell_Threaded`, but this will not
  swap the thread state of the Python interpreter and thus is safe
  to be called from within the interpreter thread.

.. cfunction:: int CPyStreamWrapper_Close_Threaded (CPyStreamWrapper *wrapper)

  Closes the underlying stream. This leaves the *wrapper* itself intact.
  Returns 1 on success and 0 on failure.

  This will swap the Python interpreter thread state to gain access to
  the underlying Python stream object. It **should not** be called from
  within the same interpreter thread, as it locks the interpreter stat
  (and thus itself).

.. cfunction:: int CPyStreamWrapper_Close (CPyStreamWrapper *wrapper)

  Same as :cfunc:`CPyStreamWrapper_Close_Threaded`, but this will not
  swap the thread state of the Python interpreter and thus is safe
  to be called from within the interpreter thread.
