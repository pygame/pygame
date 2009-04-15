PyColor
-------
.. ctype:: PyColor
.. ctype:: PyColor_Type

The PyColor object is suitable for storing RGBA color values that feature a
8-bit resolution range for each channel (allowing it to represent a 24/32-bit
color depth).

Members
^^^^^^^
.. cmember:: pgbyte r

  The red color part value.

.. cmember:: pgbyte g

  The green color part value.

.. cmember:: pgbyte b

  The blue color part value.

.. cmember:: pgbyte a

  The alpha transparency value.

Functions
^^^^^^^^^
.. cfunction:: int PyColor_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyColor` or a subclass of
  :ctype:`PyColor`.

.. cfunction:: PyObject* PyColor_New (pgbyte[] rgba)

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
  If the passed *color* is not a :ctype:`PyColor` object, 0 will be returned
  and a TypeError set.

PyRect
------
.. ctype:: PyRect
.. ctype:: PyRect_Type

The PyRect object defines a rectangular area for arbitrary usage. It features
the most typical operations, but is - due to its integer resolution - limited
in some usage scenarios.

Members
^^^^^^^
.. cmember:: pgint16 x

  The topleft x coordinate of the PyRect.

.. cmember:: pgint16 y

  The topleft y coordinate of the PyRect.

.. cmember:: pguint16 w

  The width of the PyRect.

.. cmember:: pguint16 h

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
.. cmember:: double x

  The topleft x coordinate of the PyFRect.

.. cmember:: double y

  The topleft y coordinate of the PyFRect.

.. cmember:: double w

  The width of the PyFRect.

.. cmember:: double h

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
.. cmember:: void* buffer

  A pointer to the underlying C buffer contents.
  
.. cmember:: Py_ssize_t length

  The length of the buffer in bytes
  
.. cmember:: bufferunlock_func unlock_func

  The unlock function callback hook. bufferunlock_func is defined as::
  
    int (*bufferunlock_func)(PyObject* object, PyObject* buffer)
  
Functions
^^^^^^^^^
.. cfunction:: int PyBufferProxy_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyBufferProxy` or a subclass of
  :ctype:`PyBufferProxy`.

.. cfunction:: void* PyBufferProxy_AsBuffer (PyObject *obj)
  
  Macro for accessing the *buffer* member of the :ctype:`PyBufferProxy`.
  This does not perform any type checks.

.. cfunction:: PyObject* PyBufferProxy_New (PyObject *object, void *buffer, Py_ssize_t length, bufferunlock_func func)

  Creates a new :ctype:`PyBufferProxy` object from the passed PyObject.
  *buffer* must be the buffer to refer to for read and write operations,
  *length* the maximum length in bytes that is safe to write to the buffer.
  *func* is the unlock func to release any pending locks and references on the
  buffered object. On failure, this returns NULL.

PySurface
---------
.. ctype:: PySurface
.. ctype:: PySurface_type

The PySurface object is some sort of abstract base class, to be used by
inheriting classes and other interfaces, so it is guaranteed that surface-like
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