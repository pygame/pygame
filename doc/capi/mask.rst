============
pygame2.mask
============

The :mod:`pygame2.mask` C API contains some objects and functions for
bitmask operations.

Import
------
Include headers::

  pygame2/pgmask.h

.. cfunction:: int import_pygame2_mask (void)

  Imports the :mod:`pygame2.mask` module. This returns 0 on success and
  -1 on failure.

Basic Types
-----------

.. ctype:: bitmask_t

  A simple 2D bitmask structure.

  .. cmember:: int bitmask_t.w
  
    The width of the bitmask_t.

  .. cmember:: int bitmask_t.h

    The height of the bitmask_t.

  .. cmember:: unsigned long int bitmask_t.bits

    The bits of the bitmask_t.

PyMask
------
.. ctype:: PyMask
.. ctype:: PyMask_Type

The PyMask object is suitable for fast pixel-perfect overlapping checks.

Members
^^^^^^^
.. cmember:: bitmask_t PyMask.mask

  The underlying 2D bitmask_t structure.

Functions
^^^^^^^^^

.. cfunction:: int PyMask_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyMask` or a subclass of
  :ctype:`PyMask`.

.. cfunction:: bitmask_t* PyMask_AsBitmask (PyObject *obj)

  Macro for accessing the *mask* member of the :ctype:`PyMask`.

  This does not perform any type or argument checks.

.. cfunction:: PyObject* PyMask_New (int width, int height)

  Creates a new :ctype:`PyMask` object for the given *width* and
  *height*. On failure, this returns NULL.
