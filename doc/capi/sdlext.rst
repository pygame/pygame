===================
pygame2.sdlext.base
===================

The :mod:`pygame2.sdlext.base` C API contains objects and functions for
accessing and manipulating :mod:`pygame2.sdl` objects in a specialised
manner.

Import
------
Include headers::

  pygame2/pgsdlext.h

.. cfunction:: int import_pygame2_sdlext_base (void)

  Imports the :mod:`pygame2.sdlext.base` module. This returns 0 on
  success and -1 on failure.

PyPixelArray
------------
.. ctype:: PyPixelArray
.. ctype:: PyPixelArray_Type

The PyPixelArray object is used manipulating the design of the visible
mouse cursor.

Members
^^^^^^^
.. cmember:: PyObject* PyPixelArray.surface

  The :ctype:`PySDLSurface` referenced by the :ctype:`PyPixelArray`.

.. cmember:: PyObject* PyPixelArray.parent

  The parent :ctype:`PyPixelArray`, if any.

.. cmember:: Uint32 PyPixelArray.xstart

  The X start offset for a subarray. For an initial :ctype:`PyPixelArray` this
  will be 0.

.. cmember:: Uint32 PyPixelArray.ystart

  The Y start offset for a subarray. For an initial :ctype:`PyPixelArray` this
  will be 0.

.. cmember:: Uint32 PyPixelArray.xlen

  The X segment length in pixels. For an initial :ctype:`PixelArray` this will
  be the width of the :ctype:`PySDLSurface`.

.. cmember:: Uint32 PyPixelArray.ylen

  The Y segment length in pixels. For an initial :ctype:`PixelArray` this will
  be the height of the :ctype:`PySDLSurface`.

.. cmember:: Sint32 PyPixelArray.xstep

  The step width in the X direction to reach the next accessible pixel. For an
  initial :ctype:`PixelArray` this will be 1.
  
.. cmember:: Sint32 PyPixelArray.xstep

  The step width in the Y direction to reach the next accessible row. For an
  initial :ctype:`PixelArray` this will be 1.

.. cmember:: Uint32 PyPixelArray.padding

  The overall padding in X direction to reach the next row. As the pixel buffer
  of the :ctype:`PySDLSurface` is a 1D array, the :cmember:`padding` denotes
  the overall length in bytes to reach the next row of pixels. This is usually
  the same as the pitch of the :ctype:`PySDLSurface`.

Functions
^^^^^^^^^^
.. cfunction:: int PyPixelArray_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyPixelArray` or a subclass of
  :ctype:`PyPixelArray`.

.. cfunction:: PyObject* PyPixelArray_New (PyObject *obj)

  Creates a new :ctype:`PyPixelArray` object from the passed
  :ctype:`PySDLSurface`. On failure, this returns NULL.
