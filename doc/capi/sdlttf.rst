===================
pygame2.sdlttf.base
===================

The :mod:`pygame2.sdlttf.base` C API contains objects and functions for
accessing TrueType font rendering on SDL surfaces.

Import
------
Include headers::

  pygame2/pgsdlttf.h

.. cfunction:: int import_pygame2_sdlttf_base (void)

  Imports the :mod:`pygame2.sdlttf.base` module. This returns 0 on
  success and -1 on failure.

Macros
------
.. cfunction:: ASSERT_TTF_INIT (retval)

  Checks, whether the ttf subsystem was properly initialised. If
  not, this will set a :exc:`PyExc_PyGameError` and return *retval*.


PyFont
------
.. ctype:: PyFont
.. ctype:: PyFont_Type

The PyFont object is used for rendering text to a :ctype:`PySDLSurface`.

Members
^^^^^^^
.. cmember:: TTF_Font* PyFont.font

  The TTF_Font pointer to access the underlying font.

Functions
^^^^^^^^^^
.. cfunction:: int PyFont_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyFont` or a subclass of
  :ctype:`PyFont`.

.. cfunction:: PyObject* PyFont_New (char *filename, int ptsize)

  Creates a new :ctype:`PyFont` object from the passed TrueType font
  file. *ptsize* specifies the font size (height) in points. On failure,
  this returns NULL.

.. cfunction:: TTF_Font* PyFont_AsFont (PyObject *obj)

  Macro for accessing the *font* member of the :ctype:`PyFont`. This
  does not perform any type or argument checks.
