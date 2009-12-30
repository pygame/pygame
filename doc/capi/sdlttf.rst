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


PySDLFont_TTF
-------------
.. ctype:: PySDLFont_TTF
.. ctype:: PySDLFont_TTF_Type

The PySDLFont_TTF object is used for rendering text to a :ctype:`PySDLSurface`.

Members
^^^^^^^
.. cmember:: PyFont PySDLFont_TTF.pyfont

  The parent :ctype:`PyFont` class the PySDLFont_TTF inherits from.

.. cmember:: TTF_Font* PySDLFont_TTF.font

  The TTF_Font pointer to access the underlying font.

Functions
^^^^^^^^^^
.. cfunction:: int PySDLFont_TTF_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PySDLFont_TTF` or a subclass of
  :ctype:`PySDLFont_TTF`.

.. cfunction:: PyObject* PySDLFont_TTF_New (char *filename, int ptsize)

  Creates a new :ctype:`PySDLFont_TTF` object from the passed TrueType font
  file. *ptsize* specifies the font size (height) in points. On failure,
  this returns NULL.

.. cfunction:: TTF_Font* PySDLFont_TTF_AsFont (PyObject *obj)

  Macro for accessing the *font* member of the :ctype:`PyFont`. This
  does not perform any type or argument checks.
