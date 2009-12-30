================
pygame2.freetype
================

The :mod:`pygame2.freetype` C API contains some objects and functions for
high-quality font, glyph and text operations.

Import
------
Include headers::

  pygame2/pgfreetype.h

.. cfunction:: int import_pygame2_freetype (void)

  Imports the :mod:`pygame2.freetype` module. This returns 0 on success and
  -1 on failure.

Basic Types
-----------

.. ctype:: FontId

  A simple font face information structure.

  .. cmember:: int FontId.face_index
  
    The index number of the font face within the font.

  .. cmember:: FT_Open_Args FontId.open_args

    The arguments used to open the face.

PyFreeTypeFont
--------------

.. ctype:: PyFreeTypeFont
.. ctype:: PyFreeTypeFont_Type

The PyFreeTypeFont object is suitable for creating and managing fonts, glyph
and text operations and text rendering.

Members
^^^^^^^
.. cmember:: PyFont PyFreeTypeFont.pyfont

  The parent :ctype:`PyFont` class the PyFreeTypeFont inherits from.

.. cmember:: FontId PyFreeTypeFont.id

  The used font face information.

.. cmember:: FT_Int16 PyFreeTypeFont.ptsize

  The default font size (height) in points.

.. cmember:: FT_Byte PyFreeTypeFont.style

  The default font style to apply.
  
.. cmember:: FT_Byte PyFreeTypeFont.vertical

  Indicates, whether operations should use a vertical alignment.

.. cmember:: FT_Byte PyFreeTypeFont.antialias

  Indicates, whether operations should use antialiasing.

Functions
^^^^^^^^^

.. cfunction:: PyFont* PyFreeTypeFont_AsFont (PyObject *obj)

  Macro for accessing the *pyfont* member of the :ctype:`PyFreeTypeFont`.
  
  This does not perform any type or argument checks.

.. cfunction:: int PyFreeTypeFont_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyFreeTypeFont` or a subclass of
  :ctype:`PyFreeTypeFont`.

.. cfunction:: PyObject* PyFreeTypeFont_New (const char *font, int ptsize)

  Creates a new :ctype:`PyFreeTypeFont` object for the given *font* and
  default point size *ptsize*. On failure, this returns NULL.
