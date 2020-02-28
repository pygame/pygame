.. include:: ../common.txt

.. highlight:: c

************************************
  API exported by pygame._freetype
************************************

src_c/_freetype.c
=================

This extension module defines Python type :py:class:`pygame.freetype.Font`.

Header file: src_c/include/pygame_freetype.h


.. c:type:: pgFontObject

   The :py:class:`pygame.freetype.Font` instance C struct.

.. c:type:: pgFont_Type

   The :py:class:`pygame.freetype.Font` Python type.

.. c:function:: PyObject* pgFont_New(const char *filename, long font_index)

   Open the font file with path *filename* and return a new
   new :py:class:`pygame.freetype.Font` instance for that font.
   Set *font_index* to ``0`` unless the file contains multiple, indexed, fonts.
   On error raise a Python exception and return ``NULL``.

.. c:function:: int pgFont_Check(PyObject *x)

   Return true if *x* is a :py:class:`pygame.freetype.Font` instance.
   Will return false for a subclass of :py:class:`Font`.
   This is a macro. No check is made that *x* is not ``NULL``.

.. c:function:: int pgFont_IS_ALIVE(PyObject *o)

   Return true if :py:class:`pygame.freetype.Font` object ``o``
   is an open font file.
   This is a macro. No check is made that *o* is not ``NULL``
   or not a :py:class:`Font` instance.
