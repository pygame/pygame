=================
pygame2.sdl.mouse
=================

The :mod:`pygame2.sdl.mouse` C API contains objects and functions for
accessing and manipulating the mouse input device and cursor design.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_mouse (void)

  Imports the :mod:`pygame2.sdl.mouse` module. This returns 0 on success
  and -1 on failure.

PyCursor
--------
.. ctype:: PyCursor
.. ctype:: PyCursor_Type

The PyCursor object is used manipulating the design of the visible mouse cursor.

Members
^^^^^^^
.. cmember:: SDL_Cursor* PyCursor.cursor

  The SDL_Cursor pointer to access the mouse cursor design.

Functions
^^^^^^^^^^
.. cfunction:: int PyCursor_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyCursor` or a subclass of
  :ctype:`PyCursor`.

.. cfunction:: SDL_Cursor* PyCursor_AsCursor (PyObject *obj)

  Macro for accessing the *cursor* member of the :ctype:`PyCursor`. This does
  not perform any type or argument checks.
