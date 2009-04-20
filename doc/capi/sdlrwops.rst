=======================
pygame2.sdl.rwops C API
=======================

The :mod:`pygame2.sdl.rwops` C API contains objects for
accessing :ctype:`SDL_RWops` from Python objects.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_rwops (void)

  Imports the :mod:`pygame2.sdl.rwops` module. This returns 0 on success
  and -1 on failure.

Functions
---------
.. cfunction:: SDL_RWops* RWopsFromPython (PyObject *obj)

  Creates a :ctype:`SDL_RWops` object from the passed Python object.
  *obj* must be some file-like or buffer-like object supporting the binary read,
  write, seek, tell and close methods to be fully usable as :ctype:`SDL_RWops`.
  On failure, this returns NULL.

.. cfunction:: int RWopsCheckPython (SDL_RWops *rw)

  Checks, whether the passed :ctype:`SDL_RWops` was created from a Python
  object. This returns 1 for a Python object, 0 if it is not a Python object and
  -1 on failure.

.. cfunction:: SDL_RWops* RWopsFromPythonThreaded (PyObject *obj)
  
  Creates a :ctype:`SDL_RWops` object with threading support from the passed
  Python object. *obj* must be some file-like or buffer-like object supporting
  the binary read, write, seek, tell and close methods to be fully usable as
  :ctype:`SDL_RWops`. On failure, this returns NULL.

.. cfunction:: int RWopsCheckPythonThreaded (SDL_RWops *rw)

  Checks, whether the passed :ctype:`SDL_RWops` was created from a Python
  object. This returns 1 for a Python object, 0 if it is not a Python object and
  -1 on failure.
