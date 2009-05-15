=================
pygame2.sdl.rwops
=================

The :mod:`pygame2.sdl.rwops` C API contains objects for
accessing :ctype:`SDL_RWops` from Python objects.

.. note::

  All Python objects to be used by the :mod:`pygame2.sdl.rwops` API
  *must* support binary read and write access. This is especially
  important for Python 3.x users.
 
Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_rwops (void)

  Imports the :mod:`pygame2.sdl.rwops` module. This returns 0 on success
  and -1 on failure.

Functions
---------
.. cfunction:: SDL_RWops* PyRWops_NewRO (PyObject *obj, int *canautoclose)

  Creates a read-only :ctype:`SDL_RWops` object from the passed Python object.
  *obj* must be some file-like or buffer-like object supporting the binary read,
  seek, tell and close methods to be fully usable as :ctype:`SDL_RWops`.
  *canautoclose* indicates, whether the object can be automatically closed by
  the matching RW function. If not, a manual call to :cfunc:`PyRWops_Close` will
  be required.
  On failure, this returns *NULL*.
  
  .. note::

    This function is not suitable for threaded usage.

.. cfunction:: SDL_RWops* PyRWops_NewRW (PyObject *obj, int *canautoclose)

  Creates a read-write :ctype:`SDL_RWops` object from the passed Python object.
  *obj* must be some file-like or buffer-like object supporting the binary read,
  seek, tell and close methods to be fully usable as :ctype:`SDL_RWops`.
  *canautoclose* indicates, whether the object can be automatically closed by
  the matching RW function. If not, a manual call to :cfunc:`PyRWops_Close` will
  be required.
  On failure, this returns *NULL*.
  
  .. note::

    This function is not suitable for threaded usage.

.. cfunction:: SDL_RWops* PyRWops_NewRO_Threaded (PyObject *obj, int *canautoclose)

  Creates a read-only :ctype:`SDL_RWops` object from the passed Python object.
  *obj* must be some file-like or buffer-like object supporting the binary read,
  seek, tell and close methods to be fully usable as :ctype:`SDL_RWops`.
  *canautoclose* indicates, whether the object can be automatically closed by
  the matching RW function. If not, a manual call to :cfunc:`PyRWops_Close` will
  be required.
  On failure, this returns *NULL*.
  
  .. note::
  
    If Python was built without thread support, this will default to
    :cfunc:`PyRWops_NewRO`.

.. cfunction:: SDL_RWops* PyRWops_NewRW_Threaded (PyObject *obj, int *canautoclose)

  Creates a read-write :ctype:`SDL_RWops` object from the passed Python object.
  *obj* must be some file-like or buffer-like object supporting the binary read,
  seek, tell and close methods to be fully usable as :ctype:`SDL_RWops`.
  *canautoclose* indicates, whether the object can be automatically closed by
  the matching RW function. If not, a manual call to :cfunc:`PyRWops_Close` will
  be required.
  On failure, this returns *NULL*.
  
  .. note::
  
    If Python was built without thread support, this will default to
    :cfunc:`PyRWops_NewRO`.

.. cfunction:: void PyRWops_Close (SDL_RWops *rw, int autoclose)

  Closes a :ctype:`SDL_RWops` object. if *autoclose* is not 0, the bound data
  source will be closed, too (if it is a Python object). Otherwise it will be
  kept open.
 
