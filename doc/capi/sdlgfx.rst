===================
pygame2.sdlgfx.base
===================

The :mod:`pygame2.sdlgfx.base` C API contains objects for managing
constant frame rates and update steps for SDL based screen surfaces.

Import
------
Include headers::

  pygame2/pgsdlgfx.h

.. cfunction:: int import_pygame2_sdlgfx_base (void)

  Imports the :mod:`pygame2.sdlgfx.base` module. This returns 0 on
  success and -1 on failure.

PyFPSManager
------------
.. ctype:: PyFPSManager
.. ctype:: PyFPSManager_Type

The PyFPSManager object features a high resolution timer support for
accurate and fixed frame rate support using SDL.

Members
^^^^^^^
.. cmember:: FPSmanager* PyFPSManager.fps

  The FPSmanager pointer to access the underlying fps management object.

Functions
^^^^^^^^^^
.. cfunction:: int PyFPSManager_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyFPSManager` or a subclass of
  :ctype:`PyFPSManager`.

.. cfunction:: PyObject* PyFPSManager_New (void)

  Creates a new :ctype:`PyFPSManager` object. On failure, this returns
  NULL.

.. cfunction:: FPSmanager* PyFPSManager_AsFPSmanager (PyObject *obj)

  Macro for accessing the *fps* member of the :ctype:`PyFPSManager`. This
  does not perform any type or argument checks.
