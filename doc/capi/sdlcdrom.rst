=================
pygame2.sdl.cdrom
=================

The :mod:`pygame2.sdl.cdrom` C API contains objects and functions for
accessing the physical CD- and DVD-ROMS of the computer.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_cdrom (void)

  Imports the :mod:`pygame2.sdl.cdrom` module. This returns 0 on success
  and -1 on failure.

Macros
------
.. cmacro:: ASSERT_CDROM_INIT (retval)

  Checks, whether the cdrom subsystem was properly initialised. If not,
  this will set a :exc:`PyExc_PyGameError` and return *retval*.

.. cmacro:: ASSERT_CDROM_OPEN (obj, retval)

  Checks, whether the passed :ctype:`PyCD` is open for access. If not,
  this will set a :exc:`PyExc_PyGameError` and return *retval*.

PyCD
----
.. ctype:: PyCD
.. ctype:: PyCD_Type

The PyCD object is used for gaining access and working with physical CD
drives and CDs using SDL.

Members
^^^^^^^
.. cmember:: int PyCD.index

  The drive index as reported by the SDL library.

.. cmember:: SDL_CD* PyCD.cd

  The SDL_CD pointer to access the CD drive.

Functions
^^^^^^^^^^
.. cfunction:: int PyCD_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyCD` or a subclass of
  :ctype:`PyCD`.

.. cfunction:: PyObject* PyCD_New (int index)

  Creates a new :ctype:`PyCD` object from the passed CD drive index. On
  failure, this returns NULL.

.. cfunction:: SDL_CD* PyCD_AsCD (PyObject *obj)

  Macro for accessing the *cd* member of the :ctype:`PyCD`. This does
  not perform any type or argument checks.

PyCDTrack
---------
.. ctype:: PyCDTrack
.. ctype:: PyCDTrack_Type

The PyCDTrack object contains information about a single CD track on a
loaded CD.

Members
^^^^^^^
.. cmember:: SDL_CDtrack PyCD.track

  The track information of a CD track as reported by the SDL library.

Functions
^^^^^^^^^^
.. cfunction:: int PyCDTrack_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyCDTrack` or a subclass of
  :ctype:`PyCDTrack`.

.. cfunction:: PyObject* PyCDTrack_New (SDL_CDtrack track)

  Creates a new :ctype:`PyCDTrack` object from the passed CD track
  information. On failure, this returns NULL.

.. cfunction:: SDL_CDtrack* PyCDTrack_AsCDTrack (PyObject *obj)

  Macro for accessing the *track* member of the :ctype:`PyCDTrack`. This
  does not perform any type or argument checks.
