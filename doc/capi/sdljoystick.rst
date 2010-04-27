====================
pygame2.sdl.joystick
====================

The :mod:`pygame2.sdl.joystick` C API contains objects and functions for
accessing the physical joystick input devices of the computer.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_joystick (void)

  Imports the :mod:`pygame2.sdl.joystick` module. This returns 0 on success
  and -1 on failure.

Macros
------
.. cmacro:: ASSERT_JOYSTICK_INIT (retval)

  Checks, whether the joystick subsystem was properly initialised. If
  not, this will set a :exc:`PyExc_PyGameError` and return *retval*.

.. cmacro:: ASSERT_JOYSTICK_OPEN (obj, retval)

  Checks, whether the passed :ctype:`PyJoystick` is open for access. If
  not, this will set a :exc:`PyExc_PyGameError` and return *retval*.

PyJoystick
----------
.. ctype:: PyJoystick
.. ctype:: PyJoystick_Type

The PyJoystick object is used for gaining access to and retrieving information
from joystick input devices using SDL.

Members
^^^^^^^
.. cmember:: Uint8 PyJoystick.index

  The joystick device index as reported by the SDL library.

.. cmember:: SDL_Joystick* PyJoystick.joystick

  The SDL_Joystick pointer to access the joystick device.

Functions
^^^^^^^^^^
.. cfunction:: int PyJoystick_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyJoystick` or a subclass of
  :ctype:`PyJoystick`.

.. cfunction:: PyObject* PyJoystick_New (int index)

  Creates a new :ctype:`PyJoystick` object from the passed joystick device
  index. On failure, this returns NULL.

.. cfunction:: SDL_Joystick* PyJoystick_AsJoystick (PyObject *obj)

  Macro for accessing the *joystick* member of the :ctype:`PyJoystick`. This
  does not perform any type or argument checks.
