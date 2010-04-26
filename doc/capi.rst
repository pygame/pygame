.. _capi:

###################
C API Documentation
###################

The following sections cover the external C API of Pygame2. The C API is useful
for implementing your own C extension or if you require some Python to C
interoperatibility.

Notes on Reading
================
All methods and functions covered by this C API usually check their arguments
for the correct value range, types and so on. If they fail due to wrong
arguments or an internal error, they will either set an exception using::

  PyErr_SetString (EXC_TYPE, MESSAGE)

or escalate the exception, if it was set in another place.

Those methods, which do not set an exception on failure or do not check their
arguments, are clearly marked as such.

API Reference
=============

.. toctree::

  capi/base.rst
  capi/freetype.rst
  capi/mask.rst
  capi/math.rst
  capi/openal.rst
  capi/sdlbase.rst
  capi/sdlcdrom.rst
  capi/sdlevent.rst
  capi/sdljoystick.rst
  capi/sdlmouse.rst
  capi/sdlvideo.rst
  capi/sdlrwops.rst
  capi/sdlext.rst
  capi/sdlgfx.rst
  capi/sdlttf.rst
