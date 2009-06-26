:mod:`pygame2.sdlgfx.constants` -- Constants for SDL_gfx
========================================================

This module contains the constants used throughout the :mod:`pygame2.sdlgfx`
modules.

.. module:: pygame2.sdlgfx.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdlgfx` modules.

FPS Constants
-------------

Those constants denote the FPS limites for the :class:`pygame2.sdlgx.FPSManager`
class.

.. data:: FPS_UPPER_LIMIT
   
   The maximum frames per second allowed.
   
.. data:: FPS_LOWER_LIMIT

   The minimum frames per second allowed.

.. data:: FPS_DEFAULT

   The default frames per second to use.

Rotozoom Constants
------------------

The following constants are used for smoothing surfaces on rotozoom operations
within the :mod:`pygame2.sdlgfx.rotozoom` module.

.. data:: SMOOTHING_OFF
   
   Smoothing will not be used.

.. data:: SMOOTHING_ON

   Smoothing will be used.
