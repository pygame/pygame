:mod:`pygame2.sdl.constants` -- Constants for SDL
=================================================

This module contains the constants used throughout the :mod:`pygame2.sdl`
modules.

.. module:: pygame2.sdl.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdl` modules.

Blending Constants
------------------

Those constants are used by the :meth:`pygame2.sdl.video.Surface.blit`
and :meth:`pygame2.sdl.video.Surface.fill` methods.

.. data:: BLEND_RGB_ADD

   Used for an additive blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_SUB

   Used for an subtractive blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_MULT

   Used for an multiply blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_AND

   Used for a binary AND'd blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_OR

   Used for a binary OR'd  blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_XOR

   Used for a binary XOR'd blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_MIN

   Used for a minimum blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_MAX

   Used for a maximum blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_AVG

   Used for an average blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_DIFF

   Used for a difference blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGB_SCREEN

   Used for a screen blend, ignoring the per-pixel alpha value.

.. data:: BLEND_RGBA_ADD

   Used for an additive blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_SUB

   Used for an subtractive blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_MULT

   Used for an multiply blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_MIN

   Used for a minimum blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_MAX

   Used for a maximum blend, with the per-pixel alpha value.

Event Constants
---------------

Those constants are used by the :mod:`pygame2.sdl.event` module
functions.

.. data:: ACTIVEEVENT
   
   Raised, when the SDL application state changes.

.. data:: KEYDOWN

   Raised, when a key is pressed down.

.. data:: KEYUP

   Raised, when a key is released.

.. data:: MOUSEMOTION

   Raised, when the mouse moves.

.. data:: MOUSEBUTTONDOWN

   Raised, when a mouse button is pressed down.

.. data:: MOUSEBUTTONUP
   
   Raised, when a mouse button is released.

.. data:: JOYAXISMOTION

   Raised, when a joystick axis moves.

.. data:: JOYBALLMOTION

   Raised, when a trackball on a joystick moves.

.. data:: JOYHATMOTION

   Raised, when a hat on a joystick moves.

.. data:: JOYBUTTONDOWN

   Raised, when a joystick button is pressed down.

.. data:: JOYBUTTONUP

   Raised, when a joystick button is released.

.. data:: QUIT

   Raised, when the SDL application window shall be closed.

.. data:: SYSWMEVENT

   Raised, when an unknown, window manager specific event occurs.

.. data:: VIDEORESIZE

   Raised, when the SDL application window shall be resized.

.. data:: VIDEOEXPOSE

   Raised, when the screen has been modified outside of the SDL
   application and the SDL application window needs to be redrawn.

.. data:: USEREVENT

   Raised, when a user-specific event occurs.


Application Constants
---------------------

Those constants are used by the :data:`ACTIVEEVENT` event and the
:func:`pygame2.sdl.event.get_app_state` method.

.. data:: APPACTIVE

   Indicates that that the SDL application is currently active.

.. data:: APPINPUTFOCUS

   Indicates that the SDL application has the keyboard input focus.

.. data:: APPMOUSEFOCUS

   Indicates that the SDL application has the mouse input focus.
