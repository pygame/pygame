:mod:`pygame2.sdl.constants` -- Constants for SDL
=================================================

This module contains the constants used throughout the :mod:`pygame2.sdl`
modules.

.. module:: pygame2.sdl.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdl` modules.

Initialisation Constants
------------------------

Those constants are used by the :func:`pygame2.sdl.init`,
:func:`pygame2.sdl.init_subsystem` and :func:`pygame2.sdl.quit_subsystem`
functions.

.. data:: INIT_AUDIO

   Initialises the SDL audio subsystem.

.. data:: INIT_CDROM

   Initialises the SDL cdrom subsystem.

.. data:: INIT_TIMER

   Initialises the SDL timer subsystem.

.. data:: INIT_JOYSTICK

   Initialises the SDL joystick subsystem.

.. data:: INIT_VIDEO

   Initialises the SDL video subsystem.

.. data:: INIT_EVERYTHING

   Initialises all parts of the SDL subsystems.

.. data:: INIT_NOPARACHUTE

   Initialises the SDL subsystems without a segmentation fault parachute.

.. data:: INIT_EVENTTHREAD

   Initialises the SDL event subsystem with threading support.

Blending Constants
------------------

Those constants are used by the :meth:`pygame2.sdl.video.Surface.blit`
and :meth:`pygame2.sdl.video.Surface.fill` methods.

If not stated otherwise, each of the modes will work on a per-channel basis,
so that the described operation is performed on each RGB(A) color component.
This means that e.g. BLEND_RGB_ADD performs some operation similar to
(R1 + R1, G1 + G2, B1 + B2).

.. data:: BLEND_RGB_ADD

   Used for an additive blend, ignoring the per-pixel alpha value. The sum of
   the both RGB values will used for the result.

.. data:: BLEND_RGB_SUB

   Used for an subtractive blend, ignoring the per-pixel alpha value. The
   difference of both RGB values will be used for the result. If the difference
   is smaller than 0, it will be set to 0.

.. data:: BLEND_RGB_MULT

   Used for an multiply blend, ignoring the per-pixel alpha value. The
   both RGB values will be multiplied with each other, causing the result
   to be darker.

.. data:: BLEND_RGB_AND

   Used for a binary AND'd blend, ignoring the per-pixel alpha value.
   The bitwise AND combination of both RGB values will be used for the result.

.. data:: BLEND_RGB_OR

   Used for a binary OR'd  blend, ignoring the per-pixel alpha value.
   The bitwise OR combination of both RGB values will be used for the result.

.. data:: BLEND_RGB_XOR

   Used for a binary XOR'd blend, ignoring the per-pixel alpha value.
   The bitwise XOR combination of both RGB values will be used for the result.
   
.. data:: BLEND_RGB_MIN

   Used for a minimum blend, ignoring the per-pixel alpha value. The minimum
   of both RGB values will be used for the result.

.. data:: BLEND_RGB_MAX

   Used for a maximum blend, ignoring the per-pixel alpha value. The maximum
   of both RGB values will be used for the result.

.. data:: BLEND_RGB_AVG

   Used for an average blend, ignoring the per-pixel alpha value. The average
   of an addition of both RGB values will be used for the result.

.. data:: BLEND_RGB_DIFF

   Used for a difference blend, ignoring the per-pixel alpha value. The real
   difference of both RGB values will be used for the result.

.. data:: BLEND_RGB_SCREEN

   Used for a screen blend, ignoring the per-pixel alpha value. The
   inverted multiplication result of the inverted RGB values will be used,
   causing the result to be brighter.

.. data:: BLEND_RGBA_ADD

   Used for an additive blend, with the per-pixel alpha value. The sum of
   the both RGBA values will used for the result.

.. data:: BLEND_RGBA_SUB

   Used for an subtractive blend, with the per-pixel alpha value. The
   difference of both RGBA values will be used for the result. If the difference
   is smaller than 0, it will be set to 0.

.. data:: BLEND_RGBA_MULT

   Used for an multiply blend, with the per-pixel alpha value. The
   both RGBA values will be multiplied with each other, causing the result
   to be darker.

.. data:: BLEND_RGBA_AND

   Used for a binary AND'd blend, with the per-pixel alpha value.
   The bitwise AND combination of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_OR

   Used for a binary OR'd  blend, with the per-pixel alpha value.
   The bitwise OR combination of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_XOR

   Used for a binary XOR'd blend, with the per-pixel alpha value.
   The bitwise XOR combination of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_MIN

   Used for a minimum blend, with the per-pixel alpha value. The minimum
   of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_MAX

   Used for a maximum blend, with the per-pixel alpha value. The maximum
   of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_AVG

   Used for an average blend, with the per-pixel alpha value. The average
   of an addition of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_DIFF

   Used for a difference blend, with the per-pixel alpha value. The real
   difference of both RGBA values will be used for the result.

.. data:: BLEND_RGBA_SCREEN

   Used for a screen blend, with the per-pixel alpha value. The
   inverted multiplication result of the inverted RGBA values will be used,
   causing the result to be brighter.

CD-ROM Constants
----------------

The following constants are used by the :mod:`pygame2.sdl.cdrom` module.

.. data:: MAX_TRACKS

   The maximum amount of tracks to manage on a CD-ROM.

The following constants are used by the :attr:`pygame2.sdl.cdrom.CDTrack.type`
attribute.

.. data:: AUDIO_TRACK

   Indicates an audio track.

.. data:: DATA_TRACK

   Indicates a data track.

The following constants are used by the :attr:`pygame2.sdl.cdrom.CD.status`
attribute:

.. data:: CD_TRAYEMPTY

   Indicates that no CD-ROM is in the tray.

.. data:: CD_STOPPED

   Indicates that the CD playback has been stopped.

.. data:: CD_PLAYING

   Indicates that the CD is currently playing a track.

.. data:: CD_PAUSED

   Indicates that the CD playback has been paused.

.. data:: CD_ERROR

   Indicates an error on accessing the CD.

Event Constants
---------------

Those constants are used by the :mod:`pygame2.sdl.event` module
functions.

.. data:: NOEVENT

   Indicates no event.

.. data:: NUMEVENTS

   The maximum amount of event types allowed to be used.

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

Keyboard Constants
------------------

The following constants are used by the :func:`pygame2.sdl.keyboard.set_repeat`
function:

.. data:: DEFAULT_REPEAT_DELAY

   The default delay before starting to repeat raising :data:`KEYDOWN` event
   on pressing a key down.

.. data:: DEFAULT_REPEAT_INTERVAL

   The default interval for raising :data:`KEYDOWN` events on pressing a key
   down.

The following constants are used by the :func:`pygame2.sdl.keyboard.get_state`
and :func:`pygame2.sdl.keyboard.get_key_name` functions and the :data:`KEYDOWN`
and :data:`KEYUP` events.

+-------------------+-------------------------------------------------------+
| Constant          | Meaning and Value                                     |
+===================+=======================================================+
| K_UNKNOWN         | An unknown key.                                       |
+-------------------+-------------------------------------------------------+
| K_a - K_z         | Alphabetical keys ranging from a to z. There is no    |
|                   | captalised version of them. Instead the keyboard      |
|                   | modifier state can be checked for :data:`KMOD_SHIFT`  |
|                   | being set.                                            |
+-------------------+-------------------------------------------------------+
| K_0 - K_9         | Numerical keys ranging from 0 to 9. Those differ from |
|                   | the numerical keys on the keypad.                     |
+-------------------+-------------------------------------------------------+
| K_TAB, K_SPACE,   | Tabulator, Space, Exclamation Mark, Hash, Double      |
| K_EXCLAIM, K_HASH,| Quote, Dollar sign, Single Quote, Ampersand, Left     |
| K_QUOTEDBL,       | and Right Parenthesis, Asterisk, Plus and Minus,      |
| K_DOLLAR, K_QUOTE,| Comma, Period, Slash and Backslash, Colon and         |
| K_AMPERSAND,      | Semicolon, Question Mark, At sign, Left and Right     |
| K_LEFTPAREN,      | Bracket, Caret, Underscore and Backquote keys.        |
| K_RIGHTPAREN,     |                                                       |
| K_ASTERISK,       |                                                       |
| K_PLUS, K_MINUS,  |                                                       |
| K_COMMA, K_PERIOD,|                                                       |
| K_SLASH,          |                                                       |
| K_BACKSLASH,      |                                                       |
| K_COLON,          |                                                       |
| K_SEMICOLON,      |                                                       |
| K_QUESTION, K_AT, |                                                       |
| K_LEFTBRACKET,    |                                                       |
| K_RIGHTBRACKET,   |                                                       |
| K_CARET,          |                                                       |
| K_UNDERSCORE,     |                                                       |
| K_BACKQUOTE       |                                                       |
+-------------------+-------------------------------------------------------+
| K_LESS, K_GREATER,| Less, Greater and Equality sign keys.                 |
| K_EQUALS          |                                                       |
+-------------------+-------------------------------------------------------+
| K_F1 - K_F15      | Function keys from F1 to F15.                         |
+-------------------+-------------------------------------------------------+
| K_HOME, K_END,    | Home and End, Insert and Delete, PageUp and PageDown  |
| K_INSERT,         | and Backspace keys.                                   |
| K_DELETE,         |                                                       |
| K_PAGEUP,         |                                                       |
| K_PAGEDOWN,       |                                                       |
| K_BACKSPACE       |                                                       |
+-------------------+-------------------------------------------------------+
| K_LEFT, K_RIGHT,  | Cursor keys.                                          |
| K_DOWN, K_UP      |                                                       |
+-------------------+-------------------------------------------------------+
| K_KP0 - K_KP9     | Numerical keys on the keypad, ranging from 0 to 9.    |
+-------------------+-------------------------------------------------------+
| K_KP_PERIOD,      | Period, Divide, Multiply, Plus, Minus, Equal sign and |
| K_KP_DIVIDE,      | the Enter key on the keypad.                          |
| K_KP_MULTIPLY,    |                                                       |
| K_KP_MINUS,       |                                                       |
| K_KP_PLUS,        |                                                       |
| K_KP_EQUALS,      |                                                       |
| K_KP_ENTER        |                                                       |
+-------------------+-------------------------------------------------------+
| K_HELP, K_PRINT,  | Help, Print, SysReq, Break, Menu, Power, Euro sign,   |
| K_SYSREQ, K_BREAK,| First and Last keys.                                  |
| K_MENU, K_POWER,  |                                                       |
| K_EURO, K_FIRST,  |                                                       |
| K_LAST            |                                                       |
+-------------------+-------------------------------------------------------+
| K_ESCAPE, K_PAUSE,| Escape, Pause and Clear keys.                         |
| K_CLEAR           |                                                       |
+-------------------+-------------------------------------------------------+
| K_NUMLOCK,        | NumLock, CapsLock and ScrolLock keys.                 |
| K_CAPSLOCK,       |                                                       |
| K_SCROLLOCK       |                                                       |
+-------------------+-------------------------------------------------------+
| K_RSHIFT,         | Right and Left Shift, Right and Left Control, Right   |
| K_LSHIFT, K_RCTRL,| and Left Alternative, Right and Left Meta, Right and  |
| K_LCTRL, K_RALT,  | Left Super and Mode keys.                             |
| K_LALT, K_RMETA,  |                                                       |
| K_LMETA, K_LSUPER,|                                                       |
| K_RSUPER, K_MODE  |                                                       |
+-------------------+-------------------------------------------------------+

The following constants are keyboard modifer states, used as bitwise
combinations to check, whether they were hold down on keyboard
input. They are used by the :func:`pygame2.sdl.keyboard.get_mod_state` and
:func:`pygame2.sdl.keyboard.set_mod_state` functions.

+-------------------+-------------------------------------------------------+
| Constant          | Meaning and Value                                     |
+===================+=======================================================+
| KMOD_NONE         | No modifier key was pressed.                          |
+-------------------+-------------------------------------------------------+
| KMOD_LSHIFT,      | Left Shift, Right Shift or one of both was pressed.   |
| KMOD_RSHIFT,      |                                                       |
| KMOD_SHIFT        |                                                       |
+-------------------+-------------------------------------------------------+
| KMOD_LCTRL,       | Left Control, Right Contro or one of both was pressed.|
| KMOD_RCTRL,       |                                                       |
| KMOD_CTRL         |                                                       |
+-------------------+-------------------------------------------------------+
| KMOD_LALT,        | Left Alternative, Right Alternative or one of both    |
| KMOD_RALT,        | was pressed.                                          |
| KMOD_ALT          |                                                       |
+-------------------+-------------------------------------------------------+
| KMOD_LMETA,       | Left Meta, Right Met or one of both was pressed.      |
| KMOD_RMETA,       |                                                       |
| KMOD_META         |                                                       |
+-------------------+-------------------------------------------------------+
| KMOD_NUM,         | NumLock, CapsLock or Mode was pressed.                |
| KMOD_CAPS,        |                                                       |
| KMOD_MODE         |                                                       |
+-------------------+-------------------------------------------------------+

Surface Flags
-------------

The flags explained below are used by the :class:`pygame2.sdl.video.Surface`
class and various :mod:`pygame2.sdl.video` functions. Not all of them are
however used or applicable to both, the module functions and the class itself.
Using them, although not supported by the one or other, will not result in an
error, instead the inappropriate flags are silently ignored.

.. data:: SWSURFACE

   The surface is held in system memory. 

.. data:: HWSURFACE

   The surface is held in video memory.

.. data:: PREALLOC

   The surface uses preallocated memory.

.. data:: SRCCOLORKEY

   This flag turns on colorkeying for blits from this surface. If
   :data:`HWSURFACE` is also specified and colorkeyed blits are
   hardware-accelerated, then SDL will attempt to place the surface in video
   memory. Use :meth:`pygame2.sdl.video.Surface.set_colorkey` to set or clear
   this flag after surface creation.

.. data:: SRCALPHA

   This flag turns on alpha-blending for blits from this surface. If
   :data:`HWSURFACE` is also specified and alpha-blending blits are
   hardware-accelerated, then the surface will be placed in video memory if
   possible. Use :meth:`pygame2.sdl.video.Surface.set_alpha` to set or clear
   this flag after surface creation.

.. data:: ASYNCBLIT
  
   Enables the use of asynchronous updates of the surface. This will usually
   slow down blitting on single CPU machines, but may provide a speedincrease
   on SMP systems.
   
   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: ANYFORMAT

   Normally, if a video surface of the requested bits-per-pixel (bpp) is not
   available, SDL will emulate one with a shadow surface. Passing ANYFORMAT
   prevents this and causes SDL to use the video surface, regardless of its
   pixel depth.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: HWPALETTE 

   Give SDL exclusive palette access. Without this flag you may not always get
   the colors you request with :meth:`pygame2.sdl.video.Surface.set_colors` or 
   :meth:`pygame2.sdl.video.Surface.set_palette`.

.. data:: DOUBLEBUF

   Enable hardware double buffering; only valid with HWSURFACE. Calling
   :meth:`pygame2.sdl.video.Surface.flip` will flip the buffers and update
   the screen. All drawing will take place on the surface that is not displayed
   at the moment. If double buffering could not be enabled then
   :meth:`pygame2.sdl.video.Surface.flip` will just perform a
   :meth:`pygame2.sdl.video.Surface.update` on the entire screen.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: FULLSCREEN

   SDL will attempt to use a fullscreen mode. If a hardware resolution change
   is not possible (for whatever reason), the next higher resolution will be
   used and the display window centered on a black background.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: OPENGL

   Create an OpenGL rendering context. You should have previously set OpenGL
   video attributes with :meth:`pygame2.sdl.gl.set_attribute`.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: OPENGLBLIT

   Create an OpenGL rendering context, like above, but allow normal blitting
   operations. The screen (2D) surface may have an alpha channel, and
   :meth:`pygame2.sdl.video.Surface.update` must be used for updating changes
   to the screen surface.

   .. note::
      
      This option is kept for compatibility only, and is not recommended for
      new code.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: HWACCEL

   Surface blits uses hardware acceleration.

.. data:: RLEACCEL

   Colorkey blitting is accelerated with RLE.

.. data:: RESIZABLE

   Create a resizable window. When the window is resized by the user a
   :data:`VIDEORESIZE` event is generated and :meth:`pygame2.sdl.video.set_mode`
   can be called again with the new size.

   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.

.. data:: NOFRAME

   If possible, NOFRAME causes SDL to create a window with no title bar or
   frame decoration. Fullscreen modes automatically have this flag set.
   
   This is only available for the :mod:`pygame2.sdl.video` functions and the
   display surface set by :meth:`pygame2.sdl.video.set_mode`.


Various Constants
-----------------

.. data:: BYTEORDER

   The byteorder, SDL and pygame2.sdl were compiled with. It is set to either
   :data:`LIL_ENDIAN` or :data:`BIG_ENDIAN`.

.. data:: LIL_ENDIAN

   Indicates a little endian byte order.

.. data:: BIG_ENDIAN
    
   Indicates a big endian byte order.
