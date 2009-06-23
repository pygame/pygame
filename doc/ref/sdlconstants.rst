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

.. data:: BLEND_RGBA_AND

   Used for a binary AND'd blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_OR

   Used for a binary OR'd  blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_XOR

   Used for a binary XOR'd blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_MIN

   Used for a minimum blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_MAX

   Used for a maximum blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_AVG

   Used for an average blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_DIFF

   Used for a difference blend, with the per-pixel alpha value.

.. data:: BLEND_RGBA_SCREEN

   Used for a screen blend, with the per-pixel alpha value.

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

Various Constants
-----------------

.. data:: BYTEORDER

   The byteorder, SDL and pygame2.sdl were compiled with. It is set to either
   :data:`LIL_ENDIAN` or :data:`BIG_ENDIAN`.

.. data:: LIL_ENDIAN

   Indicates a little endian byte order.

.. data:: BIG_ENDIAN
    
   Indicates a big endian byte order.
