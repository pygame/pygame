.. include:: common.txt

:mod:`pygame.display`
=====================

.. module:: pygame.display
   :synopsis: pygame module to control the display window and screen

| :sl:`pygame module to control the display window and screen`

This module offers control over the pygame display. Pygame has a single display
Surface that is either contained in a window or runs full screen. Once you
create the display you treat it as a regular Surface. Changes are not
immediately visible onscreen; you must choose one of the two flipping functions
to update the actual display.

The origin of the display, where x = 0 and y = 0, is the top left of the
screen. Both axes increase positively towards the bottom right of the screen.

The pygame display can actually be initialized in one of several modes. By
default, the display is a basic software driven framebuffer. You can request
special modules like hardware acceleration and OpenGL support. These are
controlled by flags passed to ``pygame.display.set_mode()``.

Pygame can only have a single display active at any time. Creating a new one
with ``pygame.display.set_mode()`` will close the previous display. If precise
control is needed over the pixel format or display resolutions, use the
functions ``pygame.display.mode_ok()``, ``pygame.display.list_modes()``, and
``pygame.display.Info()`` to query information about the display.

Once the display Surface is created, the functions from this module affect the
single existing display. The Surface becomes invalid if the module is
uninitialized. If a new display mode is set, the existing Surface will
automatically switch to operate on the new display.

When the display mode is set, several events are placed on the pygame event
queue. ``pygame.QUIT`` is sent when the user has requested the program to
shut down. The window will receive ``pygame.ACTIVEEVENT`` events as the display
gains and loses input focus. If the display is set with the
``pygame.RESIZABLE`` flag, ``pygame.VIDEORESIZE`` events will be sent when the
user adjusts the window dimensions. Hardware displays that draw direct to the
screen will get ``pygame.VIDEOEXPOSE`` events when portions of the window must
be redrawn.

Some display environments have an option for automatically stretching all
windows. When this option is enabled, this automatic stretching distorts the
appearance of the pygame window. In the pygame examples directory, there is
example code (prevent_display_stretching.py) which shows how to disable this
automatic stretching of the pygame display on Microsoft Windows (Vista or newer
required).

.. function:: init

   | :sl:`Initialize the display module`
   | :sg:`init() -> None`

   Initializes the pygame display module. The display module cannot do anything
   until it is initialized. This is usually handled for you automatically when
   you call the higher level ``pygame.init()``.

   Pygame will select from one of several internal display backends when it is
   initialized. The display mode will be chosen depending on the platform and
   permissions of current user. Before the display module is initialized the
   environment variable ``SDL_VIDEODRIVER`` can be set to control which backend
   is used. The systems with multiple choices are listed here.

   ::

      Windows : windib, directx
      Unix    : x11, dga, fbcon, directfb, ggi, vgl, svgalib, aalib

   On some platforms it is possible to embed the pygame display into an already
   existing window. To do this, the environment variable ``SDL_WINDOWID`` must
   be set to a string containing the window id or handle. The environment
   variable is checked when the pygame display is initialized. Be aware that
   there can be many strange side effects when running in an embedded display.

   It is harmless to call this more than once, repeated calls have no effect.

   .. ## pygame.display.init ##

.. function:: quit

   | :sl:`Uninitialize the display module`
   | :sg:`quit() -> None`

   This will shut down the entire display module. This means any active
   displays will be closed. This will also be handled automatically when the
   program exits.

   It is harmless to call this more than once, repeated calls have no effect.

   .. ## pygame.display.quit ##

.. function:: get_init

   | :sl:`Returns True if the display module has been initialized`
   | :sg:`get_init() -> bool`

   Returns True if the :mod:`pygame.display` module is currently initialized.

   .. ## pygame.display.get_init ##

.. function:: set_mode

   | :sl:`Initialize a window or screen for display`
   | :sg:`set_mode(resolution=(0,0), flags=0, depth=0, display=0) -> Surface`

   This function will create a display Surface. The arguments passed in are
   requests for a display type. The actual created display will be the best
   possible match supported by the system.

   The resolution argument is a pair of numbers representing the width and
   height. The flags argument is a collection of additional options. The depth
   argument represents the number of bits to use for color.

   The Surface that gets returned can be drawn to like a regular Surface but
   changes will eventually be seen on the monitor.

   If no resolution is passed or is set to (0, 0) and pygame uses ``SDL``
   version 1.2.10 or above, the created Surface will have the same size as the
   current screen resolution. If only the width or height are set to 0, the
   Surface will have the same width or height as the screen resolution. Using a
   ``SDL`` version prior to 1.2.10 will raise an exception.

   It is usually best to not pass the depth argument. It will default to the
   best and fastest color depth for the system. If your game requires a
   specific color format you can control the depth with this argument. Pygame
   will emulate an unavailable color depth which can be slow.

   When requesting fullscreen display modes, sometimes an exact match for the
   requested resolution cannot be made. In these situations pygame will select
   the closest compatible match. The returned surface will still always match
   the requested resolution.

   The flags argument controls which type of display you want. There are
   several to choose from, and you can even combine multiple types using the
   bitwise or operator, (the pipe "|" character). If you pass 0 or no flags
   argument it will default to a software driven window. Here are the display
   flags you will want to choose from:

   ::

      pygame.FULLSCREEN    create a fullscreen display
      pygame.DOUBLEBUF     recommended for HWSURFACE or OPENGL
      pygame.HWSURFACE     hardware accelerated, only in FULLSCREEN
      pygame.OPENGL        create an OpenGL-renderable display
      pygame.RESIZABLE     display window should be sizeable
      pygame.NOFRAME       display window will have no border or controls

   For example:

   ::

        # Open a window on the screen
        screen_width=700
        screen_height=400
        screen=pygame.display.set_mode([screen_width,screen_height])

   The display index 0 means the default display is used.

   The display argument is new with pygame 1.9.5.

   .. ## pygame.display.set_mode ##

.. function:: get_surface

   | :sl:`Get a reference to the currently set display surface`
   | :sg:`get_surface() -> Surface`

   Return a reference to the currently set display Surface. If no display mode
   has been set this will return None.

   .. ## pygame.display.get_surface ##

.. function:: flip

   | :sl:`Update the full display Surface to the screen`
   | :sg:`flip() -> None`

   This will update the contents of the entire display. If your display mode is
   using the flags ``pygame.HWSURFACE`` and ``pygame.DOUBLEBUF``, this will
   wait for a vertical retrace and swap the surfaces. If you are using a
   different type of display mode, it will simply update the entire contents of
   the surface.

   When using an ``pygame.OPENGL`` display mode this will perform a gl buffer
   swap.

   .. ## pygame.display.flip ##

.. function:: update

   | :sl:`Update portions of the screen for software displays`
   | :sg:`update(rectangle=None) -> None`
   | :sg:`update(rectangle_list) -> None`

   This function is like an optimized version of ``pygame.display.flip()`` for
   software displays. It allows only a portion of the screen to updated,
   instead of the entire area. If no argument is passed it updates the entire
   Surface area like ``pygame.display.flip()``.

   You can pass the function a single rectangle, or a sequence of rectangles.
   It is more efficient to pass many rectangles at once than to call update
   multiple times with single or a partial list of rectangles. If passing a
   sequence of rectangles it is safe to include None values in the list, which
   will be skipped.

   This call cannot be used on ``pygame.OPENGL`` displays and will generate an
   exception.

   .. ## pygame.display.update ##

.. function:: get_driver

   | :sl:`Get the name of the pygame display backend`
   | :sg:`get_driver() -> name`

   Pygame chooses one of many available display backends when it is
   initialized. This returns the internal name used for the display backend.
   This can be used to provide limited information about what display
   capabilities might be accelerated. See the ``SDL_VIDEODRIVER`` flags in
   ``pygame.display.set_mode()`` to see some of the common options.

   .. ## pygame.display.get_driver ##

.. function:: Info

   | :sl:`Create a video display information object`
   | :sg:`Info() -> VideoInfo`

   Creates a simple object containing several attributes to describe the
   current graphics environment. If this is called before
   ``pygame.display.set_mode()`` some platforms can provide information about
   the default display mode. This can also be called after setting the display
   mode to verify specific display options were satisfied. The VidInfo object
   has several attributes:

   ::

     hw:         True if the display is hardware accelerated
     wm:         True if windowed display modes can be used
     video_mem:  The megabytes of video memory on the display. This is 0 if unknown
     bitsize:    Number of bits used to store each pixel
     bytesize:   Number of bytes used to store each pixel
     masks:      Four values used to pack RGBA values into pixels
     shifts:     Four values used to pack RGBA values into pixels
     losses:     Four values used to pack RGBA values into pixels
     blit_hw:    True if hardware Surface blitting is accelerated
     blit_hw_CC: True if hardware Surface colorkey blitting is accelerated
     blit_hw_A:  True if hardware Surface pixel alpha blitting is accelerated
     blit_sw:    True if software Surface blitting is accelerated
     blit_sw_CC: True if software Surface colorkey blitting is accelerated
     blit_sw_A:  True if software Surface pixel alpha blitting is accelerated
     current_h, current_w:  Height and width of the current video mode, or of the
       desktop mode if called before the display.set_mode is called.
       (current_h, current_w are available since SDL 1.2.10, and pygame 1.8.0)
       They are -1 on error, or if an old SDL is being used.

   .. ## pygame.display.Info ##

.. function:: get_wm_info

   | :sl:`Get information about the current windowing system`
   | :sg:`get_wm_info() -> dict`

   Creates a dictionary filled with string keys. The strings and values are
   arbitrarily created by the system. Some systems may have no information and
   an empty dictionary will be returned. Most platforms will return a "window"
   key with the value set to the system id for the current display.

   New with pygame 1.7.1

   .. ## pygame.display.get_wm_info ##

.. function:: list_modes

   | :sl:`Get list of available fullscreen modes`
   | :sg:`list_modes(depth=0, flags=pygame.FULLSCREEN, display=0) -> list`

   This function returns a list of possible dimensions for a specified color
   depth. The return value will be an empty list if no display modes are
   available with the given arguments. A return value of -1 means that any
   requested resolution should work (this is likely the case for windowed
   modes). Mode sizes are sorted from biggest to smallest.

   If depth is 0, ``SDL`` will choose the current/best color depth for the
   display. The flags defaults to ``pygame.FULLSCREEN``, but you may need to
   add additional flags for specific fullscreen modes.

   The display index 0 means the default display is used.

   The display argument is new with pygame 1.9.5.

   .. ## pygame.display.list_modes ##

.. function:: mode_ok

   | :sl:`Pick the best color depth for a display mode`
   | :sg:`mode_ok(size, flags=0, depth=0, display=0) -> depth`

   This function uses the same arguments as ``pygame.display.set_mode()``. It
   is used to determine if a requested display mode is available. It will
   return 0 if the display mode cannot be set. Otherwise it will return a pixel
   depth that best matches the display asked for.

   Usually the depth argument is not passed, but some platforms can support
   multiple display depths. If passed it will hint to which depth is a better
   match.

   The most useful flags to pass will be ``pygame.HWSURFACE``,
   ``pygame.DOUBLEBUF``, and maybe ``pygame.FULLSCREEN``. The function will
   return 0 if these display flags cannot be set.

   The display index 0 means the default display is used.

   The display argument is new with pygame 1.9.5.

   .. ## pygame.display.mode_ok ##

.. function:: gl_get_attribute

   | :sl:`Get the value for an OpenGL flag for the current display`
   | :sg:`gl_get_attribute(flag) -> value`

   After calling ``pygame.display.set_mode()`` with the ``pygame.OPENGL`` flag,
   it is a good idea to check the value of any requested OpenGL attributes. See
   ``pygame.display.gl_set_attribute()`` for a list of valid flags.

   .. ## pygame.display.gl_get_attribute ##

.. function:: gl_set_attribute

   | :sl:`Request an OpenGL display attribute for the display mode`
   | :sg:`gl_set_attribute(flag, value) -> None`

   When calling ``pygame.display.set_mode()`` with the ``pygame.OPENGL`` flag,
   Pygame automatically handles setting the OpenGL attributes like color and
   double-buffering. OpenGL offers several other attributes you may want control
   over. Pass one of these attributes as the flag, and its appropriate value.
   This must be called before ``pygame.display.set_mode()``

   The ``OPENGL`` flags are;

   ::

     GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE, GL_ACCUM_RED_SIZE,
     GL_ACCUM_GREEN_SIZE,  GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE,
     GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES, GL_STEREO

   .. ## pygame.display.gl_set_attribute ##

.. function:: get_active

   | :sl:`Returns True when the display is active on the display`
   | :sg:`get_active() -> bool`

   After ``pygame.display.set_mode()`` is called the display Surface will be
   visible on the screen. Most windowed displays can be hidden by the user. If
   the display Surface is hidden or iconified this will return False.

   .. ## pygame.display.get_active ##

.. function:: iconify

   | :sl:`Iconify the display surface`
   | :sg:`iconify() -> bool`

   Request the window for the display surface be iconified or hidden. Not all
   systems and displays support an iconified display. The function will return
   True if successful.

   When the display is iconified ``pygame.display.get_active()`` will return
   False. The event queue should receive a ``ACTIVEEVENT`` event when the
   window has been iconified.

   .. ## pygame.display.iconify ##

.. function:: toggle_fullscreen

   | :sl:`Switch between fullscreen and windowed displays`
   | :sg:`toggle_fullscreen() -> bool`

   Switches the display window between windowed and fullscreen modes. This
   function only works under the UNIX X11 video driver. For most situations it
   is better to call ``pygame.display.set_mode()`` with new display flags.

   .. ## pygame.display.toggle_fullscreen ##

.. function:: set_gamma

   | :sl:`Change the hardware gamma ramps`
   | :sg:`set_gamma(red, green=None, blue=None) -> bool`

   Set the red, green, and blue gamma values on the display hardware. If the
   green and blue arguments are not passed, they will both be the same as red.
   Not all systems and hardware support gamma ramps, if the function succeeds
   it will return True.

   A gamma value of 1.0 creates a linear color table. Lower values will darken
   the display and higher values will brighten.

   .. ## pygame.display.set_gamma ##

.. function:: set_gamma_ramp

   | :sl:`Change the hardware gamma ramps with a custom lookup`
   | :sg:`set_gamma_ramp(red, green, blue) -> bool`

   Set the red, green, and blue gamma ramps with an explicit lookup table. Each
   argument should be sequence of 256 integers. The integers should range
   between 0 and 0xffff. Not all systems and hardware support gamma ramps, if
   the function succeeds it will return True.

   .. ## pygame.display.set_gamma_ramp ##

.. function:: set_icon

   | :sl:`Change the system image for the display window`
   | :sg:`set_icon(Surface) -> None`

   Sets the runtime icon the system will use to represent the display window.
   All windows default to a simple pygame logo for the window icon.

   You can pass any surface, but most systems want a smaller image around
   32x32. The image can have colorkey transparency which will be passed to the
   system.

   Some systems do not allow the window icon to change after it has been shown.
   This function can be called before ``pygame.display.set_mode()`` to create
   the icon before the display mode is set.

   .. ## pygame.display.set_icon ##

.. function:: set_caption

   | :sl:`Set the current window caption`
   | :sg:`set_caption(title, icontitle=None) -> None`

   If the display has a window title, this function will change the name on the
   window. Some systems support an alternate shorter title to be used for
   minimized displays.

   .. ## pygame.display.set_caption ##

.. function:: get_caption

   | :sl:`Get the current window caption`
   | :sg:`get_caption() -> (title, icontitle)`

   Returns the title and icontitle for the display Surface. These will often be
   the same value.

   .. ## pygame.display.get_caption ##

.. function:: set_palette

   | :sl:`Set the display color palette for indexed displays`
   | :sg:`set_palette(palette=None) -> None`

   This will change the video display color palette for 8-bit displays. This
   does not change the palette for the actual display Surface, only the palette
   that is used to display the Surface. If no palette argument is passed, the
   system default palette will be restored. The palette is a sequence of
   ``RGB`` triplets.

   .. ## pygame.display.set_palette ##

.. function:: get_num_displays

   | :sl:`Return the number of displays`
   | :sg:`get_num_displays() -> int`

   Returns the number of available displays. This is always 1 if
   :func:`pygame.get_sdl_version()` returns a major version number below 2.

   get_num_displays is new in pygame 1.9.5.

   .. ## pygame.display.get_num_displays ##

.. ## pygame.display ##
