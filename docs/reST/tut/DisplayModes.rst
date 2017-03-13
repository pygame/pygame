.. TUTORIAL: Choosing and Configuring Display Modes

.. include:: common.txt

********************************************
  Pygame Tutorials - Setting Display Modes
********************************************


Setting Display Modes
=====================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org
:Revision: 1.2, 2002-05-21 (Updated 2016-10-18, 2017-03-13)


Introduction
------------

Setting the display mode in *pygame* creates a visible image surface
on the monitor.
This surface can either cover the full screen, or be windowed
on platforms that support a window manager.
The display surface is nothing more than a standard *pygame* surface object.
There are special functions needed in the :mod:`pygame.display`
module to keep the image surface contents updated on the monitor.

Setting the display mode in *pygame* is an easier task than with most
graphic libraries.
The advantage is if your display mode is not available,
*pygame* will emulate the display mode that you asked for.
*Pygame* will select a display resolution and color depth that best matches
the settings you have requested,
then allow you to access the display with the format you have requested.
In reality, since the :mod:`pygame.display` module is
a binding around the SDL library, SDL is really doing all this work.

There are advantages and disadvantages to setting the display mode in this
manner.
The advantage is that if your game requires a specific display mode,
your game will run on platforms that do not support your requirements.
It also makes life easier when your getting something started,
it is always easy to go back later and make the mode selection a little more
particular.
The disadvantage is that what you request is not always what you will get.
There is also a performance penalty when the display mode must be emulated.
This tutorial will help you understand the different methods for querying
the platforms display capabilities, and setting the display mode for your game.


Setting Basics
--------------

The first thing to learn about is how to actually set the current display mode.
The display mode may be set at any time after the :mod:`pygame.display`
module has been initialized.
If you have previously set the display mode,
setting it again will change the current mode.
Setting the display mode is handled with the function
:func:`pygame.display.set_mode((width, height), flags, depth)
<pygame.display.set_mode>`.
The only required argument in this function is a sequence containing
the width and height of the new display mode.
The depth flag is the requested bits per pixel for the surface.
If the given depth is 8, *pygame* will create a color-mapped surface.
When given a higher bit depth, *pygame* will use a packed color mode.
Much more information about depths and color modes can be found in the
documentation for the display and surface modules.
The default value for depth is 0.
When given an argument of 0, *pygame* will select the best bit depth to use,
usually the same as the system's current bit depth.
The flags argument lets you control extra features for the display mode.
You can create the display surface in hardware memory with the
:any:`HWSURFACE <pygame.display.set_mode>` flag.
Again, more information about this is found in the *pygame* reference documents.


How to Decide
-------------

So how do you select a display mode that is going to work best with your
graphic resources and the platform your game is running on?
There are several methods for gathering information about the display device.
All of these methods must be called after the display module has been
initialized, but you likely want to call them before setting the display mode.
First, :func:`pygame.display.Info() <pygame.display.Info>`
will return a special object type of VidInfo,
which can tell you a lot about the graphics driver capabilities.
The function
:func:`pygame.display.list_modes(depth, flags) <pygame.display.list_modes>`
can be used to find the supported graphic modes by the system.
:func:`pygame.display.mode_ok((width, height), flags, depth)
<pygame.display.mode_ok>` takes the same arguments as
:func:`set_mode() <pygame.display.set_mode>`,
but returns the closest matching bit depth to the one you request.
Lastly, :func:`pygame.display.get_driver() <pygame.display.get_driver>`
will return the name of the graphics driver selected by *pygame*.

Just remember the golden rule.
*Pygame* will work with pretty much any display mode you request.
Some display modes will need to be emulated,
which will slow your game down,
since *pygame* will need to convert every update you make to the
"real" display mode. The best bet is to always let *pygame*
choose the best bit depth,
and convert all your graphic resources to that format when they are loaded.
You let *pygame* choose it's bit depth by calling
:func:`set_mode() <pygame.display.set_mode>`
with no depth argument or a depth of 0,
or you can call
:func:`mode_ok() <pygame.display.mode_ok>`
to find a closest matching bit depth to what you need.

When your display mode is windowed,
you usually must math the same bit depth as the desktop.
When you are fullscreen, some platforms can switch to any bit depth that
best suits your needs.
You can find the depth of the current desktop if you get a VidInfo object
before ever setting your display mode.

After setting the display mode,
you can find out information about it's settings by getting a VidInfo object,
or by calling any of the Surface.get* methods on the display surface.


Functions
---------

These are the routines you can use to determine the most appropriate
display mode.
You can find more information about these functions in the display module
documentation.

  :func:`pygame.display.mode_ok(size, flags, depth) <pygame.display.mode_ok>`

    This function takes the exact same arguments as pygame.display.set_mode().
    It returns the best available bit depth for the mode you have described.
    If this returns zero,
    then the desired display mode is not available without emulation.

  :func:`pygame.display.list_modes(depth, flags) <pygame.display.list_modes>`

    Returns a list of supported display modes with the requested
    depth and flags.
    An empty list is returned when there are no modes.
    The flags argument defaults to :any:`FULLSCREEN <pygame.display.set_mode>`\ .
    If you specify your own flags without :any:`FULLSCREEN <pygame.display.set_mode>`\ ,
    you will likely get a return value of -1.
    This means that any display size is fine, since the display will be windowed.
    Note that the listed modes are sorted largest to smallest.

  :func:`pygame.display.Info() <pygame.display.Info>`

    This function returns an object with many members describing
    the display device.
    Printing the VidInfo object will quickly show you all the
    members and values for this object. ::

      >>> import pygame.display
      >>> pygame.display.init()
      >>> info = pygame.display.Info()
      >>> print info
      <VideoInfo(hw = 1, wm = 1,video_mem = 27354
                 blit_hw = 1, blit_hw_CC = 1, blit_hw_A = 0,
                 blit_sw = 1, blit_sw_CC = 1, blit_sw_A = 0,
                 bitsize  = 32, bytesize = 4,
                 masks =  (16711680, 65280, 255, 0),
                 shifts = (16, 8, 0, 0),
                 losses =  (0, 0, 0, 8)>

You can test all these flags as simply members of the VidInfo object.
The different blit flags tell if hardware acceleration is supported when
blitting from the various types of surfaces to a hardware surface.


Examples
--------

Here are some examples of different methods to init the graphics display.
They should help you get an idea of how to go about setting your display mode. ::

  >>> #give me the best depth with a 640 x 480 windowed display
  >>> pygame.display.set_mode((640, 480))

  >>> #give me the biggest 16-bit display available
  >>> modes = pygame.display.list_modes(16)
  >>> if not modes:
  ...     print '16-bit not supported'
  ... else:
  ...     print 'Found Resolution:', modes[0]
  ...     pygame.display.set_mode(modes[0], FULLSCREEN, 16)

  >>> #need an 8-bit surface, nothing else will do
  >>> if pygame.display.mode_ok((800, 600), 0, 8) != 8:
  ...     print 'Can only work with an 8-bit display, sorry'
  ... else:
  ...     pygame.display.set_mode((800, 600), 0, 8)
