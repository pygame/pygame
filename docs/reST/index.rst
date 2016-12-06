.. Pygame documentation master file, created by
   sphinx-quickstart on Sat Mar  5 11:56:39 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pygame Front Page
=================

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   ref/*
   tut/*
   filepaths

Documents
---------

`Readme`_
  Basic information about Pygame, what it is, who is involved, and where to find it.

`Install`_
  Steps needed to compile Pygame on several platforms.
  Also help on finding and installing prebuilt binaries for your system.

`File Path Function Arguments`_
  How Pygame handles file system paths.

`LGPL License`_
  This is the license Pygame is distributed under.
  It provides for Pygame to be distributed with open source and commercial software.
  Generally, if Pygame is not changed, it can be used with any type of program.
  
Tutorials
---------

:doc:`Introduction to Pygame <tut/PygameIntro>`
  An introduction to the basics of Pygame.
  This is written for users of Python and appeared in volume two of the Py magazine.

:doc:`Import and Initialize <tut/ImportInit>`
  The beginning steps on importing and initializing Pygame.
  The Pygame package is made of several modules.
  Some modules are not included on all platforms.

:doc:`How do I move an Image? <tut/MoveIt>`
  A basic tutorial that covers the concepts behind 2D computer animation.
  Information about drawing and clearing objects to make them appear animated.

:doc:`Chimp Tutorial, Line by Line <tut/ChimpLineByLine>`
  The pygame examples include a simple program with an interactive fist and a chimpanzee.
  This was inspired by the annoying flash banner of the early 2000's.
  This tutorial examines every line of code used in the example.

:doc:`Sprite Module Introduction <tut/SpriteIntro>`
  Pygame includes a higher level sprite module to help organize games.
  The sprite module includes several classes that help manage details found in almost all games types.
  The Sprite classes are a bit more advanced than the regular Pygame modules,
  and need more understanding to be properly used.

:doc:`Surfarray Introduction <tut/SurfarrayIntro>`
  Pygame used the Numpy python module to allow efficient per pixel effects on images.
  Using the surfae arrays is an advanced feature that allows custom effects and filters.
  This also examines some of the simple effects from the Pygame example, arraydemo.py.

:doc:`Camera Module Introduction <tut/CameraIntro>`
  Pygame, as of 1.9, has a camera module that allows you to capture images,
  watch live streams, and do some basic computer vision.
  This tutorial covers those use cases.

:doc:`Newbie Guide <tut/newbieguide>`
  A list of thirteen helpful tips for people to get comfortable using Pygame.

:doc:`Making Games Tutorial <tut/MakeGames>`
  A large tutorial that covers the bigger topics needed to create an entire game.

:doc:`Display Modes <tut/DisplayModes>`
  Getting a display surface for the screen.

Reference
---------

:ref:`genindex`
  A list of all functions, classes, and methods in the Pygame package.

:doc:`ref/bufferproxy`
  An array protocol view of surface pixels

:doc:`ref/cdrom`
  How to access and control the CD audio devices.

:doc:`ref/color`
  Color representation.

:doc:`ref/cursors`
  Loading and compiling cursor images.

:doc:`ref/display`
  Configure the display surface.

:doc:`ref/draw`
  Drawing simple shapes like lines and ellipses to surfaces.

:doc:`ref/event`
  Manage the incoming events from various input devices and the windowing platform.

:doc:`ref/examples`
  Various programs demonstrating the use of individual pyame modules.

:doc:`ref/font`
  Loading and rendering Truetype fonts.

:doc:`ref/freetype`
  Enhanced Pygame module for loading and rendering font faces.

:doc:`ref/gfxdraw`
  Anti-aliasing draw functions.

:doc:`ref/image`
  Loading, saving, and transferring of surfaces.

:doc:`ref/joystick`
  Manage the joystick devices.

:doc:`ref/key`
  Manage the keyboard device.

:doc:`ref/locals`
  Pygame constants.

:doc:`ref/mixer`
  Load and play sounds

:doc:`ref/mouse`
  Manage the mouse device and display.

:doc:`ref/music`
  Play streaming music tracks.

:doc:`ref/overlay`
  Access advanced video overlays.

:doc:`ref/pygame`
  Top level functions to manage Pygame.

:doc:`ref/pixelarray`
  Manipulate image pixel data.

:doc:`ref/rect`
  Flexible container for a rectangle.

:doc:`ref/scrap`
  Native clipboard access.

:doc:`ref/sndarray`
  Manipulate sound sample data.

:doc:`ref/sprite`
  Higher level objects to represent game images.

:doc:`ref/surface`
  Objects for images and the screen.

:doc:`ref/surfarray`
  Manipulate image pixel data.

:doc:`ref/tests`
  Test Pygame.

:doc:`ref/time`
  Manage timing and framerate.

:doc:`ref/transform`
  Resize and move images.

:ref:`search`
  Search Pygame documents by keyword.

.. _Readme: ../readme.html

.. _Install: ../install.html

.. _File Path Function Arguments: filepaths.html

.. _LGPL License: ../LGPL
