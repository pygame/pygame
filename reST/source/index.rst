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

   *

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

`Introduction to Pygame`_
  An introduction to the basics of Pygame.
  This is written for users of Python and appeared in volume two of the Py magazine.

`Import and Initialize`_
  The beginning steps on importing and initializing Pygame.
  The Pygame package is made of several modules.
  Some modules are not included on all platforms.

`How do I move an Image?`_
  A basic tutorial that covers the concepts behind 2D computer animation.
  Information about drawing and clearing objects to make them appear animated.

`Chimp Tutorial, Line by Line`_
  The pygame examples include a simple program with an interactive fist and a chimpanzee.
  This was inspired by the annoying flash banner of the early 2000's.
  This tutorial examines every line of coded used in the example.

`Sprite Module Introduction`_
  Pygame includes a higher level sprite module to help organize games.
  The sprite module includes several classes that help manage details found in almost all games types.
  The Sprite classes are a bit more advanced than the regular Pygame modules,
  and need more understanding to be properly used.

`Surfarray Introduction`_
  Pygame used the Numpy python module to allow efficient per pixel effects on images.
  Using the surfae arrays is an advanced feature that allows custom effects and filters.
  This also examines some of the simple effects from the Pygame example, arraydemo.py.

`Camera Module Introduction`_
  Pygame, as of 1.9, has a camera module that allows you to capture images,
  watch live streams, and do some basic computer vision.
  This tutorial covers those use cases.

`Newbie Guide`_
  A list of thirteen helpful tips for people to get comfortable using Pygame.

`Making Games Tutorial`_
  A large tutorial that covers the bigger topics needed to create an entire game.

Reference
---------

:ref:`genindex`
  A list of all functions, classes, and methods in the Pygame package.

:doc:`cdrom`
  How to access and control the CD audio devices.

:doc:`color`
  Color representation.

:doc:`cursors`
  Loading and compiling cursor images.

:doc:`display`
  Configure the display surface.

:doc:`draw`
  Drawing simple shapes like lines and ellipses to surfaces.

:doc:`event`
  Manage the incoming events from various input devices and the windowing platform.

:doc:`examples`
  Various programs demonstrating the use of individual pyame modules.

:doc:`font`
  Loading and rendering Truetype fonts.

:doc:`gfxdraw`
  Anti-aliasing draw functions.

:doc:`image`
  Loading, saving, and transferring of surfaces.

:doc:`joystick`
  Manage the joystick devices.

:doc:`key`
  Manage the keyboard device.

:doc:`locals`
  Pygame constants.

:doc:`mixer`
  Load and play sounds

:doc:`mouse`
  Manage the mouse device and display.

:doc:`movie`
  Video playback from MPEG movies.

:doc:`music`
  Play streaming music tracks.

:doc:`overlay`
  Access advanced video overlays.

:doc:`pygame`
  Top level functions to manage Pygame.

:doc:`pixelarray`
  Manipulate image pixel data.

:doc:`rect`
  Flexible container for a rectangle.

:doc:`scrap`
  Native clipboard access.

:doc:`sndarray`
  Manipulate sound sample data.

:doc:`sprite`
  Higher level objects to represent game images.

:doc:`surface`
  Objects for images and the screen.

:doc:`surfarray`
  Manipulate image pixel data.

:doc:`tests`
  Test Pygame.

:doc:`time`
  Manage timing and framerate.

:doc:`transform`
  Resize and move images.

:ref:`modindex`
  Global module index.

:ref:`search`
  Search Pygame documents by keyword.

.. _Readme: readme.html

.. _Install: install.html

.. _File Path Function Arguments: filepaths.html

.. _LGPL License: LGPL

.. _Introduction to Pygame: tut/intro/intro.html

.. _Import and Initialize: tut/importinit.html

.. _How do I move an Image?: tut/Moveit.html

.. _Chimp Tutorial, Line by Line: tut/chimp/ChimpLineByLine.html

.. _Sprite Module Introduction: tut/SpriteIntro.html

.. _Surfarray Introduction: tut/surfarray/SurfarrayIntro.html

.. _Camera Module Introduction: tut/camera/CameraIntro.html

.. _Newbie Guide: tut/newbieguide.html

.. _Making Games Tutorial: tut/tom/MakeGames.html
