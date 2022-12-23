.. include:: common.txt

**********************
  Kicking things off
**********************

.. role:: citetitle(emphasis)

.. _makegames-3:

3. Kicking things off
=====================

The first sections of code are relatively simple, and, once written, can usually be reused in every game you consequently make. They
will do all of the boring, generic tasks like loading modules, loading images, opening networking connections, playing music, and so
on. They will also include some simple but effective error handling, and any customisation you wish to provide on top of functions
provided by modules like ``sys`` and ``pygame``.


.. _makegames-3-1:

3.1. The first lines, and loading modules
-----------------------------------------

First off, you need to start off your game and load up your modules. It's always a good idea to set a few things straight at the top of
the main source file, such as the name of the file, what it contains, the license it is under, and any other helpful info you might
want to give those who will be looking at it. Then you can load modules, with some error checking so that Python doesn't print out
a nasty traceback, which non-programmers won't understand. The code is fairly simple, so I won't bother explaining any of it::

  #!/usr/bin/env python
  #
  # Tom's Pong
  # A simple pong game with realistic physics and AI
  # http://tomchance.org.uk/projects/pong
  #
  # Released under the GNU General Public License

  VERSION = "0.4"

  try:
      import sys
      import random
      import math
      import os
      import getopt
      import pygame
      from socket import *
      from pygame.locals import *
  except ImportError, err:
      print(f"couldn't load module. {err}")
      sys.exit(2)


.. _makegames-3-2:

3.2. Resource handling functions
--------------------------------

In the :doc:`Line By Line Chimp <ChimpLineByLine>` example, the first code to be written was for loading images and sounds. As these
were totally independent of any game logic or game objects, they were written as separate functions, and were written first so
that later code could make use of them. I generally put all my code of this nature first, in their own, classless functions; these
will, generally speaking, be resource handling functions. You can of course create classes for these, so that you can group them
together, and maybe have an object with which you can control all of your resources. As with any good programming environment, it's up
to you to develop your own best practice and style.

It's always a good idea to write your own resource handling functions,
because although Pygame has methods for opening images and sounds, and other modules will have their methods of opening other
resources, those methods can take up more than one line, they can require consistent modification by yourself, and they often don't
provide satisfactory error handling. Writing resource handling functions gives you sophisticated, reusable code, and gives you more
control over your resources. Take this example of an image loading function::

  def load_png(name):
      """ Load image and return image object"""
      fullname = os.path.join("data", name)
      try:
          image = pygame.image.load(fullname)
          if image.get_alpha() is None:
              image = image.convert()
          else:
              image = image.convert_alpha()
      except FileNotFoundError:
          print(f"Cannot load image: {fullname}")
          raise SystemExit
      return image, image.get_rect()

Here we make a more sophisticated image loading function than the one provided by :func:`pygame.image.load`. Note that
the first line of the function is a documentation string describing what the function does, and what object(s) it returns. The
function assumes that all of your images are in a directory called data, and so it takes the filename and creates the full pathname,
for example ``data/ball.png``, using the :citetitle:`os` module to ensure cross-platform compatibility. Then it
tries to load the image, and convert any alpha regions so you can achieve transparency, and it returns a more human-readable error
if there's a problem. Finally it returns the image object, and its :class:`rect <pygame.Rect>`.

You can make similar functions for loading any other resources, such as loading sounds. You can also make resource handling classes,
to give you more flexibility with more complex resources. For example, you could make a music class, with an ``__init__``
function that loads the sound (perhaps borrowing from a ``load_sound()`` function), a function to pause the music, and a
function to restart. Another handy resource handling class is for network connections. Functions to open sockets, pass data with
suitable security and error checking, close sockets, finger addresses, and other network tasks, can make writing a game with network
capabilities relatively painless.

Remember the chief task of these functions/classes is to ensure that by the time you get around to writing game object classes,
and the main loop, there's almost nothing left to do. Class inheritance can make these basic classes especially handy. Don't go
overboard though; functions which will only be used by one class should be written as part of that class, not as a global
function.
