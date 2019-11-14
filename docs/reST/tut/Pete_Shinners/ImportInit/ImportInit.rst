.. TUTORIAL:Import and Initialize

.. include:: common.txt

********************************************
  Pygame Tutorials - Import and Initialize
********************************************
 
Import and Initialize
=====================

.. rst-class:: docinfo

:Author: Pete Shinners
:Contact: pete@shinners.org


Getting pygame imported and initialized is a very simple process. It is also
flexible enough to give you control over what is happening. Pygame is a
collection of different modules in a single python package. Some of the
modules are written in C, and some are written in python. Some modules
are also optional, and might not always be present.

This is just a quick introduction on what is going on when you import pygame.
For a clearer explanation definitely see the pygame examples.


Import
------

First we must import the pygame package. Since pygame version 1.4 this
has been updated to be much easier. Most games will import all of pygame like this. ::

  import pygame
  from pygame.locals import *

The first line here is the only necessary one. It imports all the available pygame
modules into the pygame package. The second line is optional, and puts a limited
set of constants and functions into the global namespace of your script.

An important thing to keep in mind is that several pygame modules are optional.
For example, one of these is the font module. When  you "import pygame", pygame
will check to see if the font module is available. If the font module is available
it will be imported as "pygame.font". If the module is not available, "pygame.font"
will be set to None. This makes it fairly easy to later on test if the font module is available.


Init
----

Before you can do much with pygame, you will need to initialize it. The most common
way to do this is just make one call. ::

  pygame.init()

This will attempt to initialize all the pygame modules for you. Not all pygame modules
need to be initialized, but this will automatically initialize the ones that do. You can
also easily initialize each pygame module by hand. For example to only initialize the
font module you would just call. ::

  pygame.font.init()

Note that if there is an error when you initialize with "pygame.init()", it will silently fail.
When hand initializing modules like this, any errors will raise an exception. Any
modules that must be initialized also have a "get_init()" function, which will return true
if the module has been initialized.

It is safe to call the init() function for any module more than once.


Quit
----

Modules that are initialized also usually have a quit() function that will clean up.
There is no need to explicitly call these, as pygame will cleanly quit all the
initialized modules when python finishes.
