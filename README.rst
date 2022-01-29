.. image:: https://raw.githubusercontent.com/pygame/pygame/main/docs/pygame_logo.svg
  :alt: pygame
  :target: https://www.pygame.org/


|AppVeyorBuild| |PyPiVersion| |PyPiLicense|
|Python3| |GithubCommits| |LGTMAlerts| |LGTMGradePython| |LGTMGradeC|

pygame_ is a free and open-source cross-platform library
for the development of multimedia applications like video games using Python.
It uses the `Simple DirectMedia Layer library`_ and several other
popular libraries to abstract the most common functions, making writing
these programs a more intuitive task.

`We need your help`_ to make pygame the best it can be!
New contributors are welcome.


Installation
------------

::

   pip install pygame


Help
----

If you are just getting started with pygame, you should be able to
get started fairly quickly.  Pygame comes with many tutorials and
introductions.  There is also full reference documentation for the
entire library. Browse the documentation on the `docs page`_.

The online documentation stays up to date with the development version
of pygame on github.  This may be a bit newer than the version of pygame
you are using. To upgrade to the latest full release, run 
``pip install pygame --upgrade`` in your terminal.

Best of all, the examples directory has many playable small programs
which can get you started playing with the code right away.


Building From Source
--------------------

If you want to use features that are currently in development,
or you want to contribute to pygame, you will need to build pygame
locally from its source code, rather than pip installing it.

Installing from source is fairly automated. The most work will
involve compiling and installing all the pygame dependencies.  Once
that is done, run the ``setup.py`` script which will attempt to
auto-configure, build, and install pygame.

Much more information about installing and compiling is available
on the `Compilation wiki page`_.


Credits
-------

Thanks to everyone who has helped contribute to this library.
Special thanks are also in order.

* Marcus Von Appen: many changes, and fixes, 1.7.1+ freebsd maintainer
* Lenard Lindstrom: the 1.8+ windows maintainer, many changes, and fixes
* Brian Fisher for svn auto builder, bug tracker and many contributions
* Rene Dudfield: many changes, and fixes, 1.7+ release manager/maintainer
* Phil Hassey for his work on the pygame.org website
* DR0ID for his work on the sprite module
* Richard Goedeken for his smoothscale function
* Ulf Ekström for his pixel perfect collision detection code
* Pete Shinners: original author
* David Clark for filling the right-hand-man position
* Ed Boraas and Francis Irving: Debian packages
* Maxim Sobolev: FreeBSD packaging
* Bob Ippolito: MacOS and OS X porting (much work!)
* Jan Ekhol, Ray Kelm, and Peter Nicolai: putting up with early design ideas
* Nat Pryce for starting our unit tests
* Dan Richter for documentation work
* TheCorruptor for his incredible logos and graphics
* Nicholas Dudfield: many test improvements
* Alex Folkner for pygame-ctypes

Thanks to those sending in patches and fixes: Niki Spahiev, Gordon
Tyler, Nathaniel Pryce, Dave Wallace, John Popplewell, Michael Urman,
Andrew Straw, Michael Hudson, Ole Martin Bjoerndalen, Herve Cauwelier,
James Mazer, Lalo Martins, Timothy Stranex, Chad Lester, Matthias
Spiller, Bo Jangeborg, Dmitry Borisov, Campbell Barton, Diego Essaya,
Eyal Lotem, Regis Desgroppes, Emmanuel Hainry, Randy Kaelber
Matthew L Daniel, Nirav Patel, Forrest Voight, Charlie Nolan,
Frankie Robertson, John Krukoff, Lorenz Quack, Nick Irvine,
Michael George, Saul Spatz, Thomas Ibbotson, Tom Rothamel, Evan Kroske,
Cambell Barton.

And our bug hunters above and beyond: Angus, Guillaume Proux, Frank
Raiser, Austin Henry, Kaweh Kazemi, Arturo Aldama, Mike Mulcheck,
Michael Benfield, David Lau

There's many more folks out there who've submitted helpful ideas, kept
this project going, and basically made our life easier.  Thanks!

Many thank you's for people making documentation comments, and adding to the
pygame.org wiki.

Also many thanks for people creating games and putting them on the
pygame.org website for others to learn from and enjoy.

Lots of thanks to James Paige for hosting the pygame bugzilla.

Also a big thanks to Roger Dingledine and the crew at SEUL.ORG for our
excellent hosting.

Dependencies
------------

Pygame is obviously strongly dependent on SDL and Python.  It also
links to and embeds several other smaller libraries.  The font
module relies on SDL_ttf, which is dependent on freetype.  The mixer
(and mixer.music) modules depend on SDL_mixer.  The image module
depends on SDL_image, which also can use libjpeg and libpng.  The
transform module has an embedded version of SDL_rotozoom for its
own rotozoom function.  The surfarray module requires the Python
NumPy package for its multidimensional numeric arrays.
Dependency versions:

* CPython >= 3.6 or PyPy3
* SDL >= 2.0.0
* SDL_mixer >= 2.0.0
* SDL_image >= 2.0.0
* SDL_ttf >= 2.0.11
* SDL_gfx (optional, vendored in)
* NumPy >= 1.6.2 (optional)


License
-------

This library is distributed under `GNU LGPL version 2.1`_, which can
be found in the file ``docs/LGPL.txt``.  We reserve the right to place
future versions of this library under a different license.

This basically means you can use pygame in any project you want,
but if you make any changes or additions to pygame itself, those
must be released with a compatible license (preferably submitted
back to the pygame project).  Closed source and commercial games are fine.

The programs in the ``examples`` subdirectory are in the public domain.

See docs/licenses for licenses of dependencies.


.. |AppVeyorBuild| image:: https://ci.appveyor.com/api/projects/status/x4074ybuobsh4myx?svg=true
   :target: https://ci.appveyor.com/project/pygame/pygame

.. |PyPiVersion| image:: https://img.shields.io/pypi/v/pygame.svg?v=1
   :target: https://pypi.python.org/pypi/pygame

.. |PyPiLicense| image:: https://img.shields.io/pypi/l/pygame.svg?v=1
   :target: https://pypi.python.org/pypi/pygame

.. |Python3| image:: https://img.shields.io/badge/python-3-blue.svg?v=1

.. |GithubCommits| image:: https://img.shields.io/github/commits-since/pygame/pygame/2.1.2.svg
   :target: https://github.com/pygame/pygame/compare/2.1.2...main

.. |LGTMAlerts| image:: https://img.shields.io/lgtm/alerts/g/pygame/pygame.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/pygame/pygame/alerts/

.. |LGTMGradePython| image:: https://img.shields.io/lgtm/grade/python/g/pygame/pygame.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/pygame/pygame/context:python

.. |LGTMGradeC| image:: https://img.shields.io/lgtm/grade/cpp/g/pygame/pygame.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/pygame/pygame/context:cpp

.. _pygame: https://www.pygame.org
.. _Simple DirectMedia Layer library: https://www.libsdl.org
.. _We need your help: https://www.pygame.org/contribute.html
.. _Compilation wiki page: https://www.pygame.org/wiki/Compilation
.. _docs page: https://www.pygame.org/docs/
.. _GNU LGPL version 2.1: https://www.gnu.org/copyleft/lesser.html
