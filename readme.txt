
                        Pygame Readme
   Version 1.6
   October 23, 2003 Python Game Development
   by Pete Shinners http://www.pygame.org
   pete@shinners.org

   About

     Pygame is a cross-platfrom library designed to make it easy to
     write multimedia software, such as games, in Python. Pygame
     requires the Python language and SDL multimedia library. It can
     also make use of several other popular libraries.

   Installation

     You should definitely begin by installing a binary package for your
     system. The binary packages usually come with or give the
     information needed for dependencies. Choose an appropriate
     installer for your system and version of python from the pygame
     downloads page. http://www.pygame.org/download.shtml

     Installing from source is fairly automated. The most work will
     involve compiling and installing all the pygame dependencies. Once
     that is done run the "setup.py" script which will attempt to
     auto-configure, build, and install pygame.

     Much more information about installing and compiling is available
     in the install.html file.

   Help

     If you are just getting started with pygame, you should be able to
     get started fairly quickly. Pygame comes with many tutorials and
     introductions. There is also full reference documentation for the
     entire library. Browse the documentation from the documenantation
     index. docs/index.html.

     On the pygame website, there is also an online copy of this
     documentation. You should know that the online documentation stays
     up to date with the development version of pygame in cvs. This may
     be a bit newer than the version of pygame you are using.

     Best of all the examples directory has many playable small programs
     which can get started playing with the code right away.

   Credits

     Thanks to everyone who has helped contribute to this library.
     Special thanks are also in order.

     David Clark - for filling the right-hand-man position

     Ed Boraas and Francis Irving - Debian packages

     Maxim Sobolev - FreeBSD packaging

     Bob Ippolito - MacOS and OS X porting (much work!)

     Jan Ekhol, Ray Kelm, and Peter Nicolai - putting up with my early
   design ideas

     Nat Pryce for starting our unit tests

     TheCorruptor for his incredible logos and graphics

     Thanks to those sending in patches and fixes: Niki Spahiev, Gordon
   Tyler, Nathaniel Pryce, Dave Wallace, John Popplewell, Michael Urman,
   Andrew Straw, Michael Hudson, Ole Martin Bjoerndalen, Hervé Cauwelier,
   James Mazer, Lalo Martins, Timothy Stranex

     And our bug hunters above and beyond: Angus, Guillaume Proux, Frank
   Raiser, Austin Henry, Kaweh Kazemi, Arturo Aldama, Mike Mulcheck, Rene
   Dudfield, Michael Benfield

   There's many more folks out there who've submitted helpful ideas, kept
   this project going, and basically made my life easer, Thanks!

   Also a big thanks to Roger Dingledine and the crew at SEUL.ORG for our
   excellent hosting.

   Dependencies

     Pygame is obviously strongly dependent on SDL and Python. It also
     links to and embeds several other smaller libraries. The font
     module relies on SDL_tff, which is dependent on freetype. The mixer
     (and mixer.music) modules depend on SDL_mixer. The image module
     depends on SDL_image, which also can use libjpeg and libpng. The
     transform module has an embedded version of SDL_rotozoom for its
     own rotozoom function. The surfarray module requires the python
     Numeric package for its multidimensional numeric arrays.

   Todo / Ideas (feel free to submit)
     * unify more types/classes for python 2.2
     * transform.skew() function
     * transform.scroll() function
     * image filtering (colors,blurs,etc)
     * quake-like console with python interpreter
     * game lobby. client, simple server, and protocol
     * surfarrays should be able to do RGBA along with RGB
     * draw with transparancy
     * draw large sets of primitives with a single call

   License

     This library is distributed under GNU LGPL version 2.1, which can
     be found in the file "doc/LGPL". I reserve the right to place
     future versions of this library under a different license.
     http://www.gnu.org/copyleft/lesser.html

     This basically means you can use pygame in any project you want,
     but if you make any changes or additions to pygame itself, those
     must be released with a compatible license. (preferably submitted
     back to the pygame project). Closed source and commercial games are
     fine.

     The programs in the "examples" subdirectory are in the public
     domain.
