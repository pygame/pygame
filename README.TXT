Pygame Version 1.0
April 4th, 2001

Python Game Development
by Pete Shinners

http://pygame.seul.org
pete@shinners.org




INTRO

Pygame is a cross-platfrom library designed to make it easy to write
multimedia software, such as games, in Python.

Pygame depends on SDL >= 1.1.7 and Python ?= 2.0. It also is optionally
dependent on SDL_ttf, SDL_image, SDL_mixer, and Numeric.




INSTALLTION

Full installation instructions are found in docs/fullinstall.txt

Compilation/Installation is handled by the python distutils. Just
run "python setup.py install" to perform the installation. This will
make a worthwhile attempt at configuring the compiler variables and
getting everything installed.

Distutils has a lot of installation options. To get started with them
run "python setup.py --help". Also, if you are getting errors compiling,
you can change the compiler flags in the "Setup" file (which resembles
a Makefile)




HELP

The best way to learn how to use pygame is to check out the example files
the "examples" subdirectory. All modules, functions, and objects in the
pygame package contain helpful docstrings. Full HTML reference and 
tutorials are also included in the docs directory.

The programs in the "examples" subdirectory are in the public domain.




TODO

At some point it would be nice to add support for more external
python libraries, here are some ideas...

 * Support for OpenGL (initial testing not so good)
 * more drawing primitives in the draw module
 * Interface with PIL images
 * GUI library. There's a few out there
 * Image transforms like flip, rotate, scale
 * MPEG video playback with smpeg
 * Macintosh support (just needs to compile)




CREDITS

Thanks to everyone who made this possible, including:

- David Clark for RPMs and all around help
- Francis Irving for Debian packaging
- Maxim Sobolev for FreeBSD port maintenance
- Niki Spahiev for bugfixes
- Jan Ekhol, Ray Kelm, and Peter Nicolai for initial design reviews
- And our dutiful bug hunters; Angus Young and Guillaume Proux

There's many more folks out there who've submitted helpful ideas,
kept this project going, and basically made my life easer; thanks!




LINKS

Pygame Website : http://pygame.seul.org
Python Website : http://www.python.org
SDL Website    : http://www.libsdl.org




LICENSE

This library is distributed under GNU LGPL version 2.1, which can be
found in the file  "doc/LGPL".  I reserve the right to place future
versions of this library under a different license.
http://www.gnu.org/copyleft/lesser.html

This basically means you can use pygame in any project you want, but
if you make any changes or additions to pygame itself, those must be
released with a compatible license.
( hopefully submitted back to the pygame project :] )


