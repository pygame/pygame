Pygame Version 1.1
June 23rd, 2001

Python Game Development
by Pete Shinners

http://www.pygame.org
pete@shinners.org




INTRO

Pygame is a cross-platfrom library designed to make it easy to write
multimedia software, such as games, in Python.

Pygame depends on SDL >= 1.1.7 and Python >= 2.0. It also is optionally
dependent on SDL_ttf, SDL_image, SDL_mixer, and Numeric.




INSTALLATION

Full installation instructions are found in docs/fullinstall.txt

Compilation/Installation is handled by the python distutils. Just
run "python setup.py" to perform the installation. This will make a
worthwhile attempt at configuring the compiler variables and getting
everything installed.

Distutils has a lot of installation options. To get started with them
run "python setup.py --help". Also, if you are getting errors compiling,
you can change the compiler flags in the "Setup" file (which resembles
a Makefile)




HELP

The best way to learn how to use pygame is to check out the example files
the "examples" subdirectory. All modules, functions, and objects in the
pygame package contain helpful docstrings. Full HTML reference and 
tutorials are also included in the docs directory.




TODO

At some point it would be nice to add support for more external
python libraries, here are some ideas...

 - MPEG video playback with smpeg
 - more drawing primitives in the draw module: circle, poly
 - more collision detection stuff: pixel2pixel, point2poly, etc
 - smarten the config.py and setup.py stuff




CREDITS

Thanks to everyone who made this possible, including:

- David Clark for RPMs and all around help
- Ed Boraas and Francis Irving for Debian packaging
- Maxim Sobolev for FreeBSD port maintenance
- Niki Spahiev, Gordon Tyler, and Dave Wallace for great patches
- Jan Ekhol, Ray Kelm, and Peter Nicolai for initial design reviews
- Bob Ippolito for the MacOS porting
- And our dutiful bug hunters; Angus, Guillaume Proux,
	Austin Henry, and Kaweh Kazemi

There's many more folks out there who've submitted helpful ideas,
kept this project going, and basically made my life easer, Thanks!

Also a big thanks to Roger Dingledine and the crew at SEUL.ORG
for our excellent site hosting.




LIBRARIES

Pygame is obviously strongly dependent on SDL and Python. It also
links to and embeds several other smaller libraries. The font module
relies on SDL_tff, which is dependent on freetype. The mixer (and
mixer.music) modules depend on SDL_mixer. The image module depends
on SDL_image, which also can use libjpeg and libpng. The transform
module has an embedded version of SDL_rotozoom for its own rotozoom
function. The surfarray module requires the python Numeric package
for its multidimensional numeric arrays.




LINKS

Pygame Website : http://www.pygame.org
Python Website : http://www.python.org
SDL Website    : http://www.libsdl.org




LICENSE

This library is distributed under GNU LGPL version 2.1, which can be
found in the file  "doc/LGPL".  I reserve the right to place future
versions of this library under a different license.
http://www.gnu.org/copyleft/lesser.html

This basically means you can use pygame in any project you want, but
if you make any changes or additions to pygame itself, those must be
released with a compatible license. (preferably submitted back to the
pygame project)

The programs in the "examples" subdirectory are in the public domain.

