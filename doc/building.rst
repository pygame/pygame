#################################
Building Pygame2 on your platform
#################################

These section provide an overview and guidance for building and
installing pygame2 on various target platforms. Currently building
instructions for Unix compatible operating system and Microsoft Windows
exist.

Contents:

.. toctree::
   :maxdepth: 2

   builddarwin.rst
   buildmingw.rst
   buildvc.rst
   buildunix.rst

Besides those build instructions, pygame2's build process can be tweaked
for all target platforms as described below.

Environment Settings
====================
You can use certain environment settings to influence the build. Those
are evaluated in the "cfg.py" file in the top source directory and can
be tweaked directly within that file, too.

By default, the build system will test for certain environment variables
being set to control the build of various features. If they are *not*
available, the build system assumes that you want to have as many
features as possible and thus will enable them.

Currently the following environment variables are controlling, which
components of pgreloaded should be build: ::

  WITH_SDL=[yes|no|1|True]                Example: make -DWITH_SDL=yes

Build and install the :mod:`pygame2.sdl` module. This wraps the SDL
library and is required for any other SDL related module in
pygame2. ::

  WITH_SDL_MIXER=[yes|no|1|True]          Example: make -DWITH_SDL_MIXER=no

Build and install the :mod:`pygame2.sdlmixer` module. This wraps the
SDL_mixer library. ::

  WITH_SDL_IMAGE=[yes|no|1|True]          Example: make -DWITH_SDL_IMAGE=True

Build and install the :mod:`pygame2.sdlimage` module. This wraps the
SDL_image library. ::

  WITH_SDL_TTF=[yes|no|1|True]            Example: make -DWITH_SDL_TTF=True

Build and install the :mod:`pygame2.sdlttf` module. This wraps the
SDL_ttf library. ::

  WITH_SDL_GFX=[yes|no|1|True]            Example: make -DWITH_SDL_GFX=1

Build and install the :mod:`pygame2.sdlgfx` module. This wraps the
SDL_gfx library. ::

  WITH_PNG=[yes|no|1|True]                Example: make -DWITH_PNG=True

Build with PNG format saving support for
:meth:`pygame2.sdl.video.Surface.save`. ::

  WITH_JPEG=[yes|no|1|True]               Example: make -DWITH_JPEG=False

Build with JPEG format saving support for
:meth:`pygame2.sdl.video.Surface.save`. ::

  WITH_FREETYPE=[yes|no|1|True]           Example: make -DWITH_FREETYPE=False

Build and install the :mod:`pygame2.freetype` module. This wraps the
FreeType2 library. ::

  WITH_PORTMIDI=[yes|no|1|True]           Example: make -DWITH_PORTMIDI=False

Build and install the :mod:`pygame2.pypm` and :mod:`pygame2.midi` modules. This
wraps the portmidi library and gives access to the :mod:`pygame2.midi`
module. ::

  WITH_OPENAL=[yes|no|1|True]             Example: make -DWITH_OPENAL=False
    
Build and install the :mod:`pygame2.openal` module. This wraps the OpenAL
library and gives access to the :mod:`pygame2.openal` module.
