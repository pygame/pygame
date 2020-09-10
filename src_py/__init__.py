# coding: ascii
# pygame - Python Game Library
# Copyright (C) 2000-2001  Pete Shinners
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the Free
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Pete Shinners
# pete@shinners.org
"""Pygame is a set of Python modules designed for writing games.
It is written on top of the excellent SDL library. This allows you
to create fully featured games and multimedia programs in the python
language. The package is highly portable, with games running on
Windows, MacOS, OS X, BeOS, FreeBSD, IRIX, and Linux."""

import sys
import os

# Choose Windows display driver
if os.name == 'nt':
    #pypy does not find the dlls, so we add package folder to PATH.
    pygame_dir = os.path.split(__file__)[0]
    # Adds pygame to the enviroment variables
    os.environ['PATH'] = os.environ['PATH'] + ';' + pygame_dir

# when running under X11, always set the SDL window WM_CLASS to make the
#   window managers correctly match the pygame window.
elif 'DISPLAY' in os.environ and 'SDL_VIDEO_X11_WMCLASS' not in os.environ:
    os.environ['SDL_VIDEO_X11_WMCLASS'] = os.path.basename(sys.argv[0])


class MissingModule:
    _NOT_IMPLEMENTED_ = True

    def __init__(self, name, urgent=0):
        self.name = name
        exc_type, exc_msg = sys.exc_info()[:2]
        self.info = str(exc_msg)
        self.reason = "%s: %s" % (exc_type.__name__, self.info)
        self.urgent = urgent
        # If urgent is above 0
        if urgent:
            # Warn the user of the missing module
            self.warn()

    def __getattr__(self, var):
        if not self.urgent:
            self.warn()
            self.urgent = 1
        missing_msg = "%s module not available (%s)" % (self.name, self.reason)
        raise NotImplementedError(missing_msg)

    def __nonzero__(self):
        return 0

    def warn(self):
        # If there is a missing module this function is called
        msg_type = 'import' if self.urgent else 'use'
        # This defines the message that will be shown to the user
        message = '%s %s: %s\n(%s)' % (msg_type, self.name, self.info, self.reason)
        try:
            import warnings
            level = 4 if self.urgent else 3
            # This warns the user of the missing module
            warnings.warn(message, RuntimeWarning, level)
        except ImportError:
            print(message)


# we need to import like this, each at a time. the cleanest way to import
# our modules is with the import command (not the __import__ function)

# first, the "required" modules
from pygame.base import *
from pygame.constants import *  # now has __all__
from pygame.version import *
from pygame.rect import Rect
from pygame.compat import PY_MAJOR_VERSION
from pygame.rwobject import encode_string, encode_file_path
import pygame.surflock
import pygame.color
Color = color.Color
import pygame.bufferproxy
BufferProxy = bufferproxy.BufferProxy
import pygame.math
Vector2 = pygame.math.Vector2
Vector3 = pygame.math.Vector3

__version__ = ver

# next, the "standard" modules
# we still allow them to be missing for stripped down pygame distributions
if get_sdl_version() < (2, 0, 0):
    # cdrom only available for SDL 1.2.X
    try:
        import pygame.cdrom
    except (ImportError, IOError):
        # This creates a missing module object
        # With the name of cdrom
        # And the urgency level 1
        cdrom = MissingModule("cdrom", urgent=1)

try:
    import pygame.cursors
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of cursors
    # And the urgency level 1
    cursors = MissingModule("cursors", urgent=1)

try:
    import pygame.display
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of display
    # And the urgency level 1
    display = MissingModule("display", urgent=1)

try:
    import pygame.draw
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of draw
    # And the urgency level 1
    draw = MissingModule("draw", urgent=1)

try:
    import pygame.event
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of event
    # And the urgency level 1
    event = MissingModule("event", urgent=1)

try:
    import pygame.image
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of image
    # And the urgency level 1
    image = MissingModule("image", urgent=1)

try:
    import pygame.joystick
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of joystick
    # And the urgency level 1
    joystick = MissingModule("joystick", urgent=1)

try:
    import pygame.key
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of key
    # And the urgency level 1
    key = MissingModule("key", urgent=1)

try:
    import pygame.mouse
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of mouse
    # And the urgency level 1
    mouse = MissingModule("mouse", urgent=1)

try:
    import pygame.sprite
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of sprite
    # And the urgency level 1
    sprite = MissingModule("sprite", urgent=1)

try:
    import pygame.threads
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of threads
    # And the urgency level 1
    threads = MissingModule("threads", urgent=1)

try:
    import pygame.pixelcopy
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of pixelcopy
    # And the urgency level 1
    pixelcopy = MissingModule("pixelcopy", urgent=1)


def warn_unwanted_files():
    """warn about unneeded old files"""

    # a temporary hack to warn about camera.so and camera.pyd.
    install_path = os.path.split(pygame.base.__file__)[0]
    extension_ext = os.path.splitext(pygame.base.__file__)[1]

    # here are the .so/.pyd files we need to ask to remove.
    ext_to_remove = ["camera"]

    # here are the .py/.pyo/.pyc files we need to ask to remove.
    py_to_remove = ["color"]

    # Don't warn on Symbian. The color.py is used as a wrapper.
    if os.name == "e32":
        py_to_remove = []

    # See if any of the files are there.
    extension_files = ["%s%s" % (x, extension_ext) for x in ext_to_remove]

    py_files = ["%s%s" % (x, py_ext)
                for py_ext in [".py", ".pyc", ".pyo"]
                for x in py_to_remove]

    files = py_files + extension_files

    # Creates a list of unwanted files
    unwanted_files = []
    for f in files:
        unwanted_files.append(os.path.join(install_path, f))

    ask_remove = []
    for f in unwanted_files:
        if os.path.exists(f):
            ask_remove.append(f)

    if ask_remove:
        # Setting a message if there are old files
        message = "Detected old file(s).  Please remove the old files:\n"

        for f in ask_remove:
            message += "%s " % f
        message += "\nLeaving them there might break pygame.  Cheers!\n\n"

        try:
            import warnings
            level = 4
            # Warn the user of the old files
            warnings.warn(message, RuntimeWarning, level)
        except ImportError:
            print(message)


# disable, because we hopefully don't need it.
# warn_unwanted_files()


try:
    from pygame.surface import Surface, SurfaceType
except (ImportError, IOError):
    Surface = lambda: Missing_Function


try:
    import pygame.mask
    from pygame.mask import Mask
except (ImportError, IOError):
    Mask = lambda: Missing_Function

try:
    from pygame.pixelarray import PixelArray
except (ImportError, IOError):
    PixelArray = lambda: Missing_Function

try:
    from pygame.overlay import Overlay
except (ImportError, IOError):
    Overlay = lambda: Missing_Function

try:
    import pygame.time
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of time
    # And the urgency level 1
    time = MissingModule("time", urgent=1)

try:
    import pygame.transform
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of transform
    # And the urgency level 1
    transform = MissingModule("transform", urgent=1)

# lastly, the "optional" pygame modules
if 'PYGAME_FREETYPE' in os.environ:
    try:
        import pygame.ftfont as font
        # Map the pygame.font module to font
        sys.modules['pygame.font'] = font
    except (ImportError, IOError):
        pass
try:
    import pygame.font
    import pygame.sysfont
    pygame.font.SysFont = pygame.sysfont.SysFont
    pygame.font.get_fonts = pygame.sysfont.get_fonts
    pygame.font.match_font = pygame.sysfont.match_font
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of font
    # And the urgency level 0
    font = MissingModule("font", urgent=0)

# try and load pygame.mixer_music before mixer, for py2app...
try:
    import pygame.mixer_music
    #del pygame.mixer_music
    #print ("NOTE2: failed importing pygame.mixer_music in lib/__init__.py")
except (ImportError, IOError):
    pass

try:
    import pygame.mixer
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of mixer
    # And the urgency level 0
    mixer = MissingModule("mixer", urgent=0)

try:
    import pygame.movie
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of movie
    # And the urgency level 0
    movie = MissingModule("movie", urgent=0)

# try:
#     import pygame.movieext
# except (ImportError,IOError):
#     movieext=MissingModule("movieext", urgent=0)

try:
    import pygame.scrap
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of scrap
    # And the urgency level 0
    scrap = MissingModule("scrap", urgent=0)

try:
    import pygame.surfarray
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of surfarray
    # And the urgency level 0
    surfarray = MissingModule("surfarray", urgent=0)

try:
    import pygame.sndarray
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of sndarray
    # And the urgency level 0
    sndarray = MissingModule("sndarray", urgent=0)

try:
    import pygame.fastevent
except (ImportError, IOError):
    # This creates a missing module object
    # With the name of fastevent
    # And the urgency level 0
    fastevent = MissingModule("fastevent", urgent=0)

# there's also a couple "internal" modules not needed
# by users, but putting them here helps "dependency finder"
# programs get everything they need (like py2exe)
try:
    import pygame.imageext
    del pygame.imageext
except (ImportError, IOError):
    pass


def packager_imports():
    """some additional imports that py2app/py2exe will want to see"""
    import atexit
    import numpy
    import OpenGL.GL
    import pygame.macosx
    import pygame.bufferproxy
    import pygame.colordict
    import pygame._view

# make Rects pickleable
if PY_MAJOR_VERSION >= 3: # If python version is above or equal to 3
    import copyreg as copy_reg
else:
    import copy_reg


def __rect_constructor(x, y, w, h):
    # Returns a rect object with the coordinates and sizes passed in
    return Rect(x, y, w, h)


def __rect_reduce(r):
    assert type(r) == Rect
    return __rect_constructor, (r.x, r.y, r.w, r.h)
copy_reg.pickle(Rect, __rect_reduce, __rect_constructor)


# make Colors pickleable
def __color_constructor(r, g, b, a):
    # Returns a color object with the colors passed in
    return Color(r, g, b, a)


def __color_reduce(c):
    assert type(c) == Color
    return __color_constructor, (c.r, c.g, c.b, c.a)
copy_reg.pickle(Color, __color_reduce, __color_constructor)


# Thanks for supporting pygame. Without support now, there won't be pygame later.
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    print('pygame {} (SDL {}.{}.{}, python {}.{}.{})'.format(
        ver, *get_sdl_version() + sys.version_info[0:3]
    ))
    print('Hello from the pygame community. https://www.pygame.org/contribute.html')


# cleanup namespace
del pygame, os, sys, surflock, MissingModule, copy_reg, PY_MAJOR_VERSION
