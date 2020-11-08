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
    sys.path.append(pygame_dir)

# when running under X11, always set the SDL window WM_CLASS to make the
#   window managers correctly match the pygame window.
#elif 'DISPLAY' in os.environ and 'SDL_VIDEO_X11_WMCLASS' not in os.environ:
#    os.environ['SDL_VIDEO_X11_WMCLASS'] = os.path.basename(sys.argv[0])

#import importlib.util
#import sys

"""def alternative_load_dynamic(name, path, file=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return sys.modules[name]
"""

"""def unbulk_dyn_load(name):
    alternative_load_dynamic("pygame." + name, name + ".pyd")    
"""

#import imp

#def unbulk_dyn_load(name):
#    sys.modules["pygame." + name] = imp.load_dynamic("_pygame." + name, name + ".pyd")    

from importlib.machinery import ExtensionFileLoader

def unbulk_dyn_load(name):
    foo = ExtensionFileLoader("pygame." + name, name + ".pyd").load_module()
    sys.modules["pygame." + name] = foo
    return foo

def unbulk_dyn_load_package_name(module_name, package_name, extension_name):
    foo = ExtensionFileLoader(package_name, extension_name).load_module()
    sys.modules[module_name] = foo
    return foo    

# Mandatory
base = unbulk_dyn_load("base")
bufferproxy = unbulk_dyn_load("bufferproxy")
color = unbulk_dyn_load("color")
constants = unbulk_dyn_load("constants")
rect = unbulk_dyn_load("rect")
surflock = unbulk_dyn_load("surflock")
surface = unbulk_dyn_load("surface")

display = unbulk_dyn_load("display")
draw = unbulk_dyn_load("draw")
event = unbulk_dyn_load("event")
fastevent = unbulk_dyn_load("fastevent")
rwobject = unbulk_dyn_load("rwobject")

imageext = unbulk_dyn_load_package_name("imageext", "pygame.imageext", "imageext.pyd")

import pygame.imageext

image = unbulk_dyn_load("image")
joystick = unbulk_dyn_load("joystick")
key = unbulk_dyn_load("key")
mask = unbulk_dyn_load("mask")
math = unbulk_dyn_load("math")
mouse = unbulk_dyn_load("mouse")
newbuffer = unbulk_dyn_load("newbuffer")
pixelarray = unbulk_dyn_load("pixelarray")
pixelcopy = unbulk_dyn_load("pixelcopy")

time = unbulk_dyn_load("time")
transform = unbulk_dyn_load("transform")


# Optional extensions
#alternative_load_dynamic("pygame.ftfont", "_freetype.pyd")
#alternative_load_dynamic("pygame.sprite", "_sprite.pyd")
#imp.load_dynamic("pygame.ftfont", "_freetype.pyd")
#imp.load_dynamic("pygame.sprite", "sprite.pyd")

font = unbulk_dyn_load("font")

#font = unbulk_dyn_load_package_name("pygame.font", "pygame._freetype", "_freetype.pyd")

mixer_music = unbulk_dyn_load("mixer_music")
mixer = unbulk_dyn_load("mixer")

scrap = unbulk_dyn_load("scrap")
gfxdraw = unbulk_dyn_load("gfxdraw")

class MissingModule:
    _NOT_IMPLEMENTED_ = True

    def __init__(self, name, urgent=0):
        self.name = name
        exc_type, exc_msg = sys.exc_info()[:2]
        self.info = str(exc_msg)
        self.reason = "%s: %s" % (exc_type.__name__, self.info)
        self.urgent = urgent
        if urgent:
            self.warn()

    def __getattr__(self, var):
        if not self.urgent:
            self.warn()
            self.urgent = 1
        missing_msg = "%s module not available (%s)" % (self.name, self.reason)
        raise NotImplementedError(missing_msg)

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__

    def warn(self):
        msg_type = 'import' if self.urgent else 'use'
        message = '%s %s: %s\n(%s)' % (msg_type, self.name, self.info, self.reason)
        try:
            import warnings
            level = 4 if self.urgent else 3
            warnings.warn(message, RuntimeWarning, level)
        except ImportError:
            print(message)


# we need to import like this, each at a time. the cleanest way to import
# our modules is with the import command (not the __import__ function)

# first, the "required" modules
from pygame.base import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.constants import *  # now has __all__ pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.version import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.rect import Rect
from pygame.compat import PY_MAJOR_VERSION
from pygame.rwobject import encode_string, encode_file_path
import pygame.surflock
import pygame.color 
Color = color.Color
import pygame.bufferproxy 
BufferProxy = bufferproxy.BufferProxy
import pygame.math 

Vector2 = math.Vector2
Vector3 = math.Vector3

import pygame.sprite

#import pygame.mixer

__version__ = ver


# have dependencies on base import

import _sdl2

sprite = unbulk_dyn_load_package_name("sprite", "pygame._sprite", "_sprite.pyd")

from pygame.surface import Surface, SurfaceType

import pygame.sysfont

font.SysFont = pygame.sysfont.SysFont
font.get_fonts = pygame.sysfont.get_fonts
font.match_font = pygame.sysfont.match_font

"""
# next, the "standard" modules
# we still allow them to be missing for stripped down pygame distributions
if get_sdl_version() < (2, 0, 0):
    # cdrom only available for SDL 1.2.X
    try:
        import pygame.cdrom
    except (ImportError, IOError):
        cdrom = MissingModule("cdrom", urgent=1)

try:
    import pygame.cursors
except (ImportError, IOError):
    cursors = MissingModule("cursors", urgent=1)

try:
    import pygame.display
except (ImportError, IOError):
    display = MissingModule("display", urgent=1)

try:
    import pygame.draw
except (ImportError, IOError):
    draw = MissingModule("draw", urgent=1)

try:
    import pygame.event
except (ImportError, IOError):
    event = MissingModule("event", urgent=1)

try:
    import pygame.image
except (ImportError, IOError):
    image = MissingModule("image", urgent=1)

try:
    import pygame.joystick
except (ImportError, IOError):
    joystick = MissingModule("joystick", urgent=1)

try:
    import pygame.key
except (ImportError, IOError):
    key = MissingModule("key", urgent=1)

try:
    import pygame.mouse
except (ImportError, IOError):
    mouse = MissingModule("mouse", urgent=1)

try:
    import pygame.sprite
except (ImportError, IOError):
    sprite = MissingModule("sprite", urgent=1)

try:
    import pygame.threads
except (ImportError, IOError):
    threads = MissingModule("threads", urgent=1)

try:
    import pygame.pixelcopy
except (ImportError, IOError):
    pixelcopy = MissingModule("pixelcopy", urgent=1)


def warn_unwanted_files():
    #warn about unneeded old files

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

    unwanted_files = []
    for f in files:
        unwanted_files.append(os.path.join(install_path, f))

    ask_remove = []
    for f in unwanted_files:
        if os.path.exists(f):
            ask_remove.append(f)

    if ask_remove:
        message = "Detected old file(s).  Please remove the old files:\n"

        for f in ask_remove:
            message += "%s " % f
        message += "\nLeaving them there might break pygame.  Cheers!\n\n"

        try:
            import warnings
            level = 4
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
    time = MissingModule("time", urgent=1)

try:
    import pygame.transform
except (ImportError, IOError):
    transform = MissingModule("transform", urgent=1)

# lastly, the "optional" pygame modules
if 'PYGAME_FREETYPE' in os.environ:
    try:
        import pygame.ftfont as font
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
    mixer = MissingModule("mixer", urgent=0)

try:
    import pygame.movie
except (ImportError, IOError):
    movie = MissingModule("movie", urgent=0)

# try:
#     import pygame.movieext
# except (ImportError,IOError):
#     movieext=MissingModule("movieext", urgent=0)

try:
    import pygame.scrap
except (ImportError, IOError):
    scrap = MissingModule("scrap", urgent=0)

try:
    import pygame.surfarray
except (ImportError, IOError):
    surfarray = MissingModule("surfarray", urgent=0)

try:
    import pygame.sndarray
except (ImportError, IOError):
    sndarray = MissingModule("sndarray", urgent=0)

try:
    import pygame.fastevent
except (ImportError, IOError):
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
    #some additional imports that py2app/py2exe will want to see
    import atexit
    import numpy
    import OpenGL.GL
    import pygame.macosx
    import pygame.bufferproxy
    import pygame.colordict
    import pygame._view

# make Rects pickleable
if PY_MAJOR_VERSION >= 3:
    import copyreg as copy_reg
else:
    import copy_reg


def __rect_constructor(x, y, w, h):
    return Rect(x, y, w, h)


def __rect_reduce(r):
    assert type(r) == Rect
    return __rect_constructor, (r.x, r.y, r.w, r.h)
copy_reg.pickle(Rect, __rect_reduce, __rect_constructor)


# make Colors pickleable
def __color_constructor(r, g, b, a):
    return Color(r, g, b, a)


def __color_reduce(c):
    assert type(c) == Color
    return __color_constructor, (c.r, c.g, c.b, c.a)
copy_reg.pickle(Color, __color_reduce, __color_constructor)

"""

# Thanks for supporting pygame. Without support now, there won't be pygame later.
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    print('pygame {} (SDL {}.{}.{}, python {}.{}.{})'.format(
        ver, *get_sdl_version() + sys.version_info[0:3]
    ))
    print('Hello from the pygame community. https://www.pygame.org/contribute.html')

del os, sys
