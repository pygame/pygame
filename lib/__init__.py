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
    os.environ['PATH'] = os.environ['PATH'] + ';' + pygame_dir
    # Respect existing SDL_VIDEODRIVER setting if it has been set
    if 'SDL_VIDEODRIVER' not in os.environ:

        # If the Windows version is 95/98/ME and DirectX 5 or greater is
        # installed, then use the directx driver rather than the default
        # windib driver.

        # http://docs.python.org/lib/module-sys.html
        # 0 (VER_PLATFORM_WIN32s)          Win32s on Windows 3.1
        # 1 (VER_PLATFORM_WIN32_WINDOWS)   Windows 95/98/ME
        # 2 (VER_PLATFORM_WIN32_NT)        Windows NT/2000/XP
        # 3 (VER_PLATFORM_WIN32_CE)        Windows CE
        if sys.getwindowsversion()[0] == 1:

            import _winreg

            try:

                # Get DirectX version from registry
                key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,
                                      'SOFTWARE\\Microsoft\\DirectX')
                dx_version_string = _winreg.QueryValueEx(key, 'Version')
                key.Close()

                # Set video driver to directx if DirectX 5 or better is
                # installed.
                # To interpret DirectX version numbers, see this page:
                # http://en.wikipedia.org/wiki/DirectX#Releases
                minor_dx_version = int(dx_version_string.split('.')[1])
                if minor_dx_version >= 5:
                    os.environ['SDL_VIDEODRIVER'] = 'directx'

                # Clean up namespace
                del key, dx_version_string, minor_dx_version

            except:
                pass

            # Clean up namespace
            del _winreg

# when running under X11, always set the SDL window WM_CLASS to make the
#   window managers correctly match the pygame window.
elif 'DISPLAY' in os.environ and 'SDL_VIDEO_X11_WMCLASS' not in os.environ:
    os.environ['SDL_VIDEO_X11_WMCLASS'] = os.path.basename(sys.argv[0])


class MissingModule:
    _NOT_IMPLEMENTED_ = True

    def __init__(self, name, info='', urgent=0):
        self.name = name
        self.info = str(info)
        try:
            exc = sys.exc_info()
            if exc[0] != None:
                self.reason = "%s: %s" % (exc[0].__name__, str(exc[1]))
            else:
                self.reason = ""
        finally:
            del exc
        self.urgent = urgent
        if urgent:
            self.warn()

    def __getattr__(self, var):
        if not self.urgent:
            self.warn()
            self.urgent = 1
        MissingPygameModule = "%s module not available" % self.name
        if self.reason:
            MissingPygameModule += "\n(%s)" % self.reason
        raise NotImplementedError(MissingPygameModule)

    def __nonzero__(self):
        return 0

    def warn(self):
        if self.urgent:
            type = 'import'
        else:
            type = 'use'
        message = '%s %s: %s' % (type, self.name, self.info)
        if self.reason:
            message += "\n(%s)" % self.reason
        try:
            import warnings
            if self.urgent:
                level = 4
            else:
                level = 3
            warnings.warn(message, RuntimeWarning, level)
        except ImportError:
            print (message)


# we need to import like this, each at a time. the cleanest way to import
# our modules is with the import command (not the __import__ function)

# first, the "required" modules
from pygame.base import *
from pygame.constants import *
from pygame.version import *
from pygame.rect import Rect
from pygame.compat import geterror, PY_MAJOR_VERSION
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
    _import_failed = False
    try:
        import pygame.cdrom
    except (ImportError, IOError):
        _import_failed = geterror()
    if _import_failed:
        cdrom = MissingModule("cdrom", geterror(), 1)

_import_failed = False
try:
    import pygame.cursors
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    cursors = MissingModule("cursors", geterror(), 1)

_import_failed = False
try:
    import pygame.display
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    display = MissingModule("display", geterror(), 1)

_import_failed = False
try:
    import pygame.draw
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    draw = MissingModule("draw", _import_failed, 1)


_import_failed = False
try:
    import pygame.event
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    event = MissingModule("event", geterror(), 1)

_import_failed = False
try:
    import pygame.image
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    image = MissingModule("image", geterror(), 1)

_import_failed = False
try:
    import pygame.joystick
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    joystick = MissingModule("joystick", geterror(), 1)

_import_failed = False
try:
    import pygame.key
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    key = MissingModule("key", geterror(), 1)

_import_failed = False
try:
    import pygame.mouse
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    mouse = MissingModule("mouse", geterror(), 1)

_import_failed = False
try:
    import pygame.sprite
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    sprite = MissingModule("sprite", geterror(), 1)


_import_failed = False
try:
    import pygame.threads
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    threads = MissingModule("threads", geterror(), 1)

_import_failed = False
try:
    import pygame.pixelcopy
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    pixelcopy = MissingModule("pixelcopy", geterror(), 1)


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
            print (message)


# disable, because we hopefully don't need it.
# warn_unwanted_files()


_import_failed = False
try:
    from pygame.surface import *
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    Surface = lambda: Missing_Function


_import_failed = False
try:
    import pygame.mask
    from pygame.mask import Mask
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    Mask = lambda: Missing_Function

_import_failed = False
try:
    from pygame.pixelarray import *
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    PixelArray = lambda: Missing_Function

_import_failed = False
try:
    from pygame.overlay import *
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    Overlay = lambda: Missing_Function

_import_failed = False
try:
    import pygame.time
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    time = MissingModule("time", geterror(), 1)

_import_failed = False
try:
    import pygame.transform
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    transform = MissingModule("transform", geterror(), 1)

_import_failed = False
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
    _import_failed = geterror()
if _import_failed:
    font = MissingModule("font", geterror(), 0)

_import_failed = False
# try and load pygame.mixer_music before mixer, for py2app...
try:
    import pygame.mixer_music
    #del pygame.mixer_music
    #print ("NOTE2: failed importing pygame.mixer_music in lib/__init__.py")
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    pass

_import_failed = False
try:
    import pygame.mixer
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    mixer = MissingModule("mixer", geterror(), 0)

_import_failed = False
try:
    import pygame.movie
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    movie = MissingModule("movie", geterror(), 0)

# try: import pygame.movieext
# except (ImportError,IOError):movieext=MissingModule("movieext",
# geterror(), 0)

_import_failed = False
try:
    import pygame.scrap
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    scrap = MissingModule("scrap", geterror(), 0)

_import_failed = False
try:
    import pygame.surfarray
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    surfarray = MissingModule("surfarray", geterror(), 0)

_import_failed = False
try:
    import pygame.sndarray
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    sndarray = MissingModule("sndarray", geterror(), 0)

_import_failed = False
try:
    import pygame.fastevent
except (ImportError, IOError):
    _import_failed = geterror()
if _import_failed:
    fastevent = MissingModule("fastevent", geterror(), 0)

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


# cleanup namespace
del pygame, os, sys, surflock, MissingModule, copy_reg, geterror, PY_MAJOR_VERSION, _import_failed

# Thanks for supporting pygame. Without support now, there won't be pygame later.
print('pygame %s' % ver)
print('Hello from the pygame community. https://www.pygame.org/contribute.html')
