##    pygame - Python Game Library
##    Copyright (C) 2000-2001  Pete Shinners
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##    Pete Shinners
##    pete@shinners.org
"""Pygame is a set of Python modules designed for writing games.
It is written on top of the excellent SDL library. This allows you
to create fully featured games and multimedia programs in the python
language. The package is highly portable, with games running on
Windows, MacOS, OS X, BeOS, FreeBSD, IRIX, and Linux.
"""

import sys, os, string
class MissingModule:
    def __init__(self, name, info='', urgent=0):
        self.name = name
        self.info = str(info)
        self.urgent = urgent
        if urgent:
            self.warn()

    def __getattr__(self, var):
        if not self.urgent:
            self.warn()
            self.urgent = 1
        MissingPygameModule = "%s module not available" % self.name
        raise NotImplementedError, MissingPygameModule

    def __nonzero__(self):
        return 0

    def warn(self):
        if self.urgent: type = 'import'
        else: type = 'use'
        message = '%s %s: %s' % (type, self.name, self.info)
        try:
            import warnings
            if self.urgent: level = 4
            else: level = 3
            warnings.warn(message, RuntimeWarning, level)
        except ImportError:
            print message



#we need to import like this, each at a time. the cleanest way to import
#our modules is with the import command (not the __import__ function)

#first, the "required" modules
from pygame.base import *
from pygame.constants import *
from pygame.version import *
from pygame.rect import Rect
import pygame.rwobject
import pygame.surflock
import pygame.color
Color = color.Color
__version__ = ver

#next, the "standard" modules
#we still allow them to be missing for stripped down pygame distributions
try: import pygame.cdrom
except (ImportError,IOError), msg:cdrom=MissingModule("cdrom", msg, 1)

try: import pygame.cursors
except (ImportError,IOError), msg:cursors=MissingModule("cursors", msg, 1)

try: import pygame.display
except (ImportError,IOError), msg:display=MissingModule("display", msg, 1)

try: import pygame.draw
except (ImportError,IOError), msg:draw=MissingModule("draw", msg, 1)

try: import pygame.event
except (ImportError,IOError), msg:event=MissingModule("event", msg, 1)

try: import pygame.image
except (ImportError,IOError), msg:image=MissingModule("image", msg, 1)

try: import pygame.joystick
except (ImportError,IOError), msg:joystick=MissingModule("joystick", msg, 1)

try: import pygame.key
except (ImportError,IOError), msg:key=MissingModule("key", msg, 1)

try: import pygame.mouse
except (ImportError,IOError), msg:mouse=MissingModule("mouse", msg, 1)

try: import pygame.sprite
except (ImportError,IOError), msg:sprite=MissingModule("sprite", msg, 1)

try: from pygame.surface import *
except (ImportError,IOError):Surface = lambda:Missing_Function

try: import pygame.time
except (ImportError,IOError), msg:time=MissingModule("time", msg, 1)

try: import pygame.transform
except (ImportError,IOError), msg:transform=MissingModule("transform", msg, 1)

#lastly, the "optional" pygame modules
try:
    import pygame.font
    import pygame.sysfont
    pygame.font.SysFont = pygame.sysfont.SysFont
    pygame.font.get_fonts = pygame.sysfont.get_fonts
    pygame.font.match_font = pygame.sysfont.match_font
except (ImportError,IOError), msg:font=MissingModule("font", msg, 0)

try: import pygame.mixer
except (ImportError,IOError), msg:mixer=MissingModule("mixer", msg, 0)

try: import pygame.movie
except (ImportError,IOError), msg:movie=MissingModule("movie", msg, 0)

try: import pygame.surfarray
except (ImportError,IOError), msg:surfarray=MissingModule("surfarray", msg, 0)

try: import pygame.sndarray
except (ImportError,IOError), msg:sndarray=MissingModule("sndarray", msg, 0)

#there's also a couple "internal" modules not needed
#by users, but putting them here helps "dependency finder"
#programs get everything they need (like py2exe)
try: import pygame.imageext; del pygame.imageext
except (ImportError,IOError):pass

try: import pygame.mixer_music; del pygame.mixer_music
except (ImportError,IOError):pass


#make Rects pickleable
import copy_reg
def __rect_constructor(x,y,w,h):
	return Rect(x,y,w,h)
def __rect_reduce(r):
	assert type(r) == Rect
	return __rect_constructor, (r.x, r.y, r.w, r.h)
copy_reg.pickle(Rect, __rect_reduce, __rect_constructor)

if sys.platform == 'darwin':
    import MacOS
    if not MacOS.WMAvailable():
        raise ImportError, "Can not access the window manager, use bundlebuilder or execute with the pythonw script"
    del MacOS
    if (os.getcwd() == '/') and len(sys.argv):
        os.chdir(os.path.split(sys.argv[0])[0])
    def init(_init=init):
        try:
            import AppKit
        except ImportError:
            raise ImportError, "PyObjC is required for pygame on OS X"
        if not AppKit.NSApp():
            # Set up application
            import macosx
        _init()

#cleanup namespace
del pygame, os, sys, rwobject, surflock, MissingModule, copy_reg
