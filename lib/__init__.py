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
Windows, MacOS, OSX, BeOS, FreeBSD, IRIX, and Linux.
"""

import sys, os
if sys.platform=='darwin':
    # this may change someday, but we want to chdir to where our file is if we're in / for no
    # good reason..
    if (os.getcwd() == '/') and len(sys.argv):
        os.chdir(os.path.split(sys.argv[0])[0])
    else:
        argv0=''
        if len(sys.argv): argv0=sys.argv[0]
        print "WARNING!  Running pygame apps from any method other than through python.app (aka through the finder or launchservices) is UNSUPPORTED!"
        print "          If you insist on using the terminal, type \"open %s\", and hold down the option key if you need to" % (argv0)
        print "          specify additional command line arguments.  A dialog box will pop up and make you happy, I promise."
        print ""
        print "          I sure hope you ran as \"%s %s\" exactly, otherwise you will really have problems."%(sys.executable,' '.join(sys.argv))
        print "          WindowServer doesn't like what you're doing as is, and it gets really funky if you run things from the path for whatever reason."
        print ""
        # not ready for prime time yet, it just rewrites the commandline so windowserver can pick it up
        #import pygame.macosx


#we need to import like this, each at a time. the cleanest way to import
#our modules is with the import command (not the __import__ function)

#first, the "required" modules
from pygame.base import *
from pygame.constants import *
from pygame.version import *
from pygame.rect import Rect
import pygame.rwobject
import pygame.surflock
__version__ = ver

#next, the "standard" modules
#we still allow them to be missing for stripped down pygame distributions
try: import pygame.cdrom
except (ImportError,IOError):cdrom=None

try: import pygame.cursors
except (ImportError,IOError):cursors=None

try: import pygame.display
except (ImportError,IOError):display=None

try: import pygame.draw
except (ImportError,IOError):draw=None

try: import pygame.event
except (ImportError,IOError):event=None

try: import pygame.image
except (ImportError,IOError):image=None

try: import pygame.joystick
except (ImportError,IOError):joystick=None

try: import pygame.key
except (ImportError,IOError):key=None

try: import pygame.mouse
except (ImportError,IOError):mouse=None

try: import pygame.sprite
except (ImportError,IOError):sprite=None

try: from pygame.surface import *
except (ImportError,IOError):Surface = lambda:Missing_Function

try: import pygame.time
except (ImportError,IOError):time=None

try: import pygame.transform
except (ImportError,IOError):transform=None

#lastly, the "optional" pygame modules
try: import pygame.font
except (ImportError,IOError):font=None

try: import pygame.mixer
except (ImportError,IOError):mixer=None

try: import pygame.movie
except (ImportError,IOError):movie=None

try: import pygame.surfarray
except (ImportError,IOError):surfarray=None

#there's also a couple "internal" modules not needed
#by users, but putting them here helps "dependency finder"
#programs get everything they need (like py2exe)
try: import pygame.imageext; del imageext
except (ImportError,IOError):pass

try: import pygame.mixer_music; del mixer_music
except (ImportError,IOError):pass


#cleanup namespace
del pygame, os, sys, rwobject, surflock
