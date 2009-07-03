"""
Pygame2 is a set of Python modules designed for writing games.
It is written on top of the excellent SDL library. This allows you
to create fully featured games and multimedia programs in the python
language. The package is highly portable, with games running on
Windows, MacOS X, *BSD, Linux and others.
"""

import os

# Manipulate the PATH environment, so that the DLLs are loaded correctly.
path = os.path.dirname (os.path.abspath (__file__))
os.environ['PATH'] += ";%s;%s" % (path,  os.path.join (path, "dll"))

DLLPATH = os.path.join (path, "dll")

__version__ = "2.0.0-alpha3"
version_info = (2, 0, 0, "alpha3")

from pygame2.base import *
