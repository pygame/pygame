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

"""UserRect, python class wrapped around the pygame Rect type. This allows
you to do things like inherit the rectangle object into a sprite class. While
this makes for some neat features, current testing has shown there is a bit of
a peformance penalty. (as opposed to just keeping a Rect value inside the class)
"""

#sadly, we need to hide the rect from the documentation finder for now
import pygame.rect as rectmodule
R = [rectmodule.Rect]
del rectmodule


class UserRect:
    """pygame.UserRect.UserRect() -> class instance
Python class for the pygame Rect type

This python class can be treated exactly like a normal Rect
object. The only difference is it is a real python class
object, not a C extension type.
"""
    def __init__(self, *args):
        try: self.__dict__['rect'] = R[0](*args)
        except TypeError:
            raise TypeError, 'Argument must be rectstyle object'
        for a in dir(self.rect):
            self.__dict__[a] = getattr(self.rect, a)

    def __getattr__(self, name):
        return getattr(self.rect, name)
    def __setattr__(self, name, val):
        if name is 'rect':
            self.__dict__['rect'][:] = val
        else:
            try: setattr(self.__dict__['rect'], name, val)
            except (AttributeError, KeyError): self.__dict__[name] = val
    def __len__(self): return 4
    def __getitem__(self, i): return self.rect[i]
    def __setitem__(self, i, val): self.rect[i] = val
    def __getslice__(self, i, j): return self.rect[i:j]
    def __setslice__(self, i, j, val): self.rect[i:j] = val
    def __nonzero__(self): return nonzero(self.rect)
    def __cmp__(self, o): return cmp(self.rect, o)
    def __repr__(self):
        return '<UserRect(%d, %d, %d, %d)>' % tuple(self.rect)
    def __str__(self):
        return '<UserRect(%d, %d, %d, %d)>' % tuple(self.rect)

    