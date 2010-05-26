##    pygame - Python Game Library
##    Copyright (C) 2000-2003, 2007  Pete Shinners
##              (C) 2004 Joe Wreschnig
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

"""Specialized rendering-enabled sprite groups"""

import pygame2.compat
pygame2.compat.deprecation \
    ("The sprite package is deprecated and will change in future versions")

from pygame2.sprite.groups import Group

RenderPlain = Group
RenderClear = Group

class RenderUpdates(Group):
    """Group class that tracks dirty updates
    pygame2.sprite.RenderUpdates(*sprites): return RenderUpdates

    This class is derived from pygame2.sprite.Group(). It has an extended draw()
    method that tracks the changed areas of the screen.
    """
    
    def draw(self, surface):
        spritedict = self.spritedict
        surface_blit = surface.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        for s in self.sprites():
            r = spritedict[s]
            newrect = surface_blit(s.image, s.rect)
            if r is 0:
                dirty_append(newrect)
            else:
                if newrect.colliderect(r):
                    dirty_append(newrect.union(r))
                else:
                    dirty_append(newrect)
                    dirty_append(r)
            spritedict[s] = newrect
        return dirty

class OrderedUpdates(RenderUpdates):
    """RenderUpdates class that draws Sprites in order of addition
    pygame2.sprite.OrderedUpdates(*spites): return OrderedUpdates

    This class derives from pygame2.sprite.RenderUpdates().  It maintains 
    the order in which the Sprites were added to the Group for rendering. 
    This makes adding and removing Sprites from the Group a little 
    slower than regular Groups.
    """

    def __init__(self, *sprites):
        self._spritelist = []
        RenderUpdates.__init__(self, *sprites)

    def sprites(self): 
        return list(self._spritelist)

    def add_internal(self, sprite):
        RenderUpdates.add_internal(self, sprite)
        self._spritelist.append(sprite)

    def remove_internal(self, sprite):
        RenderUpdates.remove_internal(self, sprite)
        self._spritelist.remove(sprite)
