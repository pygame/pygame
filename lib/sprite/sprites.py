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

"""Sprite objects"""

import pygame2.compat
pygame2.compat.deprecation \
    ("The sprite package is deprecated and will change in future versions")

class Sprite(object):
    """simple base class for visible game objects
    pygame2.sprite.Sprite(*groups): return Sprite

    The base class for visible game objects. Derived classes will want to 
    override the Sprite.update() and assign a Sprite.image and 
    Sprite.rect attributes.  The initializer can accept any number of 
    Group instances to be added to.

    When subclassing the Sprite, be sure to call the base initializer before
    adding the Sprite to Groups.
    """

    def __init__(self, *groups):
        self.__g = {} # The groups the sprite is in
        if groups:
            self.add (groups)

    def add(self, *groups):
        """add the sprite to groups
        Sprite.add(*groups): return None

        Any number of Group instances can be passed as arguments. The 
        Sprite will be added to the Groups it is not already a member of.
        """
        has = self.__g.has_key
        for group in groups:
            if hasattr(group, '_spritegroup'):
                if not has(group):
                    group.add_internal(self)
                    self.add_internal(group)
            else: self.add(*group)

    def remove(self, *groups):
        """remove the sprite from groups
        Sprite.remove(*groups): return None

        Any number of Group instances can be passed as arguments. The
        Sprite will be removed from the Groups it is currently a member
        of.
        """
        has = self.__g.has_key
        for group in groups:
            if hasattr(group, '_spritegroup'):
                if has(group):
                    group.remove_internal(self)
                    self.remove_internal(group)
            else: self.remove(*group)

    def add_internal(self, group):
        """Adds a dummy group"""
        self.__g[group] = 0

    def remove_internal(self, group):
        """Removes a group"""
        del self.__g[group]

    def update(self, *args):
        """method to control sprite behavior
        Sprite.update(*args):

        The default implementation of this method does nothing; it's just a
        convenient "hook" that you can override. This method is called by
        Group.update() with whatever arguments you give it.

        There is no need to use this method if not using the convenience 
        method by the same name in the Group class.
        """
        pass

    def kill(self):
        """remove the Sprite from all Groups
        Sprite.kill(): return None

        The Sprite is removed from all the Groups that contain it. This
        won't change anything about the state of the Sprite. It is
        possible to continue to use the Sprite after this method has
        been called, including adding it to Groups.
        """
        for c in self.__g.keys():
            c.remove_internal(self)
        self.__g.clear()

    def groups(self):
        """list of Groups that contain this Sprite
        Sprite.groups(): return group_list

        Return a list of all the Groups that contain this Sprite.
        """
        return self.__g.keys()

    def alive(self):
        """does the sprite belong to any groups
        Sprite.alive(): return bool

        Returns True when the Sprite belongs to one or more Groups.
        """
        return (len(self.__g) != 0)

    def __repr__(self):
        return "<%s sprite(in %d groups)>" % (self.__class__.__name__,
                                              len (self.__g))


class DirtySprite(Sprite):
    """a more featureful subclass of Sprite with more attributes
    pygame2.sprite.DirtySprite(*groups): return DirtySprite 

    Extra DirtySprite attributes with their default values:

    dirty = 1
        if set to 1, it is repainted and then set to 0 again 
        if set to 2 then it is always dirty ( repainted each frame, 
        flag is not reset)
        0 means that it is not dirty and therefor not repainted again

    blendmode = 0
        its the special_flags argument of blit, blendmodes

    source_rect = None
        source rect to use, remember that it is relative to 
        topleft (0,0) of self.image

    visible = 1
        normally 1, if set to 0 it will not be repainted 
        (you must set it dirty too to be erased from screen)

    layer = 0
        (READONLY value, it is read when adding it to the 
        LayeredUpdates, for details see doc of LayeredUpdates)
    """
    
    def __init__(self, *groups):
        
        self.dirty = 1
        self.blendmode = 0  # pygame 1.8, reffered as special_flags in 
                            # the documentation of blit 
        self._visible = 1
        self._layer = 0    # READ ONLY by LayeredUpdates or LayeredDirty
        self.source_rect = None
        Sprite.__init__(self, *groups)
        
    def _set_visible(self, val):
        """set the visible value (0 or 1) and makes the sprite dirty"""
        self._visible = val
        if self.dirty < 2:
            self.dirty = 1

    def _get_visible(self):
        """returns the visible value of that sprite"""
        return self._visible

    visible = property(lambda self: self._get_visible (),
                       lambda self, value:self._set_visible(value),
                       doc="you can make this sprite disappear without" +
                       "removing it from the group," +
                       "values 0 for invisible and 1 for visible")
        
    def __repr__(self):
        return "<%s DirtySprite(in %d groups)>" % (self.__class__.__name__,
                                                   len (self.groups ()))
