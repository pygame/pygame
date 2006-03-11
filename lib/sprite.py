##    pygame - Python Game Library
##    Copyright (C) 2000-2003  Pete Shinners
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
##    Pete Shinners
##    pete@shinners.org

"""
This module contains a base class for sprite objects. Also
several different group classes you can use to store and
identify the sprites. Some of the groups can be used to
draw the sprites they contain. Lastly there are a handful
of collision detection functions to help you quickly find
intersecting sprites in a group.

The way the groups are designed, it is very efficient at
adding and removing sprites from groups. This makes the
groups a perfect use for cataloging or tagging different
sprites. instead of keeping an identifier or type as a
member of a sprite class, just store the sprite in a
different set of groups. this ends up being a much better
way to loop through, find, and effect different sprites.
It is also a very quick to test if a sprite is contained
in a given group.

You can manage the relationship between groups and sprites
from both the groups and the actual sprite classes. Both
have add() and remove() functions that let you add sprites
to groups and groups to sprites. Both have initializing
functions that can accept a list of containers or sprites.

The methods to add and remove sprites from groups are
smart enough to not delete sprites that aren't already part
of a group, and not add sprites to a group if it already
exists. You may also pass a sequence of sprites or groups
to these functions and each one will be used.

While it is possible to design sprite and group classes
that don't derive from the Sprite and AbstractGroup classes
below, it is strongly recommended that you extend those
when you add a Sprite or Group class.
"""

##todo
## a group that holds only the 'n' most recent elements.
## sort of like the GroupSingle class, but holding more
## than one sprite
##
## drawing groups that can 'automatically' store the area
## underneath, so the can "clear" without needing a background
## function. obviously a little slower than normal, but nice
## to use in many situations. (also remember it must "clear"
## in the reverse order that it draws :])
##
## the drawing groups should also be able to take a background
## function, instead of just a background surface. the function
## would take a surface and a rectangle on that surface to erase.
##
## perhaps more types of collision functions? the current two
## should handle just about every need, but perhaps more optimized
## specific ones that aren't quite so general but fit into common
## specialized cases.

class Sprite(object):
    """The base class for your visible game objects.
       The sprite class is meant to be used as a base class
       for the objects in your game. It just provides functions
       to maintain itself in different groups.

       You can initialize a sprite by passing it a group or sequence
       of groups to be contained in.

       When you subclass Sprite, you must call this
       pygame.sprite.Sprite.__init__(self) before you add the sprite
       to any groups, or you will get an error."""

    def __init__(self, *groups):
        self.__g = {} # The groups the sprite is in
        if groups: self.add(groups)

    def add(self, *groups):
        """add(group or list of of groups, ...)
           add a sprite to container

           Add the sprite to a group or sequence of groups."""
        has = self.__g.has_key
        for group in groups:
            if hasattr(group, '_spritegroup'):
                if not has(group):
                    group.add_internal(self)
                    self.add_internal(group)
            else: self.add(*group)

    def remove(self, *groups):
        """remove(group or list of groups, ...)
           remove a sprite from container

           Remove the sprite from a group or sequence of groups."""
        has = self.__g.has_key
        for group in groups:
            if hasattr(group, '_spritegroup'):
                if has(group):
                    group.remove_internal(self)
                    self.remove_internal(group)
            else: self.remove(*group)

    def add_internal(self, group):
        self.__g[group] = 0

    def remove_internal(self, group):
        del self.__g[group]

    def update(self, *args):
        pass

    def kill(self):
        """kill()
           remove this sprite from all groups

           Removes the sprite from all the groups that contain
           it. The sprite still exists after calling this,
           so you could use it to remove a sprite from all groups,
           and then add it to some other groups."""
        for c in self.__g.keys():
            c.remove_internal(self)
        self.__g.clear()

    def groups(self):
        """groups() -> list of groups
           list used sprite containers

           Returns a list of all the groups that contain this
           sprite. These are not returned in any meaningful order."""
        return self.__g.keys()

    def alive(self):
        """alive() -> bool
           check to see if the sprite is in any groups

           Returns true if this sprite is a member of any groups."""
        return (len(self.__g) != 0)

    def __repr__(self):
        return "<%s sprite(in %d groups)>" % (self.__class__.__name__, len(self.__g))

class AbstractGroup(object):
    """A base for containers for sprites. It does everything
       needed to behave as a normal group. You can easily inherit
       a new group class from this, or the other groups below,
       if you want to add more features.

       Any AbstractGroup-derived sprite groups act like sequences,
       and support iteration, len, and so on."""

    # dummy val to identify sprite groups, and avoid infinite recursion.
    _spritegroup = True

    def __init__(self):
        self.spritedict = {}
        self.lostsprites = []

    def sprites(self):
        """sprites()
           get a list of sprites in the group

           Returns an object that can be looped over with a 'for' loop.
           (For now it is always a list, but newer version of Python
           could return different iterators.) You can also iterate directly
           over the sprite group."""
        return self.spritedict.keys()

    def add_internal(self, sprite):
        self.spritedict[sprite] = 0

    def remove_internal(self, sprite):
        r = self.spritedict[sprite]
        if r is not 0:
            self.lostsprites.append(r)
        del(self.spritedict[sprite])

    def has_internal(self, sprite):
        return self.spritedict.has_key(sprite)

    def copy(self):
        """copy()
           copy a group with all the same sprites

           Returns a copy of the group that is the same class
           type, and has the same sprites in it."""
        return self.__class__(self.sprites())

    def __iter__(self):
        return iter(self.sprites())

    def __contains__(self, sprite):
        return self.has(sprite)

    def add(self, *sprites):
        """add(sprite, list, or group, ...)
           add sprite to group

           Add a sprite or sequence of sprites to a group."""
        for sprite in sprites:
            # It's possible that some sprite is also an iterator.
            # If this is the case, we should add the sprite itself,
            # and not the objects it iterates over.
            if isinstance(sprite, Sprite):
                if not self.has_internal(sprite):
                    self.add_internal(sprite)
                    sprite.add_internal(self)
            else:
                try:
                    # See if sprite is an iterator, like a list or sprite
                    # group.
                    for spr in sprite:
                        self.add(spr)
                except (TypeError, AttributeError):
                    # Not iterable, this is probably a sprite that happens
                    # to not subclass Sprite. Alternately, it could be an
                    # old-style sprite group.
                    if hasattr(sprite, '_spritegroup'):
                        for spr in sprite.sprites():
                            if not self.has_internal(spr):
                                self.add_internal(spr)
                                spr.add_internal(self)
                    elif not self.has_internal(sprite):
                        self.add_internal(sprite)
                        sprite.add_internal(self)

    def remove(self, *sprites):
        """remove(sprite, list, or group, ...)
           remove sprite from group

           Remove a sprite or sequence of sprites from a group."""
        # This function behaves essentially the same as Group.add.
        # Check for Spritehood, check for iterability, check for
        # old-style sprite group, and fall back to assuming
        # spritehood.
        for sprite in sprites:
            if isinstance(sprite, Sprite):
                if self.has_internal(sprite):
                    self.remove_internal(sprite)
                    sprite.remove_internal(self)
            else:
                try:
                    for spr in sprite: self.remove(spr)
                except (TypeError, AttributeError):
                    if hasattr(sprite, '_spritegroup'):
                        for spr in sprite.sprites():
                            if self.has_internal(spr):
                                self.remove_internal(spr)
                                spr.remove_internal(self)
                    elif self.has_internal(sprite):
                        self.remove_internal(sprite)
                        sprite.remove_internal(self)

    def has(self, *sprites):
        """has(sprite or group, ...)
           ask if group has a sprite or sprites

           Returns true if the given sprite or sprites are
           contained in the group. You can also use 'sprite in group'
           or 'subgroup in group'."""
        # Again, this follows the basic pattern of Group.add and
        # Group.remove.
        for sprite in sprites:
            if isinstance(sprite, Sprite):
                return self.has_internal(sprite)

            try:
                for spr in sprite:
                    if not self.has(spr):
                        return False
                return True
            except (TypeError, AttributeError):
                if hasattr(sprite, '_spritegroup'):
                    for spr in sprite.sprites():
                        if not self.has_internal(spr):
                            return False
                    return True
                else:
                    return self.has_internal(sprite)

    def update(self, *args):
        """update(*args)
           call update for all member sprites

           calls the update method for all sprites in the group.
           Passes all arguments on to the Sprite update function."""
        for s in self.sprites(): s.update(*args)

    def draw(self, surface):
        """draw(surface)
           draw all sprites onto the surface

           Draws all the sprites onto the given surface."""
        sprites = self.sprites()
        surface_blit = surface.blit
        for spr in sprites:
            self.spritedict[spr] = surface_blit(spr.image, spr.rect)
        self.lostsprites = []

    def clear(self, surface, bgd):
        """clear(surface, bgd)
           erase the previous position of all sprites

           Clears the area of all drawn sprites. the bgd
           argument should be Surface which is the same
           dimensions as the surface. The bgd can also be
           a function which gets called with the passed
           surface and the area to be cleared."""
        if callable(bgd):
            for r in self.lostsprites:
                bgd(surface, r)
            for r in self.spritedict.values():
                if r is not 0: bgd(surface, r)
        else:
            surface_blit = surface.blit
            for r in self.lostsprites:
                surface_blit(bgd, r, r)
            for r in self.spritedict.values():
                if r is not 0: surface_blit(bgd, r, r)

    def empty(self):
        """empty()
           remove all sprites

           Removes all the sprites from the group."""
        for s in self.sprites():
            self.remove_internal(s)
            s.remove_internal(self)

    def __nonzero__(self):
        return (len(self.sprites()) != 0)

    def __len__(self):
        """len(group)
           number of sprites in group

           Returns the number of sprites contained in the group."""
        return len(self.sprites())

    def __repr__(self):
        return "<%s(%d sprites)>" % (self.__class__.__name__, len(self))

class Group(AbstractGroup):
    """The basic Group class you will want to use.
       It supports all of the above operations and methods.

       The RenderPlain and RenderClear groups are aliases to Group
       for compatibility."""
    
    def __init__(self, *sprites):
        AbstractGroup.__init__(self)
        self.add(*sprites)

RenderPlain = Group
RenderClear = Group

class RenderUpdates(Group):
    """A sprite group that's more efficient at updating.
       This group supports drawing to the screen, but its draw method
       also returns a list of the Rects updated by the draw (and any
       clears in between the last draw and the current one). You
       can use pygame.display.update(renderupdates_group.draw(screen))
       to minimize the updated part of the screen. This can usually
       make things much faster."""
    
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
    """RenderUpdates, but the sprites are drawn in the order they were added.
       More recently added sprites are drawn last (and so, above other
       sprites)."""

    def __init__(self, *sprites):
        self._spritelist = []
        RenderUpdates.__init__(self, *sprites)

    def sprites(self): return list(self._spritelist)

    def add_internal(self, sprite):
        RenderUpdates.add_internal(self, sprite)
        self._spritelist.append(sprite)

    def remove_internal(self, sprite):
        RenderUpdates.remove_internal(self, sprite)
        self._spritelist.remove(sprite)

class GroupSingle(AbstractGroup):
    """A group container that holds a single most recent item.
       This class works just like a regular group, but it only
       keeps a single sprite in the group. Whatever sprite has
       been added to the group last, will be the only sprite in
       the group.

       You can access its one sprite as the .sprite attribute.
       Assigning to this attribute will properly remove the old
       sprite and then add the new one."""

    def __init__(self, sprite = None):
        AbstractGroup.__init__(self)
        self.__sprite = None
        if sprite is not None: self.add(sprite)

    def copy(self):
        return GroupSingle(self.__sprite)

    def sprites(self):
        if self.__sprite is not None: return [self.__sprite]
        else: return []

    def add_internal(self, sprite):
        if self.__sprite is not None:
            self.__sprite.remove_internal(self)
        self.__sprite = sprite

    def __nonzero__(self): return (self.__sprite is not None)

    def _get_sprite(self):
        return self.__sprite

    def _set_sprite(self, sprite):
        self.add_internal(sprite)
        sprite.add_internal(self)
        return sprite

    sprite = property(_get_sprite, _set_sprite, None,
                      "The sprite contained in this group")
    
    def remove_internal(self, sprite):
        if sprite is self.__sprite: self.__sprite = None

    def has_internal(self, sprite):
        return (self.__sprite is sprite)

    # Optimizations...
    def __contains__(self, sprite): return (self.__sprite is sprite)

def spritecollide(sprite, group, dokill):
    """pygame.sprite.spritecollide(sprite, group, dokill) -> list
       collision detection between sprite and group

       given a sprite and a group of sprites, this will
       return a list of all the sprites that intersect
       the given sprite.
       all sprites must have a "rect" value, which is a
       rectangle of the sprite area. if the dokill argument
       is true, the sprites that do collide will be
       automatically removed from all groups."""
    crashed = []
    spritecollide = sprite.rect.colliderect
    if dokill:
        for s in group.sprites():
            if spritecollide(s.rect):
                s.kill()
                crashed.append(s)
    else:
        for s in group:
            if spritecollide(s.rect):
                crashed.append(s)
    return crashed

def groupcollide(groupa, groupb, dokilla, dokillb):
    """pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb) -> dict
       collision detection between group and group

       given two groups, this will find the intersections
       between all sprites in each group. it returns a
       dictionary of all sprites in the first group that
       collide. the value for each item in the dictionary
       is a list of the sprites in the second group it
       collides with. the two dokill arguments control if
       the sprites from either group will be automatically
       removed from all groups."""
    crashed = {}
    SC = spritecollide
    if dokilla:
        for s in groupa.sprites():
            c = SC(s, groupb, dokillb)
            if c:
                crashed[s] = c
                s.kill()
    else:
        for s in groupa:
            c = SC(s, groupb, dokillb)
            if c:
                crashed[s] = c
    return crashed

def spritecollideany(sprite, group):
    """pygame.sprite.spritecollideany(sprite, group) -> sprite
       finds any sprites that collide

       given a sprite and a group of sprites, this will
       return return any single sprite that collides with
       with the given sprite. If there are no collisions
       this returns None.

       if you don't need all the features of the
       spritecollide function, this function will be a
       bit quicker.

       all sprites must have a "rect" value, which is a
       rectangle of the sprite area."""
    spritecollide = sprite.rect.colliderect
    for s in group:
        if spritecollide(s.rect):
            return s
    return None
