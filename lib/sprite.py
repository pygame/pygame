##    pygame - Python Game Library
##    Copyright (C) 2000-2003  Pete Shinners
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

The design of the sprites and groups is very flexible.
There's no need to inherit from the provided classes, you
can use any object you want for the sprites, as long as it
contains "add_internal" and "remove_internal" methods,
which are called by the groups when they remove and add
sprites. The same is true for containers. A container
can be any python object that has "add_internal" and
"remove_internal" methods that the sprites call when
they want add and remove themselves from containers. The
containers must also have a member named "_spritegroup",
which can be set to any dummy value.
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


import sys
if sys.hexversion < 0x020200a1:
    class object:
        "null class for pre-python2.2 compatability"



class Sprite(object):
    """the base class for your visible game objects.
       The sprite class is meant to be used as a base class
       for the objects in your game. It just provides functions
       to maintain itself in different groups. A sprite is
       considered 'alive' as long as it is a member of one
       or more groups. The kill() method simply removes this
       sprite from all groups."""

    def __init__(self, group=()):
        """__init__(group=())
           initialize a sprite object

           You can initialize a sprite by passing it a
           group or sequence of groups to be contained in."""
        self.__g = {}
        self.add(group)

    def add(self, group):
        """add(group)
           add a sprite to container

           Add the sprite to a group or sequence of groups."""
        has = self.__g.has_key
        if hasattr(group, '_spritegroup'):
            if not has(group):
                group.add_internal(self)
                self.__g[group] = 0
        else:
            groups = group
            for group in groups:
                if not has(group):
                    group.add_internal(self)
                    self.__g[group] = 0

    def remove(self, group):
        """remove(group)
           remove a sprite from container

           Remove the sprite from a group or sequence of groups."""
        has = self.__g.has_key
        if hasattr(group, '_spritegroup'):
            if has(group):
                group.remove_internal(self)
                del self.__g[group]
        else:
            for g in group:
                if has(g):
                    g.remove_internal(self)
                    del self.__g[group]

    def add_internal(self, group):
        self.__g[group] = 0

    def remove_internal(self, group):
        del self.__g[group]

    def update(self, *args):
        #note, this just ignores all args
        pass

    def kill(self):
        """kill()
           end life of sprite, remove from all groups

           Removes the sprite from all the groups that contain
           it. The sprite is still fine after calling this kill()
           so you could use it to remove a sprite from all groups,
           and then add it to some other groups."""
        for c in self.__g.keys():
            c.remove_internal(self)
        self.__g.clear()

    def groups(self):
        """groups() -> list
           list used sprite containers

           Returns a list of all the groups that contain this
           sprite."""
        return self.__g.keys()

    def alive(self):
        """alive() -> bool
           ask the life of a sprite

           Returns true if this sprite is a member of any groups."""
        return len(self.__g)

    def __repr__(self):
        return "<%s sprite(in %d groups)>" % (self.__class__.__name__, len(self.__g))



class Group(object):
    """the Group class is a container for sprites
       This is the base sprite group class. It does everything
       needed to behave as a normal group. You can easily inherit
       a new group class from this if you want to add more features."""
    _spritegroup = 1 #dummy val to identify sprite groups

    def __init__(self, sprite=()):
        """__init__(sprite=())
           instance a Group

           You can initialize a group by passing it a
           sprite or sequence of sprites to be contained."""
        self.spritedict = {}
        if sprite:
            self.add(sprite)

    def copy(self):
        """copy() -> Group
           copy a group with all the same sprites

           Returns a copy of the group that is the same class
           type, and has the same contained sprites."""
        return self.__class__(self.spritedict.keys())

    def sprites(self):
        """sprites() -> iterator
           return an object to loop over each sprite

           Returns an object that can be looped over with
           a 'for' loop. (For now it is always a list, but
           newer version of python could return different
           objects, like iterators.)"""
        return self.spritedict.keys()

    def add(self, sprite):
        """add(sprite)
           add sprite to group

           Add a sprite or sequence of sprites to a group."""
        has = self.spritedict.has_key
        if hasattr(sprite, '_spritegroup'):
            for sprite in sprite.sprites():
                if not has(sprite):
                    self.spritedict[sprite] = 0
                    sprite.add_internal(self)
        else:
            try: len(sprite) #see if its a sequence
            except (TypeError, AttributeError):
                if not has(sprite):
                        self.spritedict[sprite] = 0
                        sprite.add_internal(self)
            else:
                for sprite in sprite:
                    if not has(sprite):
                        self.spritedict[sprite] = 0
                        sprite.add_internal(self)

    def remove(self, sprite):
        """remove(sprite)
           remove sprite from group

           Remove a sprite or sequence of sprites from a group."""
        has = self.spritedict.has_key
        if hasattr(sprite, '_spritegroup'):
            for sprite in sprite.sprites():
                if has(sprite):
                    self.remove_internal(sprite)
                    sprite.remove_internal(self)
        else:
            try: len(sprite) #see if its a sequence
            except (TypeError, AttributeError):
                if has(sprite):
                    self.remove_internal(sprite)
                    sprite.remove_internal(self)
            else:
                for sprite in sprite:
                    if has(sprite):
                        self.remove_internal(sprite)
                        sprite.remove_internal(self)

    def add_internal(self, sprite):
        self.spritedict[sprite] = 0

    def remove_internal(self, sprite):
        del self.spritedict[sprite]

    def has(self, sprite):
        """has(sprite) -> bool
           ask if group has sprite

           Returns true if the given sprite or sprites are
           contained in the group"""
        has = self.spritedict.has_key
        if hasattr(sprite, '_spritegroup'):
            for sprite in sprite.sprites():
                if not has(sprite):
                    return 0
        else:
            try: len(sprite) #see if its a sequence
            except (TypeError, AttributeError):
                return has(sprite)
            else:
                for sprite in sprite:
                    if not has(sprite):
                        return 0
        return 1

        if hasattr(sprite, '_spritegroup'):
            return sprite in has(sprite)
        sprites = sprite
        for sprite in sprites:
            if not has(sprite):
                return 0
        return 1

    def empty(self):
        """empty()
           remove all sprites

           Removes all the sprites from the group."""
        for s in self.spritedict.keys():
            self.remove_internal(s)
            s.remove_internal(self)
        self.spritedict.clear()

    def update(self, *args):
        """update(...)
           call update for all member sprites

           calls the update method for all sprites in the group.
           Passes all arguments on to the Sprite update function."""
        if args:
            a=apply
            for s in self.spritedict.keys():
                a(s.update, args)
        else:
            for s in self.spritedict.keys():
                s.update()


    def __nonzero__(self):
        """__nonzero__() -> bool
           ask if group is empty

           Returns true if the group has any sprites. This
           lets you check if the group has any sprites by
           using it in a logical if statement."""
        return len(self.spritedict)

    def __len__(self):
        """__len__() -> int
           number of sprites in group

           Returns the number of sprites contained in the group."""
        return len(self.spritedict)

    def __repr__(self):
        return "<%s(%d sprites)>" % (self.__class__.__name__, len(self))

##note that this GroupSingle class is not derived from any other
##group class, it can make as a good example if you ever want to
##create your own new group type.

class GroupSingle(object):
    """a group container that holds a single most recent item
       This class works just like a regular group, but it only
       keeps a single sprite in the group. Whatever sprite has
       been added to the group last, will be the only sprite in
       the group."""
    _spritegroup = 1 #dummy val to identify groups

    def __init__(self, sprite=()):
        self.sprite = 0
        self.add(sprite)

    def copy(self):
        if self.sprite is not 0:
            return GroupSingle(self.sprite)
        return GroupSingle()

    def sprites(self):
        if self.sprite is not 0:
            return [self.sprite]
        return []

    def add(self, sprite):
        if hasattr(sprite, '_spritegroup'):
            for sprite in sprite.sprites(): pass
        else:
            try:
                if not len(sprite): return #see if its a sequence
                sprite = sprite[-1]
            except (TypeError, AttributeError): pass
        if sprite is not self.sprite:
            self.add_internal(sprite)
            sprite.add_internal(self)

    def remove(self, sprite):
        if hasattr(sprite, '_spritegroup'):
            for sprite in sprite.sprites():
                if self.sprite is sprite:
                    self.sprite = 0
                    sprite.remove_internal(self)
                    break
        else:
            try:
                if not len(sprite): return #see if its a sequence
            except (TypeError, AttributeError):
                if self.sprite is sprite:
                    self.sprite = 0
                    sprite.remove_internal(self)
            else:
                for sprite in sprite:
                    if self.sprite is sprite:
                        self.sprite = 0
                        sprite.remove_internal(self)
                        break

    def add_internal(self, sprite):
        if self.sprite is not 0:
            self.sprite.remove_internal(self)
        self.sprite = sprite

    def remove_internal(self, sprite):
        self.sprite = 0

    def has(self, sprite):
        return self.sprite is sprite

    def empty(self):
        if self.sprite is not 0:
            self.sprite.remove_internal(self)
            self.sprite = 0

    def update(self, *args):
        if self.sprite:
            apply(self.sprite.update, args)

    def __nonzero__(self):
        return self.sprite is not 0

    def __len__(self):
        return self.sprite is not 0

    def __repr__(self):
        return "<%s(%d sprites)>" % (self.__class__.__name__, len(self))


##these render groups are derived from the normal Group
##class, they just add methods. the Updates and Clear versions
##of drawing are more complex groups. They keep track of sprites
##that have been drawn but are removed, and also keep track of
##drawing info from each sprite, by storing it in the dictionary
##values used to store the sprite. very sneaky, but a great
##example for you if you ever need your own group class that
##maintains its own data along with each sprite it holds.

class RenderPlain(Group):
    """a sprite group that can draw all its sprites
       The RenderPlain group is just like a normal group,
       it just adds a "draw" method. Any sprites used with
       this group to draw must contain two member elements
       named "image" and "rect". These are a pygame Surface
       and Rect object that are passed to blit."""

    def draw(self, surface):
        """draw(surface)
           draw all sprites onto a surface

           Draws all the sprites onto the given surface."""
        spritedict = self.spritedict
        surface_blit = surface.blit
        for s in spritedict.keys():
            surface_blit(s.image, s.rect)



class RenderClear(Group):
    """a group container that can draw and clear its sprites
       The RenderClear group is just like a normal group,
       but it can draw and clear the sprites. Any sprites
       used in this group must contain member elements
       named "image" and "rect". These are a pygame Surface
       and Rect, which are passed to a blit call."""
    def __init__(self, sprite=()):
        Group.__init__(self, sprite)
        self.lostsprites = []

    def remove_internal(self, sprite):
        r = self.spritedict[sprite]
        if r is not 0:
            self.lostsprites.append(r)
        del self.spritedict[sprite]

    def draw(self, surface):
        """draw(surface)
           draw all sprites onto a surface

           Draws all the sprites onto the given surface."""
        spritedict = self.spritedict
        surface_blit = surface.blit
        for s in spritedict.keys():
            spritedict[s] = surface_blit(s.image, s.rect)
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



class RenderUpdates(RenderClear):
    """a sprite group that can draw and clear with update rectangles
       The RenderUpdates is derived from the RenderClear group
       and keeps track of all the areas drawn and cleared. It
       also smartly handles overlapping areas between where a
       sprite was drawn and cleared when generating the update
       rectangles."""

    def draw(self, surface):
        """draw(surface)
           draw all sprites onto the surface

           Draws all the sprites onto the given surface. It
           returns a list of rectangles, which should be passed
           to pygame.display.update()"""
        spritedict = self.spritedict
        surface_blit = surface.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        for s, r in spritedict.items():
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
        for s in group.sprites():
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
        for s in groupa.sprites():
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
    for s in group.sprites():
        if spritecollide(s.rect):
            return s
    return None
