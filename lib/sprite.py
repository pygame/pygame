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

import pygame
from pygame import Rect
from pygame.time import get_ticks

# Don't depend on pygame.mask if it's not there...
try:
    from pygame.mask import from_surface
except:
    pass


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


class DirtySprite(Sprite):
    """
    DirtySprite has new attributes:
    
    dirty: if set to 1, it is repainted and then set to 0 again 
           if set to 2 then it is always dirty ( repainted each frame, 
           flag is not reset)
           0 means that it is not dirty and therefor not repainted again
    blendmode: its the special_flags argument of blit, blendmodes
    source_rect: source rect to use, remember that it relative to 
                 topleft (0,0) of self.image
    visible: normally 1, if set to 0 it will not be repainted 
             (you must set it dirty too to be erased from screen)
    """
    
    def __init__(self, *groups):
        """
        Same as pygame.sprite.Sprite but initializes the new attributes to
        default values:
        dirty = 1 (to be always dirty you have to set it)
        blendmode = 0
        layer = 0 (READONLY value, it is read when adding it to the 
                   LayeredRenderGroup, for details see doc of 
                   LayeredRenderGroup)
        """
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
    visible = property(lambda self: self._get_visible(),\
                       lambda self, value:self._set_visible(value), \
                       doc="you can make this sprite disappear without removing it from the group, values 0 for invisible and 1 for visible")
        
    def __repr__(self):
        return "<%s DirtySprite(in %d groups)>" % (self.__class__.__name__, len(self.__g))



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


class LayeredUpdates(AbstractGroup):
    """
    LayeredRenderGroup
    
    A group that handles layers. For drawing it uses the same metod as the 
    pygame.sprite.OrderedUpdates.
    
    This group is fully compatible with pygame.sprite.Sprite.
    """
    
    def __init__(self, *sprites, **kwargs):
        """
        You can set the default layer through kwargs using 'default_layer'
        and an integer for the layer. The default layer is 0.
        
        If the sprite you add has an attribute layer then that layer will
        be used.
        If the kwarg contain 'layer' then the sprites passed will be 
        added to that layer (overriding the sprite.layer attribute).
        If neither sprite has attribute layer nor kwarg then the default
        layer is used to add the sprites.
        """
        self._spritelayers = {}
        self._spritelist = []
        AbstractGroup.__init__(self)
        if kwargs.has_key('default_layer'):
            self._default_layer = kwargs['default_layer']
        else:
            self._default_layer = 0
            
        self.add(*sprites, **kwargs)
    
    def add_internal(self, sprite, layer=None):
        """
        Do not use this method directly. It is used by the group to add a
        sprite internally.
        """
        self.spritedict[sprite] = Rect(0, 0, 0, 0) # add a old rect
        
        if layer is None:
            if hasattr(sprite, '_layer'):
                layer = sprite._layer
            else:
                layer = self._default_layer
                
                
        self._spritelayers[sprite] = layer
        if hasattr(sprite, '_layer'):
            sprite._layer = layer
    
        # add the sprite at the right position
        # bisect algorithmus
        sprites = self._spritelist # speedup
        sprites_layers = self._spritelayers
        leng = len(sprites)
        low = 0
        high = leng-1
        mid = low
        while(low<=high):
            mid = low + (high-low)/2
            if(sprites_layers[sprites[mid]]<=layer):
                low = mid+1
            else:
                high = mid-1
        # linear search to find final position
        while(mid<leng and sprites_layers[sprites[mid]]<=layer):
            mid += 1
        sprites.insert(mid, sprite)
        
    def add(self, *sprites, **kwargs):
        """add(sprite, list, or group, ...)
           add sprite to group

           Add a sprite or sequence of sprites to a group.
        
        If the sprite(s) have an attribute layer then that is used 
        for the layer. If kwargs contains 'layer' then the sprite(s) 
        will be added to that argument (overriding the sprite layer 
        attribute). If neither is passed then the sprite(s) will be
        added to the default layer.
        """
        layer = None
        if kwargs.has_key('layer'):
            layer = kwargs['layer']
        if sprites is None or len(sprites)==0:
            return
        for sprite in sprites:
            # It's possible that some sprite is also an iterator.
            # If this is the case, we should add the sprite itself,
            # and not the objects it iterates over.
            if isinstance(sprite, Sprite):
                if not self.has_internal(sprite):
                    self.add_internal(sprite, layer)
                    sprite.add_internal(self)
            else:
                try:
                    # See if sprite is an iterator, like a list or sprite
                    # group.
                    for spr in sprite:
                        self.add(spr, **kwargs)
                except (TypeError, AttributeError):
                    # Not iterable, this is probably a sprite that happens
                    # to not subclass Sprite. Alternately, it could be an
                    # old-style sprite group.
                    if hasattr(sprite, '_spritegroup'):
                        for spr in sprite.sprites():
                            if not self.has_internal(spr):
                                self.add_internal(spr, layer)
                                spr.add_internal(self)
                    elif not self.has_internal(sprite):
                        self.add_internal(sprite, layer)
                        sprite.add_internal(self)
    
    def remove_internal(self, sprite):
        """
        Do not use this method directly. It is used by the group to 
        add a sprite.
        """
        self._spritelist.remove(sprite)
        # these dirty rects are suboptimal for one frame
        self.lostsprites.append(self.spritedict[sprite]) # dirty rect
        if hasattr(sprite, 'rect'):
            self.lostsprites.append(sprite.rect) # dirty rect
        
        self.spritedict.pop(sprite, 0)
        self._spritelayers.pop(sprite)
    
    def sprites(self):
        """
        Returns a ordered list of sprites (first back, last top).
        """
        return list(self._spritelist)
    
    def draw(self, surface):
        """
        Draw all sprites in the right order onto the passed surface.
        """
        spritedict = self.spritedict
        surface_blit = surface.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        for spr in self.sprites():
            rec = spritedict[spr]
            newrect = surface_blit(spr.image, spr.rect)
            if rec is 0:
                dirty_append(newrect)
            else:
                if newrect.colliderect(rec):
                    dirty_append(newrect.union(rec))
                else:
                    dirty_append(newrect)
                    dirty_append(rec)
            spritedict[spr] = newrect
        return dirty

    def get_sprites_at(self, pos):
        """
        Returns a list with all sprites at that position.
        Bottom sprites first, top last.
        """
        _sprites = self._spritelist
        rect = Rect(pos, (0, 0))
        colliding_idx = rect.collidelistall(_sprites)
        colliding = []
        colliding_append = colliding.append
        for i in colliding_idx:
            colliding_append(_sprites[i])
        return colliding

    def get_sprite(self, idx):
        """
        Returns the sprite at the index idx from the sprites().
        Raises IndexOutOfBounds.
        """
        return self._spritelist[idx]
    
    def remove_sprites_of_layer(self, layer_nr):
        """
        Removes all sprites from a layer and returns them as a list.
        """
        sprites = self.get_sprites_from_layer(layer_nr)
        self.remove(sprites)
        return sprites
        

    #---# layer methods
    def layers(self):
        """
        Returns a list of layers defined (unique), sorted from botton up.
        """
        layers = set()
        for layer in self._spritelayers.values():
            layers.add(layer)
        return list(layers)

    def change_layer(self, sprite, new_layer):
        """
        Changes the layer of the sprite.
        sprite must have been added to the renderer. It is not checked.
        """
        sprites = self._spritelist # speedup
        sprites_layers = self._spritelayers # speedup
        
        sprites.remove(sprite) 
        sprites_layers.pop(sprite)
        
        # add the sprite at the right position
        # bisect algorithmus
        leng = len(sprites)
        low = 0
        high = leng-1
        mid = low
        while(low<=high):
            mid = low + (high-low)/2
            if(sprites_layers[sprites[mid]]<=new_layer):
                low = mid+1
            else:
                high = mid-1
        # linear search to find final position
        while(mid<leng and sprites_layers[sprites[mid]]<=new_layer):
            mid += 1
        sprites.insert(mid, sprite)
        if hasattr(sprite, 'layer'):
            sprite.layer = new_layer
        
        # add layer info
        sprites_layers[sprite] = new_layer
            
    def get_layer_of_sprite(self, sprite):
        """
        Returns the layer that sprite is currently in. If the sprite is not 
        found then it will return the default layer.
        """
        return self._spritelayers.get(sprite, self._default_layer)
    
    def get_top_layer(self):
        """
        Returns the number of the top layer.
        """
        return self._spritelayers[self._spritelist[-1]]
####        return max(self._spritelayers.values())
    
    def get_bottom_layer(self):
        """
        Returns the number of the bottom layer.
        """
        return self._spritelayers[self._spritelist[0]]
####        return min(self._spritelayers.values())
    
    def move_to_front(self, sprite):
        """
        Brings the sprite to front, changing the layer o the sprite
        to be in the topmost layer (added at the end of that layer).
        """
        self.change_layer(sprite, self.get_top_layer())
        
    def move_to_back(self, sprite):
        """
        Moves the sprite to the bottom layer, moving it behind
        all other layers and adding one additional layer.
        """
        self.change_layer(sprite, self.get_bottom_layer()-1)
        
    def get_top_sprite(self):
        """
        Returns the topmost sprite.
        """
        return self._spritelist[-1]
    
    def get_sprites_from_layer(self, layer):
        """
        Returns all sprites from a layer, ordered by how they where added.
        It uses linear search and the sprites are not removed from layer.
        """
        sprites = []
        sprites_append = sprites.append
        sprite_layers = self._spritelayers
        for spr in self._spritelist:
            if sprite_layers[spr] == layer: 
                sprites_append(spr)
            elif sprite_layers[spr]>layer:# break after because no other will 
                                          # follow with same layer
                break
        return sprites
        
    def switch_layer(self, layer1_nr, layer2_nr):
        """
        Switches the sprites from layer1 to layer2.
        The layers number must exist, it is not checked.
        """
        sprites1 = self.remove_sprites_of_layer(layer1_nr)
        for spr in self.get_sprites_from_layer(layer2_nr):
            self.change_layer(spr, layer1_nr)
        self.add(sprites1, layer=layer2_nr)


class LayeredDirty(LayeredUpdates):
    """
    Yet another group. It uses the dirty flag technique and is therefore 
    faster than the pygame.sprite.RenderUpdates if you have many static 
    sprites. It also switches automatically between dirty rect update and 
    full screen rawing, so you do no have to worry what would be faster. It 
    only works with the DirtySprite or any sprite that has the following 
    attributes: image, rect, dirty, visible, blendmode (see doc of 
    DirtySprite).
    """
    
    def __init__(self, *sprites, **kwargs):
        """
        Same as for the pygame.sprite.Group.
        You can specify some additional attributes through kwargs:
        _use_update: True/False   default is False
        _default_layer: the default layer where the sprites without a layer are
                        added.
        _time_threshold: treshold time for switching between dirty rect mode and
                        fullscreen mode, defaults to 1000./80  == 1000./fps
        """
        LayeredUpdates.__init__(self, *sprites, **kwargs)
        self._clip = None
        
        self._use_update = False
        
        self._time_threshold = 1000./80. # 1000./ fps
        
        
        self._bgd = None
        for key, val in kwargs.items():
            if key in ['_use_update', '_time_threshold', '_default_layer']:
                if hasattr(self, key):
                    setattr(self, key, val)

    def add_internal(self, sprite, layer=None):
        """
        Do not use this method directly. It is used by the group to add a
        sprite internally.
        """
        # check if all attributes needed are set
        if not hasattr(sprite, 'dirty'):
            raise AttributeError()
        if not hasattr(sprite, "visible"):
            raise AttributeError()
        if not hasattr(sprite, "blendmode"):
            raise AttributeError()
        
        if not isinstance(sprite, DirtySprite):
            raise TypeError()
        
        if sprite.dirty == 0: # set it dirty if it is not
            sprite.dirty = 1
        
        LayeredUpdates.add_internal(self, sprite, layer)
        
    def draw(self, surface, bgd=None):
        """
        Draws all sprites on the surface you pass in.
        You can pass the background too. If a background is already set, 
        then the bgd argument has no effect.
        """
        # speedups
        _orig_clip = surface.get_clip()
        _clip = self._clip
        if _clip is None:
            _clip = _orig_clip
        
        
        _surf = surface
        _sprites = self._spritelist
        _old_rect = self.spritedict
        _update = self.lostsprites
        _update_append = _update.append
        _ret = None
        _surf_blit = _surf.blit
        _rect = Rect
        if bgd is not None:
            self._bgd = bgd
        _bgd = self._bgd
        
        _surf.set_clip(_clip)
        # -------
        # 0. deside if normal render of flip
        start_time = get_ticks()
        if self._use_update: # dirty rects mode
            # 1. find dirty area on screen and put the rects into _update
            # still not happy with that part
            for spr in _sprites:
                if 0 < spr.dirty:
                    # chose the right rect
                    if spr.source_rect:
                        _union_rect = _rect(spr.rect.topleft, spr.source_rect.size)
                    else:
                        _union_rect = _rect(spr.rect)
                        
                    _union_rect_collidelist = _union_rect.collidelist
                    _union_rect_union_ip = _union_rect.union_ip
                    i = _union_rect_collidelist(_update)
                    while -1 < i:
                        _union_rect_union_ip(_update[i])
                        del _update[i]
                        i = _union_rect_collidelist(_update)
                    _update_append(_union_rect.clip(_clip))
                    
                    _union_rect = _rect(_old_rect[spr])
                    _union_rect_collidelist = _union_rect.collidelist
                    _union_rect_union_ip = _union_rect.union_ip
                    i = _union_rect_collidelist(_update)
                    while -1 < i:
                        _union_rect_union_ip(_update[i])
                        del _update[i]
                        i = _union_rect_collidelist(_update)
                    _update_append(_union_rect.clip(_clip))
            # can it be done better? because that is an O(n**2) algorithm in
            # worst case
                    
            # clear using background
            if _bgd is not None:
                for rec in _update:
                    _surf_blit(_bgd, rec, rec)
                
            # 2. draw
            for spr in _sprites:
                if 1 > spr.dirty:
                    if spr._visible:
                        # sprite not dirty, blit only the intersecting part
                        _spr_rect = spr.rect
                        if spr.source_rect is not None:
                            _spr_rect = Rect(spr.rect.topleft, spr.source_rect.size)
                        _spr_rect_clip = _spr_rect.clip
                        for idx in _spr_rect.collidelistall(_update):
                            # clip
                            clip = _spr_rect_clip(_update[idx])
                            _surf_blit(spr.image, clip, \
                                       (clip[0]-_spr_rect[0], \
                                            clip[1]-_spr_rect[1], \
                                            clip[2], \
                                            clip[3]), spr.blendmode)
                else: # dirty sprite
                    if spr._visible:
                        _old_rect[spr] = _surf_blit(spr.image, spr.rect, \
                                               spr.source_rect, spr.blendmode)
                    if spr.dirty == 1:
                        spr.dirty = 0
            _ret = list(_update)
        else: # flip, full screen mode
            if _bgd is not None:
                _surf_blit(_bgd, (0, 0))
            for spr in _sprites:
                if spr._visible:
                    _old_rect[spr] = _surf_blit(spr.image, spr.rect, spr.source_rect,spr.blendmode)
            _ret = [_rect(_clip)] # return only the part of the screen changed
            
        
        # timing for switching modes
        # how to find a good treshold? it depends on the hardware it runs on
        end_time = get_ticks()
        if end_time-start_time > self._time_threshold:
            self._use_update = False
        else:
            self._use_update = True
            
##        # debug
##        print "               check: using dirty rects:", self._use_update
            
        # emtpy dirty reas list
        _update[:] = []
        
        # -------
        # restore original clip
        _surf.set_clip(_orig_clip)
        return _ret

    def clear(self, surface, bgd):
        """
        Only used to set background.
        """
        self._bgd = bgd

    def repaint_rect(self, screen_rect): 
        """
        Repaints the given area.
        screen_rect in screencoordinates.
        """
        self.lostsprites.append(screen_rect.clip(self._clip))
        
    def set_clip(self, screen_rect=None):
        """
        clip the area where to draw. Just pass None (default) to 
        reset the clip.
        """
        if screen_rect is None:
            self._clip = pygame.display.get_surface().get_rect()
        else:
            self._clip = screen_rect
        self._use_update = False
        
    def get_clip(self):
        """
        Returns the current clip.
        """
        return self._clip
    
    def change_layer(self, sprite, new_layer):
        """
        Changes the layer of the sprite.
        sprite must have been added to the renderer. It is not checked.
        """
        LayeredRenderGroup.change_layer(self, sprite, new_layer)
        if sprite.dirty == 0:
            sprite.dirty = 1
            
    def set_timing_treshold(self, time_ms):
        """
        Sets the treshold in milliseconds. Default is 1000./80 where 80 is the
        fps I want to switch to full screen mode.
        """
        self._time_threshold = time_ms
    






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





# some different collision detection functions that could be used.

def collide_rect(left, right):
    """pygame.sprite.collide_rect(left, right) -> bool
       collision detection between two sprites, using rects.

       Tests for collision between two sprites. Uses the
       pygame rect colliderect function to calculate the
       collision. Intended to be passed as a collided
       callback function to the *collide functions.
       Sprites must have a "rect" attributes."""
    return left.rect.colliderect(right.rect)

class collide_rect_ratio:
    """A callable class that checks for collisions between
       two sprites, using a scaled version of the sprites
       rects.

       Is created with a ratio, the instance is then intended
       to be passed as a collided callback function to the
       *collide functions."""
    
    def __init__( self, ratio ):
        """Creates a new collide_rect_ratio callable. ratio is
           expected to be a floating point value used to scale
           the underlying sprite rect before checking for
           collisions."""
        self.ratio = ratio

    def __call__( self, left, right ):
        """pygame.sprite.collide_rect_ratio(ratio)(left, right) -> bool
           collision detection between two sprites, using scaled rects.

           Tests for collision between two sprites. Uses the
           pygame rect colliderect function to calculate the
           collision, after scaling the rects by the stored ratio.
           Sprites must have a "rect" attributes."""
        ratio = self.ratio
        
        leftrect = left.rect
        width = leftrect.width
        height = leftrect.height
        leftrect = leftrect.inflate( width * ratio - width, height * ratio - height )
        
        rightrect = right.rect
        width = rightrect.width
        height = rightrect.height
        rightrect = rightrect.inflate( width * ratio - width, height * ratio - height )
        
        return leftrect.colliderect( rightrect )

def collide_circle( left, right ):
    """pygame.sprite.collide_circle(left, right) -> bool
       collision detection between two sprites, using circles.

       Tests for collision between two sprites, by testing to
       see if two circles centered on the sprites overlap. If
       the sprites have a "radius" attribute, that is used to
       create the circle, otherwise a circle is created that
       is big enough to completely enclose the sprites rect as
       given by the "rect" attribute. Intended to be passed as
       a collided callback function to the *collide functions.
       Sprites must have a "rect" and an optional "radius"
       attribute."""
    xdistance = left.rect.centerx - right.rect.centerx
    ydistance = left.rect.centery - right.rect.centery
    distancesquared = xdistance ** 2 + ydistance ** 2
    try:
        leftradiussquared = left.radius ** 2
    except AttributeError:
        leftrect = left.rect
        leftradiussquared = ( leftrect.width ** 2 + leftrect.height ** 2 ) / 4
    try:
        rightradiussquared = right.radius ** 2
    except AttributeError:
        rightrect = right.rect
        rightradiussquared = ( rightrect.width ** 2 + rightrect.height ** 2 ) / 4
    return distancesquared < leftradiussquared + rightradiussquared

class collide_circle_ratio( object ):
    """A callable class that checks for collisions between
       two sprites, using a scaled version of the sprites
       radius.

       Is created with a ratio, the instance is then intended
       to be passed as a collided callback function to the
       *collide functions."""
    
    def __init__( self, ratio ):
        """Creates a new collide_circle_ratio callable. ratio is
           expected to be a floating point value used to scale
           the underlying sprite radius before checking for
           collisions."""
        self.ratio = ratio
        # Constant value that folds in division for diameter to radius,
        # when calculating from a rect.
        self.halfratio = ratio ** 2 / 4.0

    def __call__( self, left, right ):
        """pygame.sprite.collide_circle_radio(ratio)(left, right) -> bool
           collision detection between two sprites, using scaled circles.

           Tests for collision between two sprites, by testing to
           see if two circles centered on the sprites overlap, after
           scaling the circles radius by the stored ratio. If
           the sprites have a "radius" attribute, that is used to
           create the circle, otherwise a circle is created that
           is big enough to completely enclose the sprites rect as
           given by the "rect" attribute. Intended to be passed as
           a collided callback function to the *collide functions.
           Sprites must have a "rect" and an optional "radius"
           attribute."""
        ratio = self.ratio
        xdistance = left.rect.centerx - right.rect.centerx
        ydistance = left.rect.centery - right.rect.centery
        distancesquared = xdistance ** 2 + ydistance ** 2
        # Optimize for not containing radius attribute, as if radius was
        # set consistently, would probably be using collide_circle instead.
        if hasattr( left, "radius" ):
            leftradiussquared = (left.radius * ratio) ** 2
            
            if hasattr( right, "radius" ):
                rightradiussquared = (right.radius * ratio) ** 2
            else:
                halfratio = self.halfratio
                rightrect = right.rect
                rightradiussquared = (rightrect.width ** 2 + rightrect.height ** 2) * halfratio
        else:
            halfratio = self.halfratio
            leftrect = left.rect
            leftradiussquared = (leftrect.width ** 2 + leftrect.height ** 2) * halfratio
            
            if hasattr( right, "radius" ):
                rightradiussquared = (right.radius * ratio) ** 2
            else:
                rightrect = right.rect
                rightradiussquared = (rightrect.width ** 2 + rightrect.height ** 2) * halfratio
        return distancesquared < leftradiussquared + rightradiussquared

def collide_mask(left, right):
    """pygame.sprite.collide_mask(left, right) -> bool
       collision detection between two sprites, using masks.

       Tests for collision between two sprites, by testing if
       thier bitmasks overlap. If the sprites have a "mask"
       attribute, that is used as the mask, otherwise a mask is
       created from the sprite image. Intended to be passed as
       a collided callback function to the *collide functions.
       Sprites must have a "rect" and an optional "mask"
       attribute."""
    xoffset = right.rect[0] - left.rect[0]
    yoffset = right.rect[1] - left.rect[1]
    try:
        leftmask = left.mask
    except AttributeError:
        leftmask = from_surface(left.image)
    try:
        rightmask = right.mask
    except AttributeError:
        rightmask = from_surface(right.image)
    return leftmask.overlap(rightmask, (xoffset, yoffset))

def spritecollide(sprite, group, dokill, collided = None):
    """pygame.sprite.spritecollide(sprite, group, dokill) -> list
       collision detection between sprite and group

       given a sprite and a group of sprites, this will
       return a list of all the sprites that intersect
       the given sprite.
       if the dokill argument is true, the sprites that
       do collide will be automatically removed from all
       groups.
       collided is a callback function used to calculate if
       two sprites are colliding. it should take two sprites
       as values, and return a boolean value indicating if
       they are colliding. if collided is not passed, all
       sprites must have a "rect" value, which is a
       rectangle of the sprite area, which will be used
       to calculate the collision."""
    crashed = []
    if collided is None:
        # Special case old behaviour for speed.
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
    else:
        if dokill:
            for s in group.sprites():
                if collided(sprite, s):
                    s.kill()
                    crashed.append(s)
        else:
            for s in group:
                if collided(sprite, s):
                    crashed.append(s)
    return crashed

def groupcollide(groupa, groupb, dokilla, dokillb, collided = None):
    """pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb) -> dict
       collision detection between group and group

       given two groups, this will find the intersections
       between all sprites in each group. it returns a
       dictionary of all sprites in the first group that
       collide. the value for each item in the dictionary
       is a list of the sprites in the second group it
       collides with. the two dokill arguments control if
       the sprites from either group will be automatically
       removed from all groups.
       collided is a callback function used to calculate if
       two sprites are colliding. it should take two sprites
       as values, and return a boolean value indicating if
       they are colliding. if collided is not passed, all
       sprites must have a "rect" value, which is a
       rectangle of the sprite area, which will be used
       to calculate the collision."""
    crashed = {}
    SC = spritecollide
    if dokilla:
        for s in groupa.sprites():
            c = SC(s, groupb, dokillb, collided)
            if c:
                crashed[s] = c
                s.kill()
    else:
        for s in groupa:
            c = SC(s, groupb, dokillb, collided)
            if c:
                crashed[s] = c
    return crashed

def spritecollideany(sprite, group, collided = None):
    """pygame.sprite.spritecollideany(sprite, group) -> sprite
       finds any sprites that collide

       given a sprite and a group of sprites, this will
       return return any single sprite that collides with
       with the given sprite. If there are no collisions
       this returns None.

       if you don't need all the features of the
       spritecollide function, this function will be a
       bit quicker.

       collided is a callback function used to calculate if
       two sprites are colliding. it should take two sprites
       as values, and return a boolean value indicating if
       they are colliding. if collided is not passed, all
       sprites must have a "rect" value, which is a
       rectangle of the sprite area, which will be used
       to calculate the collision."""
    if collided is None:
        # Special case old behaviour for speed.
        spritecollide = sprite.rect.colliderect
        for s in group:
            if spritecollide(s.rect):
                return s
    else:
        for s in group:
            if collided(sprite, s):
                return s
    return None
