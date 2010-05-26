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

"""Specialized rendering- and layer-aware groups"""

import pygame2.compat
pygame2.compat.deprecation \
    ("The sprite package is deprecated and will change in future versions")

try:
    from pygame2.sdl.time import get_ticks
    __hassdl = True
except:
    __hassdl = False
    pass

from pygame2.sprite.groups import AbstractGroup
from pygame2.sprite.sprites import Sprite, DirtySprite

class LayeredUpdates(AbstractGroup):
    """LayeredUpdates Group handles layers, that draws like OrderedUpdates.
    pygame2.sprite.LayeredUpdates(*spites, **kwargs): return LayeredUpdates
    
    This group is fully compatible with pygame2.sprite.Sprite.

    New in pygame 1.8.0
    """
    
    def __init__(self, *sprites, **kwargs):
        """
        You can set the default layer through kwargs using 'default_layer'
        and an integer for the layer. The default layer is 0.
        
        If the sprite you add has an attribute layer then that layer will
        be used.
        If the **kwarg contains 'layer' then the sprites passed will be 
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
        """add a sprite or sequence of sprites to a group
        LayeredUpdates.add(*sprites, **kwargs): return None

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
        """returns a ordered list of sprites (first back, last top).
        LayeredUpdates.sprites(): return sprites
        """
        return list(self._spritelist)
    
    def draw(self, surface):
        """draw all sprites in the right order onto the passed surface.
        LayeredUpdates.draw(surface): return Rect_list
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
        """returns a list with all sprites at that position.
        LayeredUpdates.get_sprites_at(pos): return colliding_sprites

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
        """returns the sprite at the index idx from the groups sprites
        LayeredUpdates.get_sprite(idx): return sprite

        Raises IndexOutOfBounds if the idx is not within range.
        """
        return self._spritelist[idx]
    
    def remove_sprites_of_layer(self, layer_nr):
        """removes all sprites from a layer and returns them as a list
        LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites
        """
        sprites = self.get_sprites_from_layer(layer_nr)
        self.remove(sprites)
        return sprites
        

    #---# layer methods
    def layers(self):
        """returns a list of layers defined (unique), sorted from botton up.
        LayeredUpdates.layers(): return layers
        """
        layers = set()
        for layer in self._spritelayers.values():
            layers.add(layer)
        return list(layers)

    def change_layer(self, sprite, new_layer):
        """changes the layer of the sprite
        LayeredUpdates.change_layer(sprite, new_layer): return None

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
        """returns the top layer
        LayeredUpdates.get_top_layer(): return layer
        """
        return self._spritelayers[self._spritelist[-1]]
    
    def get_bottom_layer(self):
        """returns the bottom layer
        LayeredUpdates.get_bottom_layer(): return layer
        """
        return self._spritelayers[self._spritelist[0]]
    
    def move_to_front(self, sprite):
        """brings the sprite to front layer
        LayeredUpdates.move_to_front(sprite): return None

        Brings the sprite to front, changing sprite layer to topmost layer
        (added at the end of that layer).
        """
        self.change_layer(sprite, self.get_top_layer())
        
    def move_to_back(self, sprite):
        """moves the sprite to the bottom layer
        LayeredUpdates.move_to_back(sprite): return None

        Moves the sprite to the bottom layer, moving it behind
        all other layers and adding one additional layer.
        """
        self.change_layer(sprite, self.get_bottom_layer()-1)
        
    def get_top_sprite(self):
        """returns the topmost sprite
        LayeredUpdates.get_top_sprite(): return Sprite
        """
        return self._spritelist[-1]
    
    def get_sprites_from_layer(self, layer):
        """returns all sprites from a layer, ordered by how they where added
        LayeredUpdates.get_sprites_from_layer(layer): return sprites

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
        """switches the sprites from layer1 to layer2
        LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None

        The layers number must exist, it is not checked.
        """
        sprites1 = self.remove_sprites_of_layer(layer1_nr)
        for spr in self.get_sprites_from_layer(layer2_nr):
            self.change_layer(spr, layer1_nr)
        self.add(sprites1, layer=layer2_nr)


class LayeredDirty(LayeredUpdates):
    """LayeredDirty Group is for DirtySprites.  Subclasses LayeredUpdates.
    pygame2.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty
        
    This group requires pygame2.sprite.DirtySprite or any sprite that 
    has the following attributes: 
        image, rect, dirty, visible, blendmode (see doc of DirtySprite).

    It uses the dirty flag technique and is therefore faster than the 
    pygame2.sprite.RenderUpdates if you have many static sprites.  It 
    also switches automatically between dirty rect update and full 
    screen drawing, so you do no have to worry what would be faster.

    Same as for the pygame2.sprite.Group.
    You can specify some additional attributes through kwargs:
        _use_update: True/False   default is False
        _default_layer: default layer where sprites without a layer are added.
        _time_threshold: treshold time for switching between dirty rect mode 
            and fullscreen mode, defaults to 1000./80  == 1000./fps

    New in pygame 1.8.0
    """
    
    def __init__(self, *sprites, **kwargs):
        """Same as for the pygame2.sprite.Group.
        pygame2.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty

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
        """Do not use this method directly. It is used by the group to add a
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
        """draw all sprites in the right order onto the passed surface.
        LayeredDirty.draw(surface, bgd=None): return Rect_list

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
        """used to set background
        Group.clear(surface, bgd): return None
        """
        self._bgd = bgd

    def repaint_rect(self, screen_rect): 
        """repaints the given area
        LayeredDirty.repaint_rect(screen_rect): return None

        screen_rect is in screencoordinates.
        """
        self.lostsprites.append(screen_rect.clip(self._clip))
        
    def set_clip(self, screen_rect=None):
        """ clip the area where to draw. Just pass None (default) to reset the clip
        LayeredDirty.set_clip(screen_rect=None): return None
        """
        if screen_rect is None:
            self._clip = pygame2.display.get_surface().get_rect()
        else:
            self._clip = screen_rect
        self._use_update = False
        
    def get_clip(self):
        """clip the area where to draw. Just pass None (default) to reset the clip
        LayeredDirty.get_clip(): return Rect
        """
        return self._clip
    
    def change_layer(self, sprite, new_layer):
        """changes the layer of the sprite
        change_layer(sprite, new_layer): return None

        sprite must have been added to the renderer. It is not checked.
        """
        LayeredUpdates.change_layer(self, sprite, new_layer)
        if sprite.dirty == 0:
            sprite.dirty = 1
            

    def set_timing_treshold(self, time_ms):
        """sets the treshold in milliseconds
        set_timing_treshold(time_ms): return None

        Default is 1000./80 where 80 is the fps I want to switch to full screen mode.
        """
        self._time_threshold = time_ms
