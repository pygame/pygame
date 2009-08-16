/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMESPRITE "pygame module with basic game object classes"

#define DOC_PYGAMESPRITESPRITE "pygame.sprite.Sprite(*groups): return Sprite\nsimple base class for visible game objects"

#define DOC_SPRITEUPDATE "Sprite.update(*args):\nmethod to control sprite behavior"

#define DOC_SPRITEADD "Sprite.add(*groups): return None\nadd the sprite to groups"

#define DOC_SPRITEREMOVE "Sprite.remove(*groups): return None\nremove the sprite from groups"

#define DOC_SPRITEKILL "Sprite.kill(): return None\nremove the Sprite from all Groups"

#define DOC_SPRITEALIVE "Sprite.alive(): return bool\ndoes the sprite belong to any groups"

#define DOC_SPRITEGROUPS "Sprite.groups(): return group_list\nlist of Groups that contain this Sprite"

#define DOC_PYGAMESPRITEDIRTYSPRITE "pygame.sprite.DirtySprite(*groups): return DirtySprite\na more featureful subclass of Sprite with more attributes"

#define DOC_ ""

#define DOC_PYGAMESPRITEGROUP "pygame.sprite.Group(*sprites): return Group\ncontainer class for many Sprites"

#define DOC_GROUPSPRITES "Group.sprites(): return sprite_list\nlist of the Sprites this Group contains"

#define DOC_GROUPCOPY "Group.copy(): return Group\nduplicate the Group"

#define DOC_GROUPADD "Group.add(*sprites): return None\nadd Sprites to this Group"

#define DOC_GROUPREMOVE "Group.remove(*sprites): return None\nremove Sprites from the Group"

#define DOC_GROUPHAS "Group.has(*sprites): return None\ntest if a Group contains Sprites"

#define DOC_GROUPUPDATE "Group.update(*args): return None\ncall the update method on contained Sprites"

#define DOC_GROUPDRAW "Group.draw(Surface): return None\nblit the Sprite images"

#define DOC_GROUPCLEAR "Group.clear(Surface_dest, background): return None\ndraw a background over the Sprites"

#define DOC_GROUPEMPTY "Group.empty(): return None\nremove all Sprites"

#define DOC_PYGAMESPRITERENDERUPDATES "pygame.sprite.RenderUpdates(*sprites): return RenderUpdates\nGroup class that tracks dirty updates"

#define DOC_RENDERUPDATESDRAW "RenderUpdates.draw(surface): return Rect_list\nblit the Sprite images and track changed areas"

#define DOC_PYGAMESPRITEORDEREDUPDATES "pygame.sprite.OrderedUpdates(*spites): return OrderedUpdates\nRenderUpdates class that draws Sprites in order of addition"

#define DOC_PYGAMESPRITELAYEREDUPDATES "pygame.sprite.LayeredUpdates(*spites, **kwargs): return LayeredUpdates\nLayeredUpdates Group handles layers, that draws like OrderedUpdates."

#define DOC_LAYEREDUPDATESADD "LayeredUpdates.add(*sprites, **kwargs): return None\nadd a sprite or sequence of sprites to a group"

#define DOC_LAYEREDUPDATESSPRITES "LayeredUpdates.sprites(): return sprites\nreturns a ordered list of sprites (first back, last top)."

#define DOC_LAYEREDUPDATESDRAW "LayeredUpdates.draw(surface): return Rect_list\ndraw all sprites in the right order onto the passed surface."

#define DOC_LAYEREDUPDATESGETSPRITESAT "LayeredUpdates.get_sprites_at(pos): return colliding_sprites\nreturns a list with all sprites at that position."

#define DOC_LAYEREDUPDATESGETSPRITE "LayeredUpdates.get_sprite(idx): return sprite\nreturns the sprite at the index idx from the groups sprites"

#define DOC_LAYEREDUPDATESREMOVESPRITESOFLAYER "LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites\nremoves all sprites from a layer and returns them as a list."

#define DOC_LAYEREDUPDATESLAYERS "LayeredUpdates.layers(): return layers\nreturns a list of layers defined (unique), sorted from botton up."

#define DOC_LAYEREDUPDATESCHANGELAYER "LayeredUpdates.change_layer(sprite, new_layer): return None\nchanges the layer of the sprite"

#define DOC_LAYEREDUPDATESGETLAYEROFSPRITE "LayeredUpdates.get_layer_of_sprite(sprite): return layer\nreturns the layer that sprite is currently in."

#define DOC_LAYEREDUPDATESGETTOPLAYER "LayeredUpdates.get_top_layer(): return layer\nreturns the top layer"

#define DOC_LAYEREDUPDATESGETBOTTOMLAYER "LayeredUpdates.get_bottom_layer(): return layer\nreturns the bottom layer"

#define DOC_LAYEREDUPDATESMOVETOFRONT "LayeredUpdates.move_to_front(sprite): return None\nbrings the sprite to front layer"

#define DOC_LAYEREDUPDATESMOVETOBACK "LayeredUpdates.move_to_back(sprite): return None\nmoves the sprite to the bottom layer"

#define DOC_LAYEREDUPDATESGETTOPSPRITE "LayeredUpdates.get_top_sprite(): return Sprite\nreturns the topmost sprite"

#define DOC_LAYEREDUPDATESGETSPRITESFROMLAYER "LayeredUpdates.get_sprites_from_layer(layer): return sprites\nreturns all sprites from a layer, ordered by how they where added"

#define DOC_LAYEREDUPDATESSWITCHLAYER "LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None\nswitches the sprites from layer1 to layer2"

#define DOC_PYGAMESPRITELAYEREDDIRTY "pygame.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty\nLayeredDirty Group is for DirtySprites.  Subclasses LayeredUpdates."

#define DOC_LAYEREDDIRTYDRAW "LayeredDirty.draw(surface, bgd=None): return Rect_list\ndraw all sprites in the right order onto the passed surface."

#define DOC_LAYEREDDIRTYCLEAR "LayeredDirty.clear(surface, bgd): return None\nused to set background"

#define DOC_LAYEREDDIRTYREPAINTRECT "LayeredDirty.repaint_rect(screen_rect): return None\nrepaints the given area"

#define DOC_LAYEREDDIRTYSETCLIP "LayeredDirty.set_clip(screen_rect=None): return None\nclip the area where to draw. Just pass None (default) to reset the clip"

#define DOC_LAYEREDDIRTYGETCLIP "LayeredDirty.get_clip(): return Rect\nclip the area where to draw. Just pass None (default) to reset the clip"

#define DOC_LAYEREDDIRTYCHANGELAYER "change_layer(sprite, new_layer): return None\nchanges the layer of the sprite"

#define DOC_LAYEREDDIRTYSETTIMINGTRESHOLD "set_timing_treshold(time_ms): return None\nsets the treshold in milliseconds"

#define DOC_PYGAMESPRITEGROUPSINGLE "pygame.sprite.GroupSingle(sprite=None): return GroupSingle\nGroup container that holds a single Sprite"

#define DOC_PYGAMESPRITESPRITECOLLIDE "pygame.sprite.spritecollide(sprite, group, dokill, collided = None): return Sprite_list\nfind Sprites in a Group that intersect another Sprite"

#define DOC_PYGAMESPRITECOLLIDERECT "pygame.sprite.collide_rect(left, right): return bool\ncollision detection between two sprites, using rects."

#define DOC_PYGAMESPRITECOLLIDERECTRATIO "pygame.sprite.collide_rect_ratio(ratio): return collided_callable\ncollision detection between two sprites, using rects scaled to a ratio."

#define DOC_PYGAMESPRITECOLLIDECIRCLE "pygame.sprite.collide_circle(left, right): return bool\ncollision detection between two sprites, using circles."

#define DOC_PYGAMESPRITECOLLIDECIRCLERATIO "pygame.sprite.collide_circle_ratio(ratio): return collided_callable\ncollision detection between two sprites, using circles scaled to a ratio."

#define DOC_PYGAMESPRITECOLLIDEMASK "pygame.sprite.collide_mask(SpriteLeft, SpriteRight): return bool\ncollision detection between two sprites, using masks."

#define DOC_PYGAMESPRITEGROUPCOLLIDE "pygame.sprite.groupcollide(group1, group2, dokill1, dokill2): return Sprite_dict\nfind all Sprites that collide between two Groups"

#define DOC_PYGAMESPRITESPRITECOLLIDEANY "pygame.sprite.spritecollideany(sprite, group): return bool\nsimple test if a Sprite intersects anything in a Group"

#define DOC_ ""



/* Docs in a comments... slightly easier to read. */


/*

pygame.sprite
 pygame module with basic game object classes



pygame.sprite.Sprite
 pygame.sprite.Sprite(*groups): return Sprite
simple base class for visible game objects



Sprite.update
 Sprite.update(*args):
method to control sprite behavior



Sprite.add
 Sprite.add(*groups): return None
add the sprite to groups



Sprite.remove
 Sprite.remove(*groups): return None
remove the sprite from groups



Sprite.kill
 Sprite.kill(): return None
remove the Sprite from all Groups



Sprite.alive
 Sprite.alive(): return bool
does the sprite belong to any groups



Sprite.groups
 Sprite.groups(): return group_list
list of Groups that contain this Sprite



pygame.sprite.DirtySprite
 pygame.sprite.DirtySprite(*groups): return DirtySprite
a more featureful subclass of Sprite with more attributes




 



pygame.sprite.Group
 pygame.sprite.Group(*sprites): return Group
container class for many Sprites



Group.sprites
 Group.sprites(): return sprite_list
list of the Sprites this Group contains



Group.copy
 Group.copy(): return Group
duplicate the Group



Group.add
 Group.add(*sprites): return None
add Sprites to this Group



Group.remove
 Group.remove(*sprites): return None
remove Sprites from the Group



Group.has
 Group.has(*sprites): return None
test if a Group contains Sprites



Group.update
 Group.update(*args): return None
call the update method on contained Sprites



Group.draw
 Group.draw(Surface): return None
blit the Sprite images



Group.clear
 Group.clear(Surface_dest, background): return None
draw a background over the Sprites



Group.empty
 Group.empty(): return None
remove all Sprites



pygame.sprite.RenderUpdates
 pygame.sprite.RenderUpdates(*sprites): return RenderUpdates
Group class that tracks dirty updates



RenderUpdates.draw
 RenderUpdates.draw(surface): return Rect_list
blit the Sprite images and track changed areas



pygame.sprite.OrderedUpdates
 pygame.sprite.OrderedUpdates(*spites): return OrderedUpdates
RenderUpdates class that draws Sprites in order of addition



pygame.sprite.LayeredUpdates
 pygame.sprite.LayeredUpdates(*spites, **kwargs): return LayeredUpdates
LayeredUpdates Group handles layers, that draws like OrderedUpdates.



LayeredUpdates.add
 LayeredUpdates.add(*sprites, **kwargs): return None
add a sprite or sequence of sprites to a group



LayeredUpdates.sprites
 LayeredUpdates.sprites(): return sprites
returns a ordered list of sprites (first back, last top).



LayeredUpdates.draw
 LayeredUpdates.draw(surface): return Rect_list
draw all sprites in the right order onto the passed surface.



LayeredUpdates.get_sprites_at
 LayeredUpdates.get_sprites_at(pos): return colliding_sprites
returns a list with all sprites at that position.



LayeredUpdates.get_sprite
 LayeredUpdates.get_sprite(idx): return sprite
returns the sprite at the index idx from the groups sprites



LayeredUpdates.remove_sprites_of_layer
 LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites
removes all sprites from a layer and returns them as a list.



LayeredUpdates.layers
 LayeredUpdates.layers(): return layers
returns a list of layers defined (unique), sorted from botton up.



LayeredUpdates.change_layer
 LayeredUpdates.change_layer(sprite, new_layer): return None
changes the layer of the sprite



LayeredUpdates.get_layer_of_sprite
 LayeredUpdates.get_layer_of_sprite(sprite): return layer
returns the layer that sprite is currently in.



LayeredUpdates.get_top_layer
 LayeredUpdates.get_top_layer(): return layer
returns the top layer



LayeredUpdates.get_bottom_layer
 LayeredUpdates.get_bottom_layer(): return layer
returns the bottom layer



LayeredUpdates.move_to_front
 LayeredUpdates.move_to_front(sprite): return None
brings the sprite to front layer



LayeredUpdates.move_to_back
 LayeredUpdates.move_to_back(sprite): return None
moves the sprite to the bottom layer



LayeredUpdates.get_top_sprite
 LayeredUpdates.get_top_sprite(): return Sprite
returns the topmost sprite



LayeredUpdates.get_sprites_from_layer
 LayeredUpdates.get_sprites_from_layer(layer): return sprites
returns all sprites from a layer, ordered by how they where added



LayeredUpdates.switch_layer
 LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None
switches the sprites from layer1 to layer2



pygame.sprite.LayeredDirty
 pygame.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty
LayeredDirty Group is for DirtySprites.  Subclasses LayeredUpdates.



LayeredDirty.draw
 LayeredDirty.draw(surface, bgd=None): return Rect_list
draw all sprites in the right order onto the passed surface.



LayeredDirty.clear
 LayeredDirty.clear(surface, bgd): return None
used to set background



LayeredDirty.repaint_rect
 LayeredDirty.repaint_rect(screen_rect): return None
repaints the given area



LayeredDirty.set_clip
 LayeredDirty.set_clip(screen_rect=None): return None
clip the area where to draw. Just pass None (default) to reset the clip



LayeredDirty.get_clip
 LayeredDirty.get_clip(): return Rect
clip the area where to draw. Just pass None (default) to reset the clip



LayeredDirty.change_layer
 change_layer(sprite, new_layer): return None
changes the layer of the sprite



LayeredDirty.set_timing_treshold
 set_timing_treshold(time_ms): return None
sets the treshold in milliseconds



pygame.sprite.GroupSingle
 pygame.sprite.GroupSingle(sprite=None): return GroupSingle
Group container that holds a single Sprite



pygame.sprite.spritecollide
 pygame.sprite.spritecollide(sprite, group, dokill, collided = None): return Sprite_list
find Sprites in a Group that intersect another Sprite



pygame.sprite.collide_rect
 pygame.sprite.collide_rect(left, right): return bool
collision detection between two sprites, using rects.



pygame.sprite.collide_rect_ratio
 pygame.sprite.collide_rect_ratio(ratio): return collided_callable
collision detection between two sprites, using rects scaled to a ratio.



pygame.sprite.collide_circle
 pygame.sprite.collide_circle(left, right): return bool
collision detection between two sprites, using circles.



pygame.sprite.collide_circle_ratio
 pygame.sprite.collide_circle_ratio(ratio): return collided_callable
collision detection between two sprites, using circles scaled to a ratio.



pygame.sprite.collide_mask
 pygame.sprite.collide_mask(SpriteLeft, SpriteRight): return bool
collision detection between two sprites, using masks.



pygame.sprite.groupcollide
 pygame.sprite.groupcollide(group1, group2, dokill1, dokill2): return Sprite_dict
find all Sprites that collide between two Groups



pygame.sprite.spritecollideany
 pygame.sprite.spritecollideany(sprite, group): return bool
simple test if a Sprite intersects anything in a Group




 



*/

