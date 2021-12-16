/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMESPRITE "pygame module with basic game object classes"
#define DOC_PYGAMESPRITESPRITE "Sprite(*groups) -> Sprite\nSimple base class for visible game objects."
#define DOC_SPRITEUPDATE "update(*args, **kwargs) -> None\nmethod to control sprite behavior"
#define DOC_SPRITEADD "add(*groups) -> None\nadd the sprite to groups"
#define DOC_SPRITEREMOVE "remove(*groups) -> None\nremove the sprite from groups"
#define DOC_SPRITEKILL "kill() -> None\nremove the Sprite from all Groups"
#define DOC_SPRITEALIVE "alive() -> bool\ndoes the sprite belong to any groups"
#define DOC_SPRITEGROUPS "groups() -> group_list\nlist of Groups that contain this Sprite"
#define DOC_PYGAMESPRITEDIRTYSPRITE "DirtySprite(*groups) -> DirtySprite\nA subclass of Sprite with more attributes and features."
#define DOC_PYGAMESPRITEGROUP "Group(*sprites) -> Group\nA container class to hold and manage multiple Sprite objects."
#define DOC_GROUPSPRITES "sprites() -> sprite_list\nlist of the Sprites this Group contains"
#define DOC_GROUPCOPY "copy() -> Group\nduplicate the Group"
#define DOC_GROUPADD "add(*sprites) -> None\nadd Sprites to this Group"
#define DOC_GROUPREMOVE "remove(*sprites) -> None\nremove Sprites from the Group"
#define DOC_GROUPHAS "has(*sprites) -> bool\ntest if a Group contains Sprites"
#define DOC_GROUPUPDATE "update(*args, **kwargs) -> None\ncall the update method on contained Sprites"
#define DOC_GROUPDRAW "draw(Surface) -> List[Rect]\nblit the Sprite images"
#define DOC_GROUPCLEAR "clear(Surface_dest, background) -> None\ndraw a background over the Sprites"
#define DOC_GROUPEMPTY "empty() -> None\nremove all Sprites"
#define DOC_PYGAMESPRITERENDERPLAIN "Same as pygame.sprite.Group"
#define DOC_PYGAMESPRITERENDERCLEAR "Same as pygame.sprite.Group"
#define DOC_PYGAMESPRITERENDERUPDATES "RenderUpdates(*sprites) -> RenderUpdates\nGroup sub-class that tracks dirty updates."
#define DOC_RENDERUPDATESDRAW "draw(surface) -> Rect_list\nblit the Sprite images and track changed areas"
#define DOC_PYGAMESPRITEORDEREDUPDATES "OrderedUpdates(*spites) -> OrderedUpdates\nRenderUpdates sub-class that draws Sprites in order of addition."
#define DOC_PYGAMESPRITELAYEREDUPDATES "LayeredUpdates(*spites, **kwargs) -> LayeredUpdates\nLayeredUpdates is a sprite group that handles layers and draws like OrderedUpdates."
#define DOC_LAYEREDUPDATESADD "add(*sprites, **kwargs) -> None\nadd a sprite or sequence of sprites to a group"
#define DOC_LAYEREDUPDATESSPRITES "sprites() -> sprites\nreturns a ordered list of sprites (first back, last top)."
#define DOC_LAYEREDUPDATESDRAW "draw(surface) -> Rect_list\ndraw all sprites in the right order onto the passed surface."
#define DOC_LAYEREDUPDATESGETSPRITESAT "get_sprites_at(pos) -> colliding_sprites\nreturns a list with all sprites at that position."
#define DOC_LAYEREDUPDATESGETSPRITE "get_sprite(idx) -> sprite\nreturns the sprite at the index idx from the groups sprites"
#define DOC_LAYEREDUPDATESREMOVESPRITESOFLAYER "remove_sprites_of_layer(layer_nr) -> sprites\nremoves all sprites from a layer and returns them as a list."
#define DOC_LAYEREDUPDATESLAYERS "layers() -> layers\nreturns a list of layers defined (unique), sorted from bottom up."
#define DOC_LAYEREDUPDATESCHANGELAYER "change_layer(sprite, new_layer) -> None\nchanges the layer of the sprite"
#define DOC_LAYEREDUPDATESGETLAYEROFSPRITE "get_layer_of_sprite(sprite) -> layer\nreturns the layer that sprite is currently in."
#define DOC_LAYEREDUPDATESGETTOPLAYER "get_top_layer() -> layer\nreturns the top layer"
#define DOC_LAYEREDUPDATESGETBOTTOMLAYER "get_bottom_layer() -> layer\nreturns the bottom layer"
#define DOC_LAYEREDUPDATESMOVETOFRONT "move_to_front(sprite) -> None\nbrings the sprite to front layer"
#define DOC_LAYEREDUPDATESMOVETOBACK "move_to_back(sprite) -> None\nmoves the sprite to the bottom layer"
#define DOC_LAYEREDUPDATESGETTOPSPRITE "get_top_sprite() -> Sprite\nreturns the topmost sprite"
#define DOC_LAYEREDUPDATESGETSPRITESFROMLAYER "get_sprites_from_layer(layer) -> sprites\nreturns all sprites from a layer, ordered by how they where added"
#define DOC_LAYEREDUPDATESSWITCHLAYER "switch_layer(layer1_nr, layer2_nr) -> None\nswitches the sprites from layer1 to layer2"
#define DOC_PYGAMESPRITELAYEREDDIRTY "LayeredDirty(*spites, **kwargs) -> LayeredDirty\nLayeredDirty group is for DirtySprite objects.  Subclasses LayeredUpdates."
#define DOC_LAYEREDDIRTYDRAW "draw(surface, bgd=None) -> Rect_list\ndraw all sprites in the right order onto the passed surface."
#define DOC_LAYEREDDIRTYCLEAR "clear(surface, bgd) -> None\nused to set background"
#define DOC_LAYEREDDIRTYREPAINTRECT "repaint_rect(screen_rect) -> None\nrepaints the given area"
#define DOC_LAYEREDDIRTYSETCLIP "set_clip(screen_rect=None) -> None\nclip the area where to draw. Just pass None (default) to reset the clip"
#define DOC_LAYEREDDIRTYGETCLIP "get_clip() -> Rect\nclip the area where to draw. Just pass None (default) to reset the clip"
#define DOC_LAYEREDDIRTYCHANGELAYER "change_layer(sprite, new_layer) -> None\nchanges the layer of the sprite"
#define DOC_LAYEREDDIRTYSETTIMINGTRESHOLD "set_timing_treshold(time_ms) -> None\nsets the threshold in milliseconds"
#define DOC_LAYEREDDIRTYSETTIMINGTHRESHOLD "set_timing_threshold(time_ms) -> None\nsets the threshold in milliseconds"
#define DOC_PYGAMESPRITEGROUPSINGLE "GroupSingle(sprite=None) -> GroupSingle\nGroup container that holds a single sprite."
#define DOC_PYGAMESPRITESPRITECOLLIDE "spritecollide(sprite, group, dokill, collided = None) -> Sprite_list\nFind sprites in a group that intersect another sprite."
#define DOC_PYGAMESPRITECOLLIDERECT "collide_rect(left, right) -> bool\nCollision detection between two sprites, using rects."
#define DOC_PYGAMESPRITECOLLIDERECTRATIO "collide_rect_ratio(ratio) -> collided_callable\nCollision detection between two sprites, using rects scaled to a ratio."
#define DOC_PYGAMESPRITECOLLIDECIRCLE "collide_circle(left, right) -> bool\nCollision detection between two sprites, using circles."
#define DOC_PYGAMESPRITECOLLIDECIRCLERATIO "collide_circle_ratio(ratio) -> collided_callable\nCollision detection between two sprites, using circles scaled to a ratio."
#define DOC_PYGAMESPRITECOLLIDEMASK "collide_mask(sprite1, sprite2) -> (int, int)\ncollide_mask(sprite1, sprite2) -> None\nCollision detection between two sprites, using masks."
#define DOC_PYGAMESPRITEGROUPCOLLIDE "groupcollide(group1, group2, dokill1, dokill2, collided = None) -> Sprite_dict\nFind all sprites that collide between two groups."
#define DOC_PYGAMESPRITESPRITECOLLIDEANY "spritecollideany(sprite, group, collided = None) -> Sprite\nspritecollideany(sprite, group, collided = None) -> None\nSimple test if a sprite intersects anything in a group."


/* Docs in a comment... slightly easier to read. */

/*

pygame.sprite
pygame module with basic game object classes

pygame.sprite.Sprite
 Sprite(*groups) -> Sprite
Simple base class for visible game objects.

pygame.sprite.Sprite.update
 update(*args, **kwargs) -> None
method to control sprite behavior

pygame.sprite.Sprite.add
 add(*groups) -> None
add the sprite to groups

pygame.sprite.Sprite.remove
 remove(*groups) -> None
remove the sprite from groups

pygame.sprite.Sprite.kill
 kill() -> None
remove the Sprite from all Groups

pygame.sprite.Sprite.alive
 alive() -> bool
does the sprite belong to any groups

pygame.sprite.Sprite.groups
 groups() -> group_list
list of Groups that contain this Sprite

pygame.sprite.DirtySprite
 DirtySprite(*groups) -> DirtySprite
A subclass of Sprite with more attributes and features.

pygame.sprite.Group
 Group(*sprites) -> Group
A container class to hold and manage multiple Sprite objects.

pygame.sprite.Group.sprites
 sprites() -> sprite_list
list of the Sprites this Group contains

pygame.sprite.Group.copy
 copy() -> Group
duplicate the Group

pygame.sprite.Group.add
 add(*sprites) -> None
add Sprites to this Group

pygame.sprite.Group.remove
 remove(*sprites) -> None
remove Sprites from the Group

pygame.sprite.Group.has
 has(*sprites) -> bool
test if a Group contains Sprites

pygame.sprite.Group.update
 update(*args, **kwargs) -> None
call the update method on contained Sprites

pygame.sprite.Group.draw
 draw(Surface) -> List[Rect]
blit the Sprite images

pygame.sprite.Group.clear
 clear(Surface_dest, background) -> None
draw a background over the Sprites

pygame.sprite.Group.empty
 empty() -> None
remove all Sprites

pygame.sprite.RenderPlain
Same as pygame.sprite.Group

pygame.sprite.RenderClear
Same as pygame.sprite.Group

pygame.sprite.RenderUpdates
 RenderUpdates(*sprites) -> RenderUpdates
Group sub-class that tracks dirty updates.

pygame.sprite.RenderUpdates.draw
 draw(surface) -> Rect_list
blit the Sprite images and track changed areas

pygame.sprite.OrderedUpdates
 OrderedUpdates(*spites) -> OrderedUpdates
RenderUpdates sub-class that draws Sprites in order of addition.

pygame.sprite.LayeredUpdates
 LayeredUpdates(*spites, **kwargs) -> LayeredUpdates
LayeredUpdates is a sprite group that handles layers and draws like OrderedUpdates.

pygame.sprite.LayeredUpdates.add
 add(*sprites, **kwargs) -> None
add a sprite or sequence of sprites to a group

pygame.sprite.LayeredUpdates.sprites
 sprites() -> sprites
returns a ordered list of sprites (first back, last top).

pygame.sprite.LayeredUpdates.draw
 draw(surface) -> Rect_list
draw all sprites in the right order onto the passed surface.

pygame.sprite.LayeredUpdates.get_sprites_at
 get_sprites_at(pos) -> colliding_sprites
returns a list with all sprites at that position.

pygame.sprite.LayeredUpdates.get_sprite
 get_sprite(idx) -> sprite
returns the sprite at the index idx from the groups sprites

pygame.sprite.LayeredUpdates.remove_sprites_of_layer
 remove_sprites_of_layer(layer_nr) -> sprites
removes all sprites from a layer and returns them as a list.

pygame.sprite.LayeredUpdates.layers
 layers() -> layers
returns a list of layers defined (unique), sorted from bottom up.

pygame.sprite.LayeredUpdates.change_layer
 change_layer(sprite, new_layer) -> None
changes the layer of the sprite

pygame.sprite.LayeredUpdates.get_layer_of_sprite
 get_layer_of_sprite(sprite) -> layer
returns the layer that sprite is currently in.

pygame.sprite.LayeredUpdates.get_top_layer
 get_top_layer() -> layer
returns the top layer

pygame.sprite.LayeredUpdates.get_bottom_layer
 get_bottom_layer() -> layer
returns the bottom layer

pygame.sprite.LayeredUpdates.move_to_front
 move_to_front(sprite) -> None
brings the sprite to front layer

pygame.sprite.LayeredUpdates.move_to_back
 move_to_back(sprite) -> None
moves the sprite to the bottom layer

pygame.sprite.LayeredUpdates.get_top_sprite
 get_top_sprite() -> Sprite
returns the topmost sprite

pygame.sprite.LayeredUpdates.get_sprites_from_layer
 get_sprites_from_layer(layer) -> sprites
returns all sprites from a layer, ordered by how they where added

pygame.sprite.LayeredUpdates.switch_layer
 switch_layer(layer1_nr, layer2_nr) -> None
switches the sprites from layer1 to layer2

pygame.sprite.LayeredDirty
 LayeredDirty(*spites, **kwargs) -> LayeredDirty
LayeredDirty group is for DirtySprite objects.  Subclasses LayeredUpdates.

pygame.sprite.LayeredDirty.draw
 draw(surface, bgd=None) -> Rect_list
draw all sprites in the right order onto the passed surface.

pygame.sprite.LayeredDirty.clear
 clear(surface, bgd) -> None
used to set background

pygame.sprite.LayeredDirty.repaint_rect
 repaint_rect(screen_rect) -> None
repaints the given area

pygame.sprite.LayeredDirty.set_clip
 set_clip(screen_rect=None) -> None
clip the area where to draw. Just pass None (default) to reset the clip

pygame.sprite.LayeredDirty.get_clip
 get_clip() -> Rect
clip the area where to draw. Just pass None (default) to reset the clip

pygame.sprite.LayeredDirty.change_layer
 change_layer(sprite, new_layer) -> None
changes the layer of the sprite

pygame.sprite.LayeredDirty.set_timing_treshold
 set_timing_treshold(time_ms) -> None
sets the threshold in milliseconds

pygame.sprite.LayeredDirty.set_timing_threshold
 set_timing_threshold(time_ms) -> None
sets the threshold in milliseconds

pygame.sprite.GroupSingle
 GroupSingle(sprite=None) -> GroupSingle
Group container that holds a single sprite.

pygame.sprite.spritecollide
 spritecollide(sprite, group, dokill, collided = None) -> Sprite_list
Find sprites in a group that intersect another sprite.

pygame.sprite.collide_rect
 collide_rect(left, right) -> bool
Collision detection between two sprites, using rects.

pygame.sprite.collide_rect_ratio
 collide_rect_ratio(ratio) -> collided_callable
Collision detection between two sprites, using rects scaled to a ratio.

pygame.sprite.collide_circle
 collide_circle(left, right) -> bool
Collision detection between two sprites, using circles.

pygame.sprite.collide_circle_ratio
 collide_circle_ratio(ratio) -> collided_callable
Collision detection between two sprites, using circles scaled to a ratio.

pygame.sprite.collide_mask
 collide_mask(sprite1, sprite2) -> (int, int)
 collide_mask(sprite1, sprite2) -> None
Collision detection between two sprites, using masks.

pygame.sprite.groupcollide
 groupcollide(group1, group2, dokill1, dokill2, collided = None) -> Sprite_dict
Find all sprites that collide between two groups.

pygame.sprite.spritecollideany
 spritecollideany(sprite, group, collided = None) -> Sprite
 spritecollideany(sprite, group, collided = None) -> None
Simple test if a sprite intersects anything in a group.

*/