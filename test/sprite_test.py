


import unittest
import pygame
from pygame import sprite

class SpriteTest( unittest.TestCase ):
    def testAbstractGroup_has( self ):
        """ See if abstractGroup has works as expected.
        """
        ag = sprite.AbstractGroup()
        ag2 = sprite.AbstractGroup()
        s1 = sprite.Sprite(ag)
        s2 = sprite.Sprite(ag)
        s3 = sprite.Sprite(ag2)
        s4 = sprite.Sprite(ag2)

        self.assertEqual(True, s1 in ag )

        self.assertEqual(True, ag.has(s1) )

        self.assertEqual(True, ag.has([s1, s2]) )

        # see if one of them not being in there.
        self.assertNotEqual(True, ag.has([s1, s2, s3]) )

        # see if a second AbstractGroup works.
        self.assertEqual(True, ag2.has(s3) )



    def test_spritecollide(self):

        ag = sprite.AbstractGroup()
        ag2 = sprite.AbstractGroup()
        s1 = sprite.Sprite(ag)
        s2 = sprite.Sprite(ag2)
        s1.image = pygame.Surface((10,10), pygame.SRCALPHA, 32)
        s2.image = pygame.Surface((10,10), pygame.SRCALPHA, 32)
        
        s1.rect = s1.image.get_rect()
        s2.rect = s2.image.get_rect()
        
        r = sprite.spritecollide(s1, ag2, dokill = False, collided = None)
        self.assertTrue(r)
        
        
        # need to pass a function.
        self.assertRaises(TypeError, sprite.spritecollide, s1, ag2, dokill = False, collided = 1)

        self.assertTrue( sprite.spritecollide( s1, ag2, dokill = False, collided = sprite.collide_rect) )

        # if there are no mask attributes.
        self.assertRaises( AttributeError, sprite.spritecollide, s1, ag2, dokill = False, collided = sprite.collide_mask)
        
        # make some sprites that are fully transparent, so they won't collide.
        s1.image.fill((255,255,255,0))
        s2.image.fill((255,255,255,0))
        
        s1.mask = pygame.mask.from_surface(s1.image, 255)
        s2.mask = pygame.mask.from_surface(s2.image, 255)
        
        self.assertFalse( sprite.spritecollide( s1, ag2, dokill = False, collided = sprite.collide_mask) )
        
        # make some fully opaque sprites that will collide with masks.
        s1.image.fill((255,255,255,255))
        s2.image.fill((255,255,255,255))
        
        s1.mask = pygame.mask.from_surface(s1.image)
        s2.mask = pygame.mask.from_surface(s2.image)
        
        self.assertTrue( sprite.spritecollide( s1, ag2, dokill = False, collided = sprite.collide_mask) )
        
        




import pygame

import unittest
import pygame.sprite as FastRenderGroup
from pygame.sprite import LayeredUpdates as LayeredRenderGroup


class Unit_test_LRG(unittest.TestCase):
    
    
    def setUp(self):
        self.LRG = LayeredRenderGroup()
        
    def test_get_layer_of_sprite(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr, layer=666)
        self.assert_(len(self.LRG._spritelist)==1)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==666)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==self.LRG._spritelayers[spr])
        
        
    def test_add_sprite(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr)
        self.assert_(len(self.LRG._spritelist)==1)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==self.LRG._default_layer)
        
    def test_add_sprite_with_layer_attribute(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        spr._layer = 100
        self.LRG.add(spr)
        self.assert_(len(self.LRG._spritelist)==1)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==100)
        
    def test_add_sprite_passing_layer(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr, layer=100)
        self.assert_(len(self.LRG._spritelist)==1)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==100)
        
    def test_add_sprite_overriding_layer_attr(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        spr._layer = 100
        self.LRG.add(spr, layer=200)
        self.assert_(len(self.LRG._spritelist)==1)
        self.assert_(self.LRG.get_layer_of_sprite(spr)==200)
        
    def test_add_sprite_init(self):
        spr = pygame.sprite.Sprite()
        lrg2 = LayeredRenderGroup(spr)
        self.assert_(len(lrg2._spritelist)==1)
        self.assert_(lrg2._spritelayers[spr]==lrg2._default_layer)
        
    def test_add_sprite_init_layer_attr(self):
        spr = pygame.sprite.Sprite()
        spr._layer = 20
        lrg2 = LayeredRenderGroup(spr)
        self.assert_(len(lrg2._spritelist)==1)
        self.assert_(lrg2._spritelayers[spr]==20)
        
    def test_add_sprite_init_passing_layer(self):
        spr = pygame.sprite.Sprite()
        lrg2 = LayeredRenderGroup(spr, layer=33)
        self.assert_(len(lrg2._spritelist)==1)
        self.assert_(lrg2._spritelayers[spr]==33)
        
    def test_add_sprite_init_overiding_layer(self):
        spr = pygame.sprite.Sprite()
        spr._layer = 55
        lrg2 = LayeredRenderGroup(spr, layer=33)
        self.assert_(len(lrg2._spritelist)==1)
        self.assert_(lrg2._spritelayers[spr]==33)
        
    def test_add_spritelist(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
        self.LRG.add(sprites)
        self.assert_(len(self.LRG._spritelist)==10)
        for i in range(10):
            self.assert_(self.LRG.get_layer_of_sprite(sprites[i])==self.LRG._default_layer)
        
    def test_add_spritelist_with_layer_attr(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
            sprites[-1]._layer = i
        self.LRG.add(sprites)
        self.assert_(len(self.LRG._spritelist)==10)
        for i in range(10):
            self.assert_(self.LRG.get_layer_of_sprite(sprites[i])==i)
        
    def test_add_spritelist_passing_layer(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
        self.LRG.add(sprites, layer=33)
        self.assert_(len(self.LRG._spritelist)==10)
        for i in range(10):
            self.assert_(self.LRG.get_layer_of_sprite(sprites[i])==33)
        
    def test_add_spritelist_overriding_layer(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
            sprites[-1].layer = i
        self.LRG.add(sprites, layer=33)
        self.assert_(len(self.LRG._spritelist)==10)
        for i in range(10):
            self.assert_(self.LRG.get_layer_of_sprite(sprites[i])==33)
            
    def test_add_spritelist_init(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
        lrg2 = LayeredRenderGroup(sprites)
        self.assert_(len(lrg2._spritelist)==10)
        for i in range(10):
            self.assert_(lrg2.get_layer_of_sprite(sprites[i])==self.LRG._default_layer)
        
    def test_remove_sprite(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
            sprites[-1].rect = 0
        self.LRG.add(sprites)
        self.assert_(len(self.LRG._spritelist)==10)
        for i in range(10):
            self.LRG.remove(sprites[i])
        self.assert_(len(self.LRG._spritelist)==0)
        
    def test_sprites(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            sprites.append(pygame.sprite.Sprite())
            sprites[-1]._layer = 10-i
        self.LRG.add(sprites)
        self.assert_(len(self.LRG._spritelist)==10)
        for idx,spr in enumerate(self.LRG.sprites()):
            self.assert_(spr == sprites[9-idx])
        
    def test_layers(self):
        self.assert_(len(self.LRG._spritelist)==0)
        sprites = []
        for i in range(10):
            for j in range(5):
                sprites.append(pygame.sprite.Sprite())
                sprites[-1]._layer = i
        self.LRG.add(sprites)
        lays = self.LRG.layers()
        for i in range(10):
            self.assert_(lays[i] == i)
            
    def test_layers2(self):
        self.assert_(len(self.LRG)==0)
        layers = [1,4,6,8,3,6,2,6,4,5,6,1,0,9,7,6,54,8,2,43,6,1]
        for lay in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=lay)
        layers.sort()
        for idx, spr in enumerate(self.LRG.sprites()):
            self.assert_(self.LRG.get_layer_of_sprite(spr)==layers[idx])
            
    def test_change_layer(self):
        self.assert_(len(self.LRG._spritelist)==0)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr, layer=99)
        self.assert_(self.LRG._spritelayers[spr] == 99)
        self.LRG.change_layer(spr, 44)
        self.assert_(self.LRG._spritelayers[spr] == 44)
        
        spr2 = pygame.sprite.Sprite()
        spr2.layer = 55
        self.LRG.add(spr2)
        self.LRG.change_layer(spr2, 77)
        self.assert_(spr2.layer == 77)
        
    def test_get_top_layer(self):
        layers = [1,5,2,8,4,5,3,88,23,0]
        for i in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=i)
        self.assert_(self.LRG.get_top_layer()==max(layers))
        self.assert_(self.LRG.get_top_layer()==max(self.LRG._spritelayers.values()))
        self.assert_(self.LRG.get_top_layer()==self.LRG._spritelayers[self.LRG._spritelist[-1]])
            
    def test_get_bottom_layer(self):
        layers = [1,5,2,8,4,5,3,88,23,0]
        for i in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=i)
        self.assert_(self.LRG.get_bottom_layer()==min(layers))
        self.assert_(self.LRG.get_bottom_layer()==min(self.LRG._spritelayers.values()))
        self.assert_(self.LRG.get_bottom_layer()==self.LRG._spritelayers[self.LRG._spritelist[0]])
            
    def test_move_to_front(self):
        layers = [1,5,2,8,4,5,3,88,23,0]
        for i in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=i)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr, layer=3)
        self.assert_(spr != self.LRG._spritelist[-1]) 
        self.LRG.move_to_front(spr)
        self.assert_(spr == self.LRG._spritelist[-1]) 
        
    def test_move_to_back(self):
        layers = [1,5,2,8,4,5,3,88,23,0]
        for i in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=i)
        spr = pygame.sprite.Sprite()
        self.LRG.add(spr, layer=55)
        self.assert_(spr != self.LRG._spritelist[0]) 
        self.LRG.move_to_back(spr)
        self.assert_(spr == self.LRG._spritelist[0]) 
        
    def test_get_top_sprite(self):
        layers = [1,5,2,8,4,5,3,88,23,0]
        for i in layers:
            self.LRG.add(pygame.sprite.Sprite(), layer=i)
        self.assert_(self.LRG.get_layer_of_sprite(self.LRG.get_top_sprite())== self.LRG.get_top_layer())
        
    def test_get_sprites_from_layer(self):
        self.assert_(len(self.LRG)==0)
        sprites = {}
        layers = [1,4,5,6,3,7,8,2,1,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,0,1,6,5,4,3,2]
        for lay in layers:
            spr = pygame.sprite.Sprite()
            spr._layer = lay
            self.LRG.add(spr)
            if not sprites.has_key(lay):
                sprites[lay] = []
            sprites[lay].append(spr)
            
        for lay in self.LRG.layers():
            for spr in self.LRG.get_sprites_from_layer(lay):
                self.assert_(spr in sprites[lay])
                sprites[lay].remove(spr)
                if len(sprites[lay]) == 0:
                    del sprites[lay]
        self.assert_(len(sprites.values())==0)
        
    def test_switch_layer(self):
        self.assert_(len(self.LRG)==0)
        sprites1 = []
        sprites2 = []
        layers = [3,2,3,2,3,3,2,2,3,2,3,2,3,2,3,2,3,3,2,2,3,2,3]
        for lay in layers:
            spr = pygame.sprite.Sprite()
            spr._layer = lay
            self.LRG.add(spr)
            if lay==2:
                sprites1.append(spr)
            else:
                sprites2.append(spr)
                
        for spr in sprites1:
            self.assert_(spr in self.LRG.get_sprites_from_layer(2))
        for spr in sprites2:
            self.assert_(spr in self.LRG.get_sprites_from_layer(3))
        self.assert_(len(self.LRG)==len(sprites1)+len(sprites2))
        
        self.LRG.switch_layer(2,3)
        
        for spr in sprites1:
            self.assert_(spr in self.LRG.get_sprites_from_layer(3))
        for spr in sprites2:
            self.assert_(spr in self.LRG.get_sprites_from_layer(2))
        self.assert_(len(self.LRG)==len(sprites1)+len(sprites2))
        
#TODO: test FRG and DirtySprite (visible, layer, blendmode and dirty)

if __name__ == "__main__":
    unittest.main()
        
    
    
    




if __name__ == '__main__':
    unittest.main()
