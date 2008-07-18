import test_utils
import test.unittest as unittest
import pygame

from test_utils import test_not_implemented

from pygame.locals import *

class SurfaceTypeTest(unittest.TestCase):
    def test_set_clip( self ):
        """ see if surface.set_clip(None) works correctly.
        """
        s = pygame.Surface((800, 600))
        r = pygame.Rect(10, 10, 10, 10)
        s.set_clip(r)
        r.move_ip(10, 0)
        s.set_clip(None)
        res = s.get_clip()
        # this was garbled before.
        self.assertEqual(res[0], 0)
        self.assertEqual(res[2], 800)

    def test_print(self):
        surf = pygame.Surface((70,70), 0, 32)
        self.assertEqual(repr(surf), '<Surface(70x70x32 SW)>')

    def test_keyword_arguments(self):
        surf = pygame.Surface((70,70), flags=SRCALPHA, depth=32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)
        self.assertEqual(surf.get_bitsize(), 32)
        
        # sanity check to make sure the check below is valid
        surf_16 = pygame.Surface((70,70), 0, 16)
        self.assertEqual(surf_16.get_bytesize(), 2)
        
        # try again with an argument list
        surf_16 = pygame.Surface((70,70), depth=16)
        self.assertEqual(surf_16.get_bytesize(), 2)

    def test_set_at(self):

        #24bit surfaces 
        s = pygame.Surface( (100, 100), 0, 24)
        s.fill((0,0,0))

        # set it with a tuple.
        s.set_at((0,0), (10,10,10, 255))
        r = s.get_at((0,0))
        self.assertEqual(r, (10,10,10, 255))

        # try setting a color with a single integer.
        s.fill((0,0,0,255))
        s.set_at ((10, 1), 0x0000FF)
        r = s.get_at((10,1))
        self.assertEqual(r, (0,0,255, 255))


    def test_SRCALPHA(self):
        # has the flag been passed in ok?
        surf = pygame.Surface((70,70), SRCALPHA, 32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)

        #24bit surfaces can not have SRCALPHA.
        self.assertRaises(ValueError, pygame.Surface, (100, 100), pygame.SRCALPHA, 24)

        # if we have a 32 bit surface, the SRCALPHA should have worked too.
        surf2 = pygame.Surface((70,70), SRCALPHA)
        if surf2.get_bitsize() == 32:
            self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)

    def test_get_buffer (self):
        surf = pygame.Surface ((70, 70), 0, 32)
        buf = surf.get_buffer ()
        # 70*70*4 bytes = 19600
        self.assertEqual (repr (buf), "<BufferProxy(19600)>")

    def test_get_bounding_rect (self):
        surf = pygame.Surface ((70, 70), SRCALPHA, 32)
        surf.fill((0,0,0,0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30,30),(255,255,255,1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((29,29),(255,255,255,1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 29)
        self.assertEqual(bound_rect.top, 29)
        self.assertEqual(bound_rect.width, 2)
        self.assertEqual(bound_rect.height, 2)
        
        surf = pygame.Surface ((70, 70), 0, 24)
        surf.fill((0,0,0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, surf.get_width())
        self.assertEqual(bound_rect.height, surf.get_height())

        surf.set_colorkey((0,0,0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30,30),(255,255,255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((60,60),(255,255,255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 31)
        self.assertEqual(bound_rect.height, 31)


    def test_blit(self):        # See blit_test.py

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.blit:

          # Surface.blit(source, dest, area=None, special_flags = 0): return Rect
          # draw one image onto another

        self.assert_(test_not_implemented()) 

    def test_convert(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.convert:

          # Surface.convert(Surface): return Surface
          # Surface.convert(depth, flags=0): return Surface
          # Surface.convert(masks, flags=0): return Surface
          # Surface.convert(): return Surface
          # change the pixel format of an image

        self.assert_(test_not_implemented())

    def test_convert_alpha(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.convert_alpha:

          # Surface.convert_alpha(Surface): return Surface
          # Surface.convert_alpha(): return Surface
          # change the pixel format of an image including per pixel alphas

        self.assert_(test_not_implemented()) 

    def test_copy(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.copy:

          # Surface.copy(): return Surface
          # create a new copy of a Surface
        
        color = (25, 25, 25, 25)
        s1 = pygame.Surface((32,32), pygame.SRCALPHA, 32)
        s1.fill(color)
        
        s2 = s1.copy()
        
        s1rect = s1.get_rect()
        s2rect = s2.get_rect()
        
        self.assert_(s1rect.size == s2rect.size)
        self.assert_(s2.get_at((10,10)) == color)

    def test_fill(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.fill:

          # Surface.fill(color, rect=None, special_flags=0): return Rect
          # fill Surface with a solid color
        
        color = (25, 25, 25, 25)
        fill_rect = pygame.Rect(0, 0, 16, 16)
        
        s1 = pygame.Surface((32,32), pygame.SRCALPHA, 32)
        s1.fill(color, fill_rect)   
        
        for pt in test_utils.rect_area_pts(fill_rect):
            self.assert_(s1.get_at(pt) == color )
        
        for pt in test_utils.rect_outer_bounds(fill_rect):
            self.assert_(s1.get_at(pt) != color )

        self.assert_(test_not_implemented())

    def test_get_abs_offset(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_abs_offset:

          # Surface.get_abs_offset(): return (x, y)
          # find the absolute position of a child subsurface inside its top level parent

        self.assert_(test_not_implemented()) 

    def test_get_abs_parent(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_abs_parent:

          # Surface.get_abs_parent(): return Surface
          # find the top level parent of a subsurface

        self.assert_(test_not_implemented()) 

    ########################################################################

    def test_get_alpha(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_alpha:

          # Surface.get_alpha(): return int_value or None
          # get the current Surface transparency value

        s1 = pygame.Surface((32,32), pygame.SRCALPHA, 32)
        self.assert_(s1.get_alpha() == 255)
        
        for alpha in (0, 32, 127, 255):
            s1.set_alpha(alpha)
            for t in range(4): s1.set_alpha(s1.get_alpha())
            self.assert_(s1.get_alpha() == alpha)
                
    def test_set_alpha(self):
        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.set_alpha:
    
          # Surface.set_alpha(value, flags=0): return None
          # Surface.set_alpha(None): return None
          # set the alpha value for the full Surface image
    
        self.assert_(test_not_implemented()) 
    
    ########################################################################
    
    def test_get_at(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_at:

          # Surface.get_at((x, y)): return Color
          # get the color value at a single pixel

        self.assert_(test_not_implemented()) 

    ########################################################################

    def test_get_bitsize(self):
        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_bitsize:

          # Surface.get_bitsize(): return int
          # get the bit depth of the Surface pixel format

        self.assert_(test_not_implemented()) 

    def test_get_bytesize(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_bytesize:

          # Surface.get_bytesize(): return int
          # get the bytes used per Surface pixel
        
        s1 = pygame.Surface((32,32), pygame.SRCALPHA, 32)
        self.assert_(s1.get_bytesize() == 4)
        self.assert_(s1.get_bitsize() == 32)

    ########################################################################

    def test_get_clip(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_clip:

          # Surface.get_clip(): return Rect
          # get the current clipping area of the Surface

        self.assert_(test_not_implemented()) 

    def test_get_flags(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_flags:

          # Surface.get_flags(): return int
          # get the additional flags used for the Surface
        
        s1 = pygame.Surface((32,32), pygame.SRCALPHA, 32)
        self.assert_(s1.get_flags() == pygame.SRCALPHA)

    def test_get_locked(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_locked:

          # Surface.get_locked(): return bool
          # test if the Surface is current locked

        self.assert_(test_not_implemented())

    def test_get_locks(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_locks:

          # Surface.get_locks(): return tuple
          # Gets the locks for the Surface

        self.assert_(test_not_implemented()) 

    def test_get_losses(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_losses:

          # Surface.get_losses(): return (R, G, B, A)
          # the significant bits used to convert between a color and a mapped integer

        self.assert_(test_not_implemented()) 

    def test_get_masks(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_masks:

          # Surface.get_masks(): return (R, G, B, A)
          # the bitmasks needed to convert between a color and a mapped integer

        self.assert_(test_not_implemented()) 

    def test_get_offset(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_offset:

          # Surface.get_offset(): return (x, y)
          # find the position of a child subsurface inside a parent

        self.assert_(test_not_implemented()) 
    
    ########################################################################
    
    def test_subsurface(self):
        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.subsurface:
    
          # Surface.subsurface(Rect): return Surface
          # create a new surface that references its parent
    
        self.assert_(test_not_implemented()) 
    
    def test_get_parent(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_parent:

          # Surface.get_parent(): return Surface
          # find the parent of a subsurface

        parent = pygame.Surface((16, 16))
        child = parent.subsurface((0,0,5,5))

        self.assert_(child.get_parent() is parent) 
        
    ########################################################################

    def test_get_pitch(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_pitch:

          # Surface.get_pitch(): return int
          # get the number of bytes used per Surface row

        self.assert_(test_not_implemented()) 

    def test_get_rect(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_rect:

          # Surface.get_rect(**kwargs): return Rect
          # get the rectangular area of the Surface
        
        surf = pygame.Surface((16, 16))
        
        rect = surf.get_rect()
        
        self.assert_(rect.size == (16, 16))

    def test_get_shifts(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_shifts:

          # Surface.get_shifts(): return (R, G, B, A)
          # the bit shifts needed to convert between a color and a mapped integer

        self.assert_(test_not_implemented()) 

    ########################################################################

    def test_get_size(self):
        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_size:

          # Surface.get_size(): return (width, height)
          # get the dimensions of the Surface

        self.assert_(test_not_implemented()) 

    def test_get_width__size_and_height(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_width:

          # Surface.get_width(): return width
          # get the width of the Surface
        
        for w in xrange(0, 255, 32):
            for h in xrange(0, 127, 15):
                s = pygame.Surface((w, h))
                self.assertEquals(s.get_width(), w) 
                self.assertEquals(s.get_height(), h) 
                self.assertEquals(s.get_size(), (w, h))

    ########################################################################

    def test_lock(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.lock:

          # Surface.lock(): return None
          # lock the Surface memory for pixel access

        self.assert_(test_not_implemented())

    def test_map_rgb(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.map_rgb:

          # Surface.map_rgb(Color): return mapped_int
          # convert a color into a mapped color value

        self.assert_(test_not_implemented()) 

    def test_mustlock(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.mustlock:

          # Surface.mustlock(): return bool
          # test if the Surface requires locking

        self.assert_(test_not_implemented()) 

    ########################################################################
    
    def test_get_colorkey(self):
        
        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_colorkey:
    
          # Surface.get_colorkey(): return RGB or None
          # Get the current transparent colorkey
    
        self.assert_(test_not_implemented()) 
    
    def test_set_colorkey(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.set_colorkey:

          # Surface.set_colorkey(Color, flags=0): return None
          # Surface.set_colorkey(None): return None
          # Set the transparent colorkey
        
        s = pygame.Surface((16,16), pygame.SRCALPHA, 32)
        
        colorkeys = ((20,189,20, 255),(128,50,50,255), (23, 21, 255,255))

        for colorkey in colorkeys:
            s.set_colorkey(colorkey)
            for t in range(4): s.set_colorkey(s.get_colorkey())
            self.assertEquals(s.get_colorkey(), colorkey) 

    ########################################################################

    def test_set_palette(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.set_palette:

          # Surface.set_palette([RGB, RGB, RGB, ...]): return None
          # set the color palette for an 8bit Surface

        self.assert_(test_not_implemented())

    def test_get_palette(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_palette:

          # Surface.get_palette(): return [RGB, RGB, RGB, ...]
          # get the color index palette for an 8bit Surface

        self.assert_(test_not_implemented()) 

    ########################################################################

    def test_set_palette_at(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.set_palette_at:

          # Surface.set_at(index, RGB): return None
          # set the color for a single index in an 8bit Surface palette

        self.assert_(test_not_implemented()) 

    def test_get_palette_at(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.get_palette_at:

          # Surface.get_palette_at(index): return RGB
          # get the color for a single entry in a palette

        self.assert_(test_not_implemented()) 

    ########################################################################

    def test_unlock(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.unlock:

          # Surface.unlock(): return None
          # unlock the Surface memory from pixel access

        self.assert_(test_not_implemented()) 

    def test_unmap_rgb(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.unmap_rgb:

          # Surface.map_rgb(mapped_int): return Color
          # convert a mapped integer color value into a Color

        self.assert_(test_not_implemented()) 



    def test_set_masks(self):
        s = pygame.Surface((32,32))
        r,g,b,a = s.get_masks()
        s.set_masks((b,g,r,a))
        r2,g2,b2,a2 = s.get_masks()
        self.assertEqual((r,g,b,a), (b2,g2,r2,a2))


    def test_set_shifts(self):
        s = pygame.Surface((32,32))
        r,g,b,a = s.get_shifts()
        s.set_shifts((b,g,r,a))
        r2,g2,b2,a2 = s.get_shifts()
        self.assertEqual((r,g,b,a), (b2,g2,r2,a2))






if __name__ == '__main__':
    unittest.main()
