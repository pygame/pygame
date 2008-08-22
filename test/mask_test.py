import test_utils
import test.unittest as unittest

from test_utils import test_not_implemented

import pygame
import pygame.mask

from pygame.locals import *

def maskFromSurface(surface, threshold = 127):
    mask = pygame.Mask(surface.get_size())
    key = surface.get_colorkey()
    if key:
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                if surface.get_at((x+0.1,y+0.1)) != key:
                    mask.set_at((x,y),1)
    else:
        for y in range(surface.get_height()):
            for x in range (surface.get_width()):
                if surface.get_at((x,y))[3] > threshold:
                    mask.set_at((x,y),1)
    return mask

#pygame.init()
#pygame.display.set_mode((10,10))

class MaskTypeTest( unittest.TestCase ):
    def todo_test_get_at(self):

        # __doc__ (as of 2008-08-02) for pygame.mask.Mask.get_at:

          # Mask.get_at((x,y)) -> int
          # Returns nonzero if the bit at (x,y) is set.
          # 
          # Coordinates start at (0,0) is top left - just like Surfaces. 

        self.fail() 

    def todo_test_get_size(self):

        # __doc__ (as of 2008-08-02) for pygame.mask.Mask.get_size:

          # Mask.get_size() -> width,height
          # Returns the size of the mask.

        self.fail() 

    def todo_test_overlap(self):

        # __doc__ (as of 2008-08-02) for pygame.mask.Mask.overlap:

          # Mask.overlap(othermask, offset) -> x,y
          # Returns the point of intersection if the masks overlap with the
          # given offset - or None if it does not overlap.

          # The overlap tests uses the following offsets (which may be negative): 
          #    +----+----------..
          #    |A   | yoffset
          #    |  +-+----------..
          #    +--|B
          #    |xoffset
          #    |  |
          #    :  :

        self.fail() 

    def todo_test_overlap_area(self):

        # __doc__ (as of 2008-08-02) for pygame.mask.Mask.overlap_area:

          # Mask.overlap_area(othermask, offset) -> numpixels
          # Returns the number of overlapping 'pixels'.
          # 
          # You can see how many pixels overlap with the other mask given.  This
          # can be used to see in which direction things collide, or to see how
          # much the two masks collide.

        self.fail() 

    def todo_test_set_at(self):

        # __doc__ (as of 2008-08-02) for pygame.mask.Mask.set_at:

          # Mask.set_at((x,y),value)
          # Sets the position in the mask given by x and y.

        self.fail() 
    
    def test_mask_access( self ):
        """ do the set_at, and get_at parts work correctly?
        """
        m = pygame.Mask((10,10))
        m.set_at((0,0), 1)
        self.assertEqual(m.get_at((0,0)), 1)
        m.set_at((9,0), 1)
        self.assertEqual(m.get_at((9,0)), 1)

        #s = pygame.Surface((10,10))
        #s.set_at((1,0), (0, 0, 1, 255))
        #self.assertEqual(s.get_at((1,0)), (0, 0, 1, 255))
        #s.set_at((-1,0), (0, 0, 1, 255))

        # out of bounds, should get IndexError
        self.assertRaises(IndexError, lambda : m.get_at((-1,0)) )
        self.assertRaises(IndexError, lambda : m.set_at((-1,0), 1) )
        self.assertRaises(IndexError, lambda : m.set_at((10,0), 1) )
        self.assertRaises(IndexError, lambda : m.set_at((0,10), 1) )
        
    def test_drawing(self):
        """ Test fill, clear, invert, draw, erase
        """
        
        m = pygame.Mask((100,100))
        self.assertEqual(m.count(), 0)
        
        m.fill()
        self.assertEqual(m.count(), 10000)
        
        m2 = pygame.Mask((10,10))
        m2.fill()
        m.erase(m2, (50,50))
        self.assertEqual(m.count(), 9900)
        
        m.invert()
        self.assertEqual(m.count(), 100)
        
        m.draw(m2, (0,0))
        self.assertEqual(m.count(), 200)    
        
        m.clear()
        self.assertEqual(m.count(), 0)
        
    def test_outline(self):
        """
        """
    
        m = pygame.Mask((20,20))
        self.assertEqual(m.outline(), [])
        
        m.set_at((10,10), 1)
        self.assertEqual(m.outline(), [(10,10)])
        
        m.set_at((10,12), 1)
        self.assertEqual(m.outline(10), [(10,10)])
        
        m.set_at((11,11), 1)
        self.assertEqual(m.outline(), [(10,10), (11,11), (10,12), (11,11), (10,10)])
        self.assertEqual(m.outline(2), [(10,10), (10,12), (10,10)])
        
        #TODO: Test more corner case outlines.
        
    def test_connected_components(self):
        """
        """
        
        m = pygame.Mask((10,10))
        self.assertEquals(repr(m.connected_components()), "[]")
        
        comp = m.connected_component()
        self.assertEquals(m.count(), comp.count())
        
        m.set_at((0,0), 1)
        m.set_at((1,1), 1)
        comp = m.connected_component()
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEquals(comp.count(), comps[0].count())
        self.assertEquals(comps1[0].count(), 2)
        self.assertEquals(comps2[0].count(), 2)
        self.assertEquals(repr(comps3), "[]")
        
        m.set_at((9, 9), 1)
        comp = m.connected_component()
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEquals(comp.count(), 2)
        self.assertEquals(len(comps), 2)
        self.assertEquals(len(comps1), 2)
        self.assertEquals(len(comps2), 1)
        self.assertEquals(len(comps3), 0)
        

    def test_get_bounding_rects(self):
        """
        """

        m = pygame.Mask((10,10))
        m.set_at((0,0), 1)
        m.set_at((1,0), 1)

        m.set_at((0,1), 1)

        m.set_at((0,3), 1)
        m.set_at((3,3), 1)
        
        r = m.get_bounding_rects()
        
        self.assertEquals(repr(r), "[<rect(0, 0, 2, 2)>, <rect(0, 3, 1, 1)>, <rect(3, 3, 1, 1)>]")
        
        
        


        #1100
        #1111
        m = pygame.Mask((4,2))
        m.set_at((0,0), 1)
        m.set_at((1,0), 1)
        m.set_at((2,0), 0)
        m.set_at((3,0), 0)

        m.set_at((0,1), 1)
        m.set_at((1,1), 1)
        m.set_at((2,1), 1)
        m.set_at((3,1), 1)
 
        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[<rect(0, 0, 4, 2)>]")

        
        #00100
        #01110
        #00100
        m = pygame.Mask((5,3))
        m.set_at((0,0), 0)
        m.set_at((1,0), 0)
        m.set_at((2,0), 1)
        m.set_at((3,0), 0)
        m.set_at((4,0), 0)

        m.set_at((0,1), 0)
        m.set_at((1,1), 1)
        m.set_at((2,1), 1)
        m.set_at((3,1), 1)
        m.set_at((4,1), 0)

        m.set_at((0,2), 0)
        m.set_at((1,2), 0)
        m.set_at((2,2), 1)
        m.set_at((3,2), 0)
        m.set_at((4,2), 0)

        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[<rect(1, 0, 3, 3)>]")



        #00010
        #00100
        #01000
        m = pygame.Mask((5,3))
        m.set_at((0,0), 0)
        m.set_at((1,0), 0)
        m.set_at((2,0), 0)
        m.set_at((3,0), 1)
        m.set_at((4,0), 0)

        m.set_at((0,1), 0)
        m.set_at((1,1), 0)
        m.set_at((2,1), 1)
        m.set_at((3,1), 0)
        m.set_at((4,1), 0)

        m.set_at((0,2), 0)
        m.set_at((1,2), 1)
        m.set_at((2,2), 0)
        m.set_at((3,2), 0)
        m.set_at((4,2), 0)

        r = m.get_bounding_rects()
        self.assertEquals(repr(r), "[<rect(1, 0, 3, 3)>]")




        #00011
        #11111
        m = pygame.Mask((5,2))
        m.set_at((0,0), 0)
        m.set_at((1,0), 0)
        m.set_at((2,0), 0)
        m.set_at((3,0), 1)
        m.set_at((4,0), 1)

        m.set_at((0,1), 1)
        m.set_at((1,1), 1)
        m.set_at((2,1), 1)
        m.set_at((3,1), 1)
        m.set_at((3,1), 1)
 
        r = m.get_bounding_rects()
        #TODO: this should really make one bounding rect.
        #self.assertEquals(repr(r), "[<rect(0, 0, 5, 2)>]")

class MaskModuleTest(unittest.TestCase):
    def test_from_surface(self):
        """  Does the mask.from_surface() work correctly?
        """

        mask_from_surface = pygame.mask.from_surface

        surf = pygame.Surface((70,70), SRCALPHA, 32)

        surf.fill((255,255,255,255))

        amask = pygame.mask.from_surface(surf)
        #amask = mask_from_surface(surf)

        self.assertEqual(amask.get_at((0,0)), 1)
        self.assertEqual(amask.get_at((66,1)), 1)
        self.assertEqual(amask.get_at((69,1)), 1)

        surf.set_at((0,0), (255,255,255,127))
        surf.set_at((1,0), (255,255,255,128))
        surf.set_at((2,0), (255,255,255,0))
        surf.set_at((3,0), (255,255,255,255))

        amask = mask_from_surface(surf)
        self.assertEqual(amask.get_at((0,0)), 0)
        self.assertEqual(amask.get_at((1,0)), 1)
        self.assertEqual(amask.get_at((2,0)), 0)
        self.assertEqual(amask.get_at((3,0)), 1)

        surf.fill((255,255,255,0))
        amask = mask_from_surface(surf)
        self.assertEqual(amask.get_at((0,0)), 0)

        #TODO: test a color key surface.
        
    def test_from_threshold(self):
        """ Does mask.from_threshold() work correctly?
        """
        
        a = [16, 24, 32]
        
        for i in a:
            surf = pygame.surface.Surface((70,70), 0, i)
            surf.fill((100,50,200),(20,20,20,20))
            mask = pygame.mask.from_threshold(surf,(100,50,200,255),(10,10,10,255))
            
            self.assertEqual(mask.count(), 400)
            self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((20,20,20,20))])
            
        for i in a:
            surf = pygame.surface.Surface((70,70), 0, i)
            surf2 = pygame.surface.Surface((70,70), 0, i)
            surf.fill((100,100,100))
            surf2.fill((150,150,150))
            surf2.fill((100,100,100), (40,40,10,10))
            mask = pygame.mask.from_threshold(surf, (0,0,0,0), (10,10,10,255), surf2)
            
            self.assertEqual(mask.count(), 100)
            self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((40,40,10,10))])            

if __name__ == '__main__':

    if 1:
        unittest.main()
    else:
        mask_from_surface = maskFromSurface

        surf = pygame.Surface((70,70), SRCALPHA, 32)
        #surf = surf.convert_alpha()
        surf.set_at((0,0), (255,255,255,0))
        print surf.get_at((0,0))

        print "asdf"
        print surf
