
import unittest
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


class MaskTest( unittest.TestCase ):
    
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



    def test_mask_from_surface(self):
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

