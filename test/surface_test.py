
import unittest
import pygame

from pygame.locals import *

class SurfaceTest( unittest.TestCase ):
    
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
        #s.fill((255, 0, 0))

        #self.assertEqual(s.get_bitsize(), 24)
        #self.assertEqual(s.get_bytesize(), 4)


        # if we have a 32 bit surface, the SRCALPHA should have worked too.
        surf2 = pygame.Surface((70,70), SRCALPHA)
        if surf2.get_bitsize() == 32:
            self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)






if __name__ == '__main__':
    unittest.main()
