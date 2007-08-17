
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


    def test_SRCALPHA(self):

        # has the flag been passed in ok?
        surf = pygame.Surface((70,70), SRCALPHA, 32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)

        # if we have a 32 bit surface, the SRCALPHA should have worked too.
        surf2 = pygame.Surface((70,70), SRCALPHA)
        if surf2.get_bitsize() == 32:
            self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)


if __name__ == '__main__':
    unittest.main()
