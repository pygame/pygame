if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest

import pygame
from pygame.locals import *

class BlitTest( unittest.TestCase ):
    def test_SRCALPHA( self ):
        """ SRCALPHA tests.
        """
        #blend(s, 0, d) = d
        s = pygame.Surface((1,1), SRCALPHA, 32)
        s.fill((255, 255,255, 0))

        d = pygame.Surface((1,1), SRCALPHA, 32)
        d.fill((0, 0,255, 255))

        s.blit(d, (0,0))
        self.assertEqual(s.get_at((0,0)), d.get_at((0,0)) )

        #blend(s, 255, d) = s
        s = pygame.Surface((1,1), SRCALPHA, 32)
        s.fill((123, 0, 0, 255))
        s1 = pygame.Surface((1,1), SRCALPHA, 32)
        s1.fill((123, 0, 0, 255))
        d = pygame.Surface((1,1), SRCALPHA, 32)
        d.fill((10, 0,0, 0))
        s.blit(d, (0,0))
        self.assertEqual(s.get_at((0,0)), s1.get_at((0,0)) )

        #TODO: these should be true too.
        #blend(0, sA, 0) = 0
        #blend(255, sA, 255) = 255
        #blend(s, sA, d) <= 255

    def test_BLEND( self ):
        """ BLEND_ tests.
        """

        #test that it doesn't overflow, and that it is saturated.
        s = pygame.Surface((1,1), SRCALPHA, 32)
        s.fill((255, 255,255, 0))

        d = pygame.Surface((1,1), SRCALPHA, 32)
        d.fill((0, 0,255, 255))

        s.blit(d, (0,0), None, BLEND_ADD)

        #print "d %s" % (d.get_at((0,0)),)
        #print s.get_at((0,0))
        #self.assertEqual(s.get_at((0,0))[2], 255 )
        #self.assertEqual(s.get_at((0,0))[3], 0 )



        s.blit(d, (0,0), None, BLEND_RGBA_ADD)
        #print s.get_at((0,0))
        self.assertEqual(s.get_at((0,0))[3], 255 )


        # test adding works.
        s.fill((20, 255,255, 0))
        d.fill((10, 0,255, 255))
        s.blit(d, (0,0), None, BLEND_ADD)
        self.assertEqual(s.get_at((0,0))[2], 255 )

        # test subbing works.
        s.fill((20, 255,255, 0))
        d.fill((10, 0,255, 255))
        s.blit(d, (0,0), None, BLEND_SUB)
        self.assertEqual(s.get_at((0,0))[0], 10 )

        # no overflow in sub blend.
        s.fill((20, 255,255, 0))
        d.fill((30, 0,255, 255))
        s.blit(d, (0,0), None, BLEND_SUB)
        self.assertEqual(s.get_at((0,0))[0], 0 )



    def test_blits(self):

        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        dst.fill((230, 230, 230))

        blit_list = []
        for i in range(10):
            dest = (i * 10, 0)
            surf = pygame.Surface((10, 10), SRCALPHA, 32)
            color = (i * 10, i * 10, i * 10)
            surf.fill(color)
            blit_list.append((surf, dest))

        def blits(blit_list):
            for surface, dest in blit_list:
                dst.blit(surface, dest)

        #dst.blits(blit_list)
        blits(blit_list)

        # check if we blit all the different colors in the correct spots.
        for i in range(10):
            color = (i * 10, i * 10, i * 10)
            self.assertEqual(dst.get_at((i * 10, 0)), color)
            self.assertEqual(dst.get_at(((i * 10) + 5, 5)), color)


if __name__ == '__main__':
    unittest.main()
