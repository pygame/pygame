import unittest

import pygame
from pygame.locals import *


class BlitTest(unittest.TestCase):
    def test_SRCALPHA(self):
        """SRCALPHA tests."""
        # blend(s, 0, d) = d
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((255, 255, 255, 0))

        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((0, 0, 255, 255))

        s.blit(d, (0, 0))
        self.assertEqual(s.get_at((0, 0)), d.get_at((0, 0)))

        # blend(s, 255, d) = s
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((123, 0, 0, 255))
        s1 = pygame.Surface((1, 1), SRCALPHA, 32)
        s1.fill((123, 0, 0, 255))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((10, 0, 0, 0))
        s.blit(d, (0, 0))
        self.assertEqual(s.get_at((0, 0)), s1.get_at((0, 0)))

        # TODO: these should be true too.
        # blend(0, sA, 0) = 0
        # blend(255, sA, 255) = 255
        # blend(s, sA, d) <= 255

    def test_BLEND(self):
        """BLEND_ tests."""

        # test that it doesn't overflow, and that it is saturated.
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((255, 255, 255, 0))

        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((0, 0, 255, 255))

        s.blit(d, (0, 0), None, BLEND_ADD)

        # print("d %s" % (d.get_at((0,0)),))
        # print(s.get_at((0,0)))
        # self.assertEqual(s.get_at((0,0))[2], 255 )
        # self.assertEqual(s.get_at((0,0))[3], 0 )

        s.blit(d, (0, 0), None, BLEND_RGBA_ADD)
        # print(s.get_at((0,0)))
        self.assertEqual(s.get_at((0, 0))[3], 255)

        # test adding works.
        s.fill((20, 255, 255, 0))
        d.fill((10, 0, 255, 255))
        s.blit(d, (0, 0), None, BLEND_ADD)
        self.assertEqual(s.get_at((0, 0))[2], 255)

        # test subbing works.
        s.fill((20, 255, 255, 0))
        d.fill((10, 0, 255, 255))
        s.blit(d, (0, 0), None, BLEND_SUB)
        self.assertEqual(s.get_at((0, 0))[0], 10)

        # no overflow in sub blend.
        s.fill((20, 255, 255, 0))
        d.fill((30, 0, 255, 255))
        s.blit(d, (0, 0), None, BLEND_SUB)
        self.assertEqual(s.get_at((0, 0))[0], 0)

    def make_blit_list(self, num_surfs):
        """Generate a list of tuples representing surfaces and destinations
        for blitting"""

        blit_list = []
        for i in range(num_surfs):
            dest = (i * 10, 0)
            surf = pygame.Surface((10, 10), SRCALPHA, 32)
            color = (i * 1, i * 1, i * 1)
            surf.fill(color)
            blit_list.append((surf, dest))
        return blit_list
    
    def test_blits(self):
        """Test pygame.Surface.blits by comparing its performance
        and correctness against manual blitting and includes checks
        for correct color placement and timing measurements."""

        NUM_SURFS = 255
        PRINT_TIMING = 0
        dst = pygame.Surface((NUM_SURFS * 10, 10), SRCALPHA, 32)
        dst.fill((230, 230, 230))
        blit_list = self.make_blit_list(NUM_SURFS)

        def manuel_blits(blit_list):
            """Manually iterates over the blit_list and blits each surface
            onto the destination rectangle."""
            for surface, dest in blit_list:
                dst.blit(surface, dest)

        from time import time

        # Measures time performance of custom blits method
        t0 = time()
        results = manuel_blits(blit_list)
        t1 = time()
        if PRINT_TIMING:
            print(f"python blits: {t1 - t0}")

        # Measures time performance of Surface.blits
        dst.fill((230, 230, 230))
        t0 = time()
        results = dst.blits(blit_list)
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits :{t1 - t0}")

        # Check if we blit all the different colors in the correct spots.
        for i in range(NUM_SURFS):
            color = (i * 1, i * 1, i * 1)
            self.assertEqual(dst.get_at((i * 10, 0)), color)
            self.assertEqual(dst.get_at(((i * 10) + 5, 5)), color)

        self.assertEqual(len(results), NUM_SURFS)

        # Measure time performance of Surface.blits with doreturn=0
        t0 = time()
        results = dst.blits(blit_list, doreturn=0)
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits doreturn=0: {t1 - t0}")
        # Tests the doreturn parameter returns None when set to 0
        self.assertEqual(results, None)

        # Measure the time performance of Surface.blits using a generator
        t0 = time()
        results = dst.blits(((surf, dest) for surf, dest in blit_list))
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits generator: {t1 - t0}")

    def test_blits_not_sequence(self):
        """Tests that calling blits() with an invalid non-sequence None argument 
        raises a ValueError."""

        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        with self.assertRaises(ValueError):
            dst.blits(None)

    def test_blits_wrong_length(self):
        """Tests that calling blits() with an invalid sequence containing a single surface 
        (without a destination) raises a ValueError."""

        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        with self.assertRaises(ValueError):
            dst.blits([pygame.Surface((10, 10), SRCALPHA, 32)])

    def test_blits_bad_surf_args(self):
        """Tests that calling blits() with a sequence containing an invalid tuple of 
        None arguments raises a TypeError."""

        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        with self.assertRaises(TypeError):
            dst.blits([(None, None)])

    def test_blits_bad_dest(self):
        """Tests that calling blits() with a sequence containing an invalid tuple with a 
        destination of None raises a TypeError."""
        
        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        with self.assertRaises(TypeError):
            dst.blits([(pygame.Surface((10, 10), SRCALPHA, 32), None)])


if __name__ == "__main__":
    unittest.main()
