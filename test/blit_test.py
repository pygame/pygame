import unittest
from time import time

import pygame
from pygame.locals import *


class BlitTest(unittest.TestCase):
    def test_SRCALPHA_1(self):
        """Uses the SRCALPHA flag to request a per pixel alpha.
        blend(s, 0, d) = d -> Any surface s with opacity 0 with
        a blit of any surface d will take the values of d."""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((255, 255, 255, 0))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((0, 0, 255, 255))

        s.blit(d, (0, 0))

        self.assertTrue(s.get_at((0, 0)), d.get_at((0, 0)))

    def test_SRCALPHA_2(self):
        """blend(s, 255, d) = s - > Any surface s with opacity
        255 with a blit of any surface d will take the value s."""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((123, 0, 0, 255))
        s1 = pygame.Surface((1, 1), SRCALPHA, 32)
        s1.fill((123, 0, 0, 255))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((10, 0, 0, 0))

        s.blit(d, (0, 0))

        self.assertEqual(s.get_at((0, 0)), s1.get_at((0, 0)))

    def test_SRCALPHA_3(self):
        """blend(0, sA, 0) = 0 -> A surface s with RGB (0, 0, 0)
        at any opacity with a blit of surface d with RGB (0, 0, 0)
        will maintain its RGB value regardless of opacity"""
        sA = 160
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((0, 0, 0, sA))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((0, 0, 0, sA))

        s.blit(d, (0, 0))

        self.assertEqual(s.get_at((0, 0))[:3], (0, 0, 0))

    def test_SRCALPHA_4(self):
        """blend(255, sA, 255) = 255 -> A surface s with RGB (255, 255, 255)
        at any opacity with a blit of surface d with RGB (255, 255, 255)
        will maintain its RGB value regardless of opacity"""
        sA = 160
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((255, 255, 255, sA))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((255, 255, 255, sA))

        s.blit(d, (0, 0))

        self.assertEqual(s.get_at((0, 0))[:3], (255, 255, 255))

    def test_SRCALPHA_5(self):
        """blend(s, sA, d) <= 255 -> Any surface s blitted with any
        surface d, regardless of opacity, should return an RGB value
        in the accepted value range of 0 to 255."""
        sA = 145
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((50, 255, 140, sA))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((255, 0, 25, sA))

        s.blit(d, (0, 0))

        self.assertLessEqual(s.get_at((0, 0))[:3], (255, 255, 255))

    def test_BLEND_saturated(self):
        """Test that no overflow occurs, and that surface is saturated.
        get_at() returns a RGBA color value"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((255, 255, 255, 0))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((0, 0, 255, 255))

        s.blit(d, (0, 0), None, BLEND_ADD)
        s.blit(d, (0, 0), None, BLEND_RGBA_ADD)

        self.assertEqual(s.get_at((0, 0))[3], 255)

    def test_BLEND_add(self):
        """Test adding works. BLEND_ADD adds two source pixels and clips
        the result at 255."""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((20, 255, 255, 0))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((10, 0, 255, 255))

        s.blit(d, (0, 0), None, BLEND_ADD)

        self.assertEqual(s.get_at((0, 0))[2], 255)

    def test_BLEND_sub(self):
        """Test subtraction works. BLEND_SUB subtracts two pixels and clips
        the result at 0."""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((20, 255, 255, 0))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((10, 0, 255, 255))

        s.blit(d, (0, 0), None, BLEND_SUB)

        self.assertEqual(s.get_at((0, 0))[0], 10)

    def test_BLEND_overflow(self):
        """Tests for no overflow in sub blend"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((20, 255, 255, 0))
        d = pygame.Surface((1, 1), SRCALPHA, 32)
        d.fill((30, 0, 255, 255))

        s.blit(d, (0, 0), None, BLEND_SUB)

        self.assertEqual(s.get_at((0, 0))[0], 0)

    def make_blit_list(self, num_surfs):
        """Helper function to make a list of blits."""
        blit_list = []
        for i in range(num_surfs):
            d = (i * 10, 0)
            s = pygame.Surface((10, 10), SRCALPHA, 32)
            color = (i * 1, i * 1, i * 1)
            s.fill(color)
            blit_list.append((s, d))
        return blit_list

    def test_blits(self):
        """In depth testing for blits(). Creates a number of surfaces to blit,
        checks time to blit multiple surfaces onto target surface."""
        NUM_SURFS = 255
        PRINT_TIMING = 0
        dst = pygame.Surface((NUM_SURFS * 10, 10), SRCALPHA, 32)
        dst.fill((230, 230, 230))
        blit_list = self.make_blit_list(NUM_SURFS)

        def blits(blit_list):
            for surface, dest in blit_list:
                dst.blit(surface, dest)

        t0 = time()
        blits(blit_list)
        t1 = time()
        if PRINT_TIMING:
            print(f"python blits: {t1 - t0}")

        dst.fill((230, 230, 230))
        t0 = time()
        results = dst.blits(blit_list)
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits :{t1 - t0}")

        # check if we blit all the different colors in the correct spots.
        for i in range(NUM_SURFS):
            color = (i * 1, i * 1, i * 1)
            self.assertEqual(dst.get_at((i * 10, 0)), color)
            self.assertEqual(dst.get_at(((i * 10) + 5, 5)), color)

        self.assertEqual(len(results), NUM_SURFS)

        t0 = time()
        results = dst.blits(blit_list, doreturn=0)
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits doreturn=0: {t1 - t0}")
        self.assertEqual(results, None)

        t0 = time()
        dst.blits(((surf, dest) for surf, dest in blit_list))
        t1 = time()
        if PRINT_TIMING:
            print(f"Surface.blits generator: {t1 - t0}")

    def test_blits_not_sequence(self):
        """Checks for a type error when blits() is not passed a sequence as
        input."""
        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        self.assertRaises(ValueError, dst.blits, None)

    def test_blits_wrong_length(self):
        """Checks for a type error when blits() is passed an argument of
        improper length (no destination)."""
        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        self.assertRaises(
            ValueError, dst.blits, [pygame.Surface((10, 10), SRCALPHA, 32)]
        )

    def test_blits_bad_surf_args(self):
        """Checks for a type error when blits() is passed illegal surface
        argument values."""
        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        self.assertRaises(TypeError, dst.blits, [(None, None)])

    def test_blits_bad_dest(self):
        """Checks for a type error when blits() is passed an illegal
        destination surface."""
        dst = pygame.Surface((100, 10), SRCALPHA, 32)
        self.assertRaises(
            TypeError, dst.blits, [(pygame.Surface((10, 10),
                                    SRCALPHA, 32), None)]
        )


if __name__ == "__main__":
    unittest.main()
