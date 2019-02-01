import unittest
import pygame
import pygame.mask
from pygame.locals import *

import random

def random_mask(size = (100,100)):
    """random_mask(size=(100,100)): return Mask
    Create a mask of the given size, with roughly half the bits set at random."""
    m = pygame.Mask(size)
    for i in range(size[0] * size[1] // 2):
        x, y = random.randint(0,size[0] - 1), random.randint(0, size[1] - 1)
        m.set_at((x,y))
    return m

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
    def assertMaskEquals(self, m1, m2):
        self.assertEqual(m1.get_size(), m2.get_size())

        for i in range(m1.get_size()[0]):
            for j in range(m1.get_size()[1]):
                self.assertEqual(m1.get_at((i,j)), m2.get_at((i,j)))

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

    def test_overlap_mask__count(self):
        """Ensure overlap_mask's mask has the correct count."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4))
        mask1.fill()
        mask2.fill()

        # Using rects to help determine the expected count of the overlapping
        # mask.
        rect1 = pygame.Rect((0, 0), mask1.get_size())
        rect2 = pygame.Rect((0, 0), mask2.get_size())
        offsets = ((0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1))

        for offset in offsets:
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = overlap_rect.w * overlap_rect.h

            overlap_mask = mask1.overlap_mask(mask2, offset)

            self.assertEqual(overlap_mask.count(), expected_count,
                             'offset={}'.format(offset))

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

    def test_erase(self):
        """Ensure erase() clears bits."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4))
        mask2.fill()

        # Using rects to help determine the overlapping area.
        rect1 = pygame.Rect((0, 0), mask1.get_size())
        rect2 = pygame.Rect((0, 0), mask2.get_size())
        rect1_area = rect1.w * rect1.h
        offsets = ((0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1))

        for offset in offsets:
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = rect1_area - (overlap_rect.w * overlap_rect.h)
            mask1.fill()  # Ensure it's filled for testing each offset.

            mask1.erase(mask2, offset)

            self.assertEqual(mask1.count(), expected_count,
                             'offset={}'.format(offset))

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

    def test_convolve__size(self):
        sizes = [(1,1), (31,31), (32,32), (100,100)]
        for s1 in sizes:
            m1 = pygame.Mask(s1)
            for s2 in sizes:
                m2 = pygame.Mask(s2)
                o = m1.convolve(m2)
                for i in (0,1):
                    self.assertEqual(o.get_size()[i],
                                     m1.get_size()[i] + m2.get_size()[i] - 1)

    def test_convolve__point_identities(self):
        """Convolving with a single point is the identity, while convolving a point with something flips it."""
        m = random_mask((100,100))
        k = pygame.Mask((1,1))
        k.set_at((0,0))

        self.assertMaskEquals(m,m.convolve(k))
        self.assertMaskEquals(m,k.convolve(k.convolve(m)))

    def test_convolve__with_output(self):
        """checks that convolution modifies only the correct portion of the output"""

        m = random_mask((10,10))
        k = pygame.Mask((2,2))
        k.set_at((0,0))

        o = pygame.Mask((50,50))
        test = pygame.Mask((50,50))

        m.convolve(k,o)
        test.draw(m,(1,1))
        self.assertMaskEquals(o, test)

        o.clear()
        test.clear()

        m.convolve(k,o, (10,10))
        test.draw(m,(11,11))
        self.assertMaskEquals(o, test)

    def test_convolve__out_of_range(self):
        full = pygame.Mask((2,2))
        full.fill()

        self.assertEqual(full.convolve(full, None, ( 0,  3)).count(), 0)
        self.assertEqual(full.convolve(full, None, ( 0,  2)).count(), 3)
        self.assertEqual(full.convolve(full, None, (-2, -2)).count(), 1)
        self.assertEqual(full.convolve(full, None, (-3, -3)).count(), 0)

    def test_convolve(self):
        """Tests the definition of convolution"""
        m1 = random_mask((100,100))
        m2 = random_mask((100,100))
        conv = m1.convolve(m2)

        for i in range(conv.get_size()[0]):
            for j in range(conv.get_size()[1]):
                self.assertEqual(conv.get_at((i,j)) == 0,
                                 m1.overlap(m2, (i - 99, j - 99)) is None)

    def test_connected_components(self):
        """
        """

        m = pygame.Mask((10,10))
        self.assertEqual(repr(m.connected_components()), "[]")

        comp = m.connected_component()
        self.assertEqual(m.count(), comp.count())

        m.set_at((0,0), 1)
        m.set_at((1,1), 1)
        comp = m.connected_component()
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEqual(comp.count(), comps[0].count())
        self.assertEqual(comps1[0].count(), 2)
        self.assertEqual(comps2[0].count(), 2)
        self.assertEqual(repr(comps3), "[]")

        m.set_at((9, 9), 1)
        comp = m.connected_component()
        comp1 = m.connected_component((1, 1))
        comp2 = m.connected_component((2, 2))
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEqual(comp.count(), 2)
        self.assertEqual(comp1.count(), 2)
        self.assertEqual(comp2.count(), 0)
        self.assertEqual(len(comps), 2)
        self.assertEqual(len(comps1), 2)
        self.assertEqual(len(comps2), 1)
        self.assertEqual(len(comps3), 0)


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

        self.assertEqual(
                repr(r),
                "[<rect(0, 0, 2, 2)>, <rect(0, 3, 1, 1)>, <rect(3, 3, 1, 1)>]")

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
        self.assertEqual(repr(r), "[<rect(0, 0, 4, 2)>]")

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
        self.assertEqual(repr(r), "[<rect(1, 0, 3, 3)>]")

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
        self.assertEqual(repr(r), "[<rect(1, 0, 3, 3)>]")

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
        #self.assertEqual(repr(r), "[<rect(0, 0, 5, 2)>]")

    def test_negative_size_mask(self):
        mask = pygame.Mask((100, 100))
        with self.assertRaises(ValueError):
            mask.scale((-1, -1))
        with self.assertRaises(ValueError):
            mask.scale((-1, 10))
        with self.assertRaises(ValueError):
            mask.scale((10, -1))


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

            rects = mask.get_bounding_rects()

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

    def test_overlap_mask(self):
        """Ensure overlap_mask's mask has correct bits set."""
        mask = pygame.mask.Mask((50, 50))
        mask.fill()
        mask2 = pygame.mask.Mask((300, 10))
        mask2.fill()
        mask3 = mask.overlap_mask(mask2, (-1, 0))

        for i in range(50):
            for j in range(10):
                self.assertEqual(mask3.get_at((i, j)), 1,
                                 '({}, {})'.format(i, j))

        for i in range(50):
            for j in range(11, 50):
                self.assertEqual(mask3.get_at((i, j)), 0,
                                 '({}, {})'.format(i, j))

    def test_zero_mask(self):
        mask = pygame.mask.Mask((0, 0))
        self.assertEqual(mask.get_size(), (0, 0))

        mask = pygame.mask.Mask((100, 0))
        self.assertEqual(mask.get_size(), (100, 0))

        mask = pygame.mask.Mask((0, 100))
        self.assertEqual(mask.get_size(), (0, 100))

    def test_zero_mask_overlap(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100))
            self.assertEqual(mask.overlap(mask2, (0, 0)), None)
            self.assertEqual(mask2.overlap(mask, (0, 0)), None)

    def test_zero_mask_overlap_area(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100))
            self.assertEqual(mask.overlap_area(mask2, (0, 0)), 0)
            self.assertEqual(mask2.overlap_area(mask, (0, 0)), 0)


    def test_zero_mask_overlap_mask(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100))

            overlap_mask = mask.overlap_mask(mask2, (0, 0))
            overlap_mask2 = mask2.overlap_mask(mask, (0, 0))

            self.assertEqual(mask.get_size(), overlap_mask.get_size())
            self.assertEqual(mask2.get_size(), overlap_mask2.get_size())

    def test_zero_mask_fill(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.fill()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_clear(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.clear()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_flip(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.invert()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_scale(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = mask.scale((2, 3))
            self.assertEqual(mask2.get_size(), (2, 3))

    def test_zero_mask_draw(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100))
            mask2.fill()
            before = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            mask.draw(mask2, (0, 0))
            after = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            self.assertEqual(before, after)

    def test_zero_mask_erase(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100))
            mask2.fill()
            before = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            mask.erase(mask2, (0, 0))
            after = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            self.assertEqual(before, after)

    def test_zero_mask_count(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.fill()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_centroid(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            self.assertEqual(mask.centroid(), (0, 0))

    def test_zero_mask_angle(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            self.assertEqual(mask.angle(), 0.0)

    def test_zero_size_from_surface(self):
        zero_w_mask = pygame.mask.from_surface(pygame.Surface((0, 100)))
        self.assertEqual(zero_w_mask.get_size(), (0, 100))

        zero_h_mask = pygame.mask.from_surface(pygame.Surface((100, 0)))
        self.assertEqual(zero_h_mask.get_size(), (100, 0))

        zero_mask = pygame.mask.from_surface(pygame.Surface((0, 0)))
        self.assertEqual(zero_mask.get_size(), (0, 0))

    def test_zero_size_from_threshold(self):
        a = [16, 24, 32]
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            for i in a:
                surf = pygame.surface.Surface(size, 0, i)
                surf.fill((100, 50, 200), (20, 20, 20, 20))
                mask = pygame.mask.from_threshold(surf, (100, 50, 200, 255), (10, 10, 10, 255))

                self.assertEqual(mask.count(), 0)

                rects = mask.get_bounding_rects()
                self.assertEqual(rects, [])

            for i in a:
                surf = pygame.surface.Surface(size, 0, i)
                surf2 = pygame.surface.Surface(size, 0, i)
                surf.fill((100, 100, 100))
                surf2.fill((150, 150, 150))
                surf2.fill((100, 100, 100), (40, 40, 10, 10))
                mask = pygame.mask.from_threshold(surf, (0, 0, 0, 0), (10, 10, 10, 255), surf2)

                self.assertEqual(mask.count(), 0)

                rects = mask.get_bounding_rects()
                self.assertEqual(rects, [])

if __name__ == '__main__':
    unittest.main()
