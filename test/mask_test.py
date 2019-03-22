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


class MaskTypeTest(unittest.TestCase):
    ORIGIN_OFFSETS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
                      (-1, -1), (-1, 0), (-1, 1))

    def _assertMaskEqual(self, m1, m2, msg=None):
        # Checks to see if the 2 given masks are equal.
        m1_count = m1.count()

        self.assertEqual(m1.get_size(), m2.get_size(), msg=msg)
        self.assertEqual(m1_count, m2.count(), msg=msg)
        self.assertEqual(m1_count, m1.overlap_area(m2, (0, 0)), msg=msg)

        # This can be used to help debug exact locations.
        ##for i in range(m1.get_size()[0]):
        ##    for j in range(m1.get_size()[1]):
        ##        self.assertEqual(m1.get_at((i, j)), m2.get_at((i, j)))

    def test_mask(self):
        """Ensure masks are created correctly without fill parameter."""
        expected_count = 0
        expected_size = (11, 23)
        mask1 = pygame.mask.Mask(expected_size)
        mask2 = pygame.mask.Mask(size=expected_size)

        self.assertEqual(mask1.count(), expected_count)
        self.assertEqual(mask1.get_size(), expected_size)

        self.assertEqual(mask2.count(), expected_count)
        self.assertEqual(mask2.get_size(), expected_size)

    def test_mask__negative_size(self):
        """Ensure the mask constructor handles negative sizes correctly."""
        for size in ((1, -1), (-1, 1), (-1, -1)):
            with self.assertRaises(ValueError):
                mask = pygame.Mask(size)

    def test_mask__fill_kwarg(self):
        """Ensure masks are created correctly using the fill keyword."""
        width, height = 37, 47
        expected_size = (width, height)
        fill_counts = {True : width * height, False : 0 }

        for fill, expected_count in fill_counts.items():
            msg = 'fill={}'.format(fill)

            mask = pygame.mask.Mask(expected_size, fill=fill)

            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)

    def test_mask__fill_arg(self):
        """Ensure masks are created correctly using a fill arg."""
        width, height = 59, 71
        expected_size = (width, height)
        fill_counts = {True : width * height, False : 0 }

        for fill, expected_count in fill_counts.items():
            msg = 'fill={}'.format(fill)

            mask = pygame.mask.Mask(expected_size, fill)

            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)

    def test_mask__size_kwarg(self):
        """Ensure masks are created correctly using the size keyword."""
        width, height = 73, 83
        expected_size = (width, height)
        fill_counts = {True : width * height, False : 0 }

        for fill, expected_count in fill_counts.items():
            msg = 'fill={}'.format(fill)

            mask1 = pygame.mask.Mask(fill=fill, size=expected_size)
            mask2 = pygame.mask.Mask(size=expected_size, fill=fill)

            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask2.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_get_size(self):
        """Ensure a mask's size is correctly retrieved."""
        expected_size = (93, 101)
        mask = pygame.mask.Mask(expected_size)

        self.assertEqual(mask.get_size(), expected_size)

    def test_get_at(self):
        """Ensure individual mask bits are correctly retrieved."""
        width, height = 5, 7
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_bit = 0
        mask1_expected_bit = 1
        pos = (width - 1, height - 1)

        # Check twice to make sure bits aren't toggled.
        self.assertEqual(mask0.get_at(pos), mask0_expected_bit)
        self.assertEqual(mask0.get_at(pos), mask0_expected_bit)
        self.assertEqual(mask1.get_at(pos), mask1_expected_bit)
        self.assertEqual(mask1.get_at(pos), mask1_expected_bit)

    def test_get_at__out_of_bounds(self):
        """Ensure get_at() checks bounds."""
        width, height = 11, 3
        mask = pygame.mask.Mask((width, height))

        with self.assertRaises(IndexError):
            mask.get_at((width, 0))

        with self.assertRaises(IndexError):
            mask.get_at((0, height))

        with self.assertRaises(IndexError):
            mask.get_at((-1, 0))

        with self.assertRaises(IndexError):
            mask.get_at((0, -1))

    def test_set_at(self):
        """Ensure individual mask bits are set to 1."""
        width, height = 13, 17
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 1
        mask1_expected_count = mask1.count()
        expected_bit = 1
        pos = (width - 1, height - 1)

        mask0.set_at(pos, expected_bit)  # set 0 to 1
        mask1.set_at(pos, expected_bit)  # set 1 to 1

        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__to_0(self):
        """Ensure individual mask bits are set to 0."""
        width, height = 11, 7
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 0
        mask1_expected_count = mask1.count() - 1
        expected_bit = 0
        pos = (width - 1, height - 1)

        mask0.set_at(pos, expected_bit)  # set 0 to 0
        mask1.set_at(pos, expected_bit)  # set 1 to 0

        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__default_value(self):
        """Ensure individual mask bits are set using the default value."""
        width, height = 3, 21
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 1
        mask1_expected_count = mask1.count()
        expected_bit = 1
        pos = (width - 1, height - 1)

        mask0.set_at(pos)  # set 0 to 1
        mask1.set_at(pos)  # set 1 to 1

        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__out_of_bounds(self):
        """Ensure set_at() checks bounds."""
        width, height = 11, 3
        mask = pygame.mask.Mask((width, height))

        with self.assertRaises(IndexError):
            mask.set_at((width, 0))

        with self.assertRaises(IndexError):
            mask.set_at((0, height))

        with self.assertRaises(IndexError):
            mask.set_at((-1, 0))

        with self.assertRaises(IndexError):
            mask.set_at((0, -1))

    def test_overlap(self):
        """Ensure the overlap intersection is correctly calculated.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap 1 (mask2-filled)
            (mask1-empty)  0 overlap 1 (mask2-filled)
            (mask1-filled) 1 overlap 0 (mask2-empty)
            (mask1-empty)  0 overlap 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = None
        expected_overlaps = {(True, True) : offset}

        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()

            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = 'key={}'.format(key)
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_pos = expected_overlaps.get(key, expected_default)

                overlap_pos = mask1.overlap(mask2, offset)

                self.assertEqual(overlap_pos, expected_pos, msg)

                # Ensure mask1/mask2 unchanged.
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_overlap__offset(self):
        """Ensure an offset overlap intersection is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            expected_pos = (max(offset[0], 0), max(offset[1], 0))

            overlap_pos = mask1.overlap(mask2, offset)

            self.assertEqual(overlap_pos, expected_pos, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap__offset_with_unset_bits(self):
        """Ensure an offset overlap intersection is correctly calculated
        when (0, 0) bits not set."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        unset_pos = (0, 0)
        mask1.set_at(unset_pos, 0)
        mask2.set_at(unset_pos, 0)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            x, y = offset
            expected_y = max(y, 0)
            if 0 == y:
                expected_x = max(x + 1, 1)
            elif 0 < y:
                expected_x = max(x + 1, 0)
            else:
                expected_x = max(x, 1)

            overlap_pos = mask1.overlap(mask2, offset)

            self.assertEqual(overlap_pos, (expected_x, expected_y), msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)
            self.assertEqual(mask1.get_at(unset_pos), 0, msg)
            self.assertEqual(mask2.get_at(unset_pos), 0, msg)

    def test_overlap__no_overlap(self):
        """Ensure an offset overlap intersection is correctly calculated
        when there is no overlap."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask1_count = mask1.count()
        mask1_size = mask1.get_size()

        mask2_w, mask2_h = 67, 5
        mask2_size = (mask2_w, mask2_h)
        mask2 = pygame.mask.Mask(mask2_size)
        set_pos = (mask2_w - 1, mask2_h - 1)
        mask2.set_at(set_pos)
        mask2_count = 1

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)

            overlap_pos = mask1.overlap(mask2, offset)

            self.assertIsNone(overlap_pos, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)
            self.assertEqual(mask2.get_at(set_pos), 1, msg)

    def test_overlap__invalid_mask_arg(self):
        """Ensure overlap handles invalid mask arguments correctly."""
        size = (5, 3)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)

        with self.assertRaises(TypeError):
            overlap_pos = mask.overlap(invalid_mask, offset)

    def test_overlap__invalid_offset_arg(self):
        """Ensure overlap handles invalid offset arguments correctly."""
        size = (2, 7)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)

        with self.assertRaises(TypeError):
            overlap_pos = mask1.overlap(mask2, offset)

    def test_overlap_area(self):
        """Ensure the overlap_area is correctly calculated.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_area 1 (mask2-filled)
            (mask1-empty)  0 overlap_area 1 (mask2-filled)
            (mask1-filled) 1 overlap_area 0 (mask2-empty)
            (mask1-empty)  0 overlap_area 0 (mask2-empty)
        """
        expected_size = width, height = (4, 4)
        offset = (0, 0)
        expected_default = 0
        expected_counts = {(True, True) : width * height}

        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()

            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = 'key={}'.format(key)
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_count = expected_counts.get(key, expected_default)

                overlap_count = mask1.overlap_area(mask2, offset)

                self.assertEqual(overlap_count, expected_count, msg)

                # Ensure mask1/mask2 unchanged.
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_overlap_area__offset(self):
        """Ensure an offset overlap_area is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Using rects to help determine the overlapping area.
        rect1 = pygame.Rect((0, 0), mask1_size)
        rect2 = pygame.Rect((0, 0), mask2_size)

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = overlap_rect.w * overlap_rect.h

            overlap_count = mask1.overlap_area(mask2, offset)

            self.assertEqual(overlap_count, expected_count, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap_area__invalid_mask_arg(self):
        """Ensure overlap_area handles invalid mask arguments correctly."""
        size = (3, 5)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)

        with self.assertRaises(TypeError):
            overlap_count = mask.overlap_area(invalid_mask, offset)

    def test_overlap_area__invalid_offset_arg(self):
        """Ensure overlap_area handles invalid offset arguments correctly."""
        size = (7, 2)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)

        with self.assertRaises(TypeError):
            overlap_count = mask1.overlap_area(mask2, offset)

    def test_overlap_mask(self):
        """Ensure overlap_mask's mask has correct bits set.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_mask 1 (mask2-filled)
            (mask1-empty)  0 overlap_mask 1 (mask2-filled)
            (mask1-filled) 1 overlap_mask 0 (mask2-empty)
            (mask1-empty)  0 overlap_mask 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size)
        expected_masks = {
            (True, True) : pygame.mask.Mask(expected_size, fill=True)}

        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()

            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = 'key={}'.format(key)
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_mask = expected_masks.get(key, expected_default)

                overlap_mask = mask1.overlap_mask(mask2, offset)

                self._assertMaskEqual(overlap_mask, expected_mask, msg)

                # Ensure mask1/mask2 unchanged.
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_overlap_mask__bits_set(self):
        """Ensure overlap_mask's mask has correct bits set."""
        mask1 = pygame.mask.Mask((50, 50), fill=True)
        mask2 = pygame.mask.Mask((300, 10), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        mask3 = mask1.overlap_mask(mask2, (-1, 0))

        for i in range(50):
            for j in range(10):
                self.assertEqual(mask3.get_at((i, j)), 1,
                                 '({}, {})'.format(i, j))

        for i in range(50):
            for j in range(11, 50):
                self.assertEqual(mask3.get_at((i, j)), 0,
                                 '({}, {})'.format(i, j))

        # Ensure mask1/mask2 unchanged.
        self.assertEqual(mask1.count(), mask1_count)
        self.assertEqual(mask2.count(), mask2_count)
        self.assertEqual(mask1.get_size(), mask1_size)
        self.assertEqual(mask2.get_size(), mask2_size)

    def test_overlap_mask__offset(self):
        """Ensure an offset overlap_mask's mask is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        expected_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Using rects to help determine the overlapping area.
        rect1 = pygame.Rect((0, 0), expected_size)
        rect2 = pygame.Rect((0, 0), mask2_size)

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = overlap_rect.w * overlap_rect.h

            overlap_mask = mask1.overlap_mask(mask2, offset)

            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), expected_size, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap_mask__invalid_mask_arg(self):
        """Ensure overlap_mask handles invalid mask arguments correctly."""
        size = (3, 2)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)

        with self.assertRaises(TypeError):
            overlap_mask = mask.overlap_mask(invalid_mask, offset)

    def test_overlap_mask__invalid_offset_arg(self):
        """Ensure overlap_mask handles invalid offset arguments correctly."""
        size = (5, 2)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)

        with self.assertRaises(TypeError):
            overlap_mask = mask1.overlap_mask(mask2, offset)

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

    def test_fill(self):
        """Ensure a mask can be filled."""
        width, height = 11, 23
        expected_count = width * height
        expected_size = (width, height)
        mask = pygame.mask.Mask(expected_size)

        mask.fill()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_clear(self):
        """Ensure a mask can be cleared."""
        expected_count = 0
        expected_size = (13, 27)
        mask = pygame.mask.Mask(expected_size, fill=True)

        mask.clear()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_invert(self):
        """Ensure a mask can be inverted."""
        side = 73
        expected_size = (side, side)
        mask1 = pygame.mask.Mask(expected_size)
        mask2 = pygame.mask.Mask(expected_size, fill=True)
        expected_count1 = side * side
        expected_count2 = 0

        for i in range(side):
            expected_count1 -= 1
            expected_count2 += 1
            pos = (i, i)
            mask1.set_at(pos)
            mask2.set_at(pos, 0)

        mask1.invert()
        mask2.invert()

        self.assertEqual(mask1.count(), expected_count1)
        self.assertEqual(mask2.count(), expected_count2)
        self.assertEqual(mask1.get_size(), expected_size)
        self.assertEqual(mask2.get_size(), expected_size)

        for i in range(side):
            pos = (i, i)
            msg = 'pos={}'.format(pos)

            self.assertEqual(mask1.get_at(pos), 0, msg)
            self.assertEqual(mask2.get_at(pos), 1, msg)

    def test_invert__full(self):
        """Ensure a full mask can be inverted."""
        expected_count = 0
        expected_size = (43, 97)
        mask = pygame.mask.Mask(expected_size, fill=True)

        mask.invert()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_invert__empty(self):
        """Ensure an empty mask can be inverted."""
        width, height = 43, 97
        expected_size = (width, height)
        expected_count = width * height
        mask = pygame.mask.Mask(expected_size)

        mask.invert()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_scale(self):
        """Ensure a mask can be scaled."""
        width, height = 43, 61
        original_size = (width, height)

        for fill in (True, False):
            original_mask = pygame.mask.Mask(original_size, fill=fill)
            original_count = width * height if fill else 0

            # Test a range of sizes. Also tests scaling to 'same'
            # size when new_w, new_h = width, height
            for new_w in range(width - 10, width + 10):
                for new_h in range(height - 10, height + 10):
                    expected_size = (new_w, new_h)
                    expected_count = new_w * new_h if fill else 0
                    msg = 'size={}'.format(expected_size)

                    mask = original_mask.scale(expected_size)

                    self.assertEqual(mask.count(), expected_count, msg)
                    self.assertEqual(mask.get_size(), expected_size)

                    # Ensure the original mask is unchanged.
                    self.assertEqual(original_mask.count(), original_count,
                                     msg)
                    self.assertEqual(original_mask.get_size(), original_size,
                                     msg)

    def test_scale__negative_size(self):
        """Ensure scale handles negative sizes correctly."""
        mask = pygame.Mask((100, 100))

        with self.assertRaises(ValueError):
            mask.scale((-1, -1))

        with self.assertRaises(ValueError):
            mask.scale((-1, 10))

        with self.assertRaises(ValueError):
            mask.scale((10, -1))

    def test_draw(self):
        """Ensure a mask can be drawn onto another mask.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 draw 1 (mask2-filled)
            (mask1-empty)  0 draw 1 (mask2-filled)
            (mask1-filled) 1 draw 0 (mask2-empty)
            (mask1-empty)  0 draw 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size, fill=True)
        expected_masks = {(False, False) : pygame.mask.Mask(expected_size)}

        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()

            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = 'key={}'.format(key)
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                expected_mask = expected_masks.get(key, expected_default)

                mask1.draw(mask2, offset)

                self._assertMaskEqual(mask1, expected_mask, msg)

                # Ensure mask2 unchanged.
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_draw__offset(self):
        """Ensure an offset mask can be drawn onto another mask."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Using rects to help determine the overlapping area.
        rect1 = pygame.Rect((0, 0), mask1_size)
        rect2 = pygame.Rect((0, 0), mask2_size)

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = overlap_rect.w * overlap_rect.h
            mask1.clear()  # Ensure it's empty for testing each offset.

            mask1.draw(mask2, offset)

            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)

            # Ensure mask2 unchanged.
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_draw__invalid_mask_arg(self):
        """Ensure draw handles invalid mask arguments correctly."""
        size = (7, 3)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)

        with self.assertRaises(TypeError):
            mask.draw(invalid_mask, offset)

    def test_draw__invalid_offset_arg(self):
        """Ensure draw handles invalid offset arguments correctly."""
        size = (5, 7)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)

        with self.assertRaises(TypeError):
            mask1.draw(mask2, offset)

    def test_erase(self):
        """Ensure a mask can erase another mask.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 erase 1 (mask2-filled)
            (mask1-empty)  0 erase 1 (mask2-filled)
            (mask1-filled) 1 erase 0 (mask2-empty)
            (mask1-empty)  0 erase 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size)
        expected_masks = {
                (True, False) : pygame.mask.Mask(expected_size, fill=True)}

        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()

            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = 'key={}'.format(key)
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                expected_mask = expected_masks.get(key, expected_default)

                mask1.erase(mask2, offset)

                self._assertMaskEqual(mask1, expected_mask, msg)

                # Ensure mask2 unchanged.
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_erase__offset(self):
        """Ensure an offset mask can erase another mask."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Using rects to help determine the overlapping area.
        rect1 = pygame.Rect((0, 0), mask1_size)
        rect2 = pygame.Rect((0, 0), mask2_size)
        rect1_area = rect1.w * rect1.h

        for offset in self.ORIGIN_OFFSETS:
            msg = 'offset={}'.format(offset)
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = rect1_area - (overlap_rect.w * overlap_rect.h)
            mask1.fill()  # Ensure it's filled for testing each offset.

            mask1.erase(mask2, offset)

            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)

            # Ensure mask2 unchanged.
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_erase__invalid_mask_arg(self):
        """Ensure erase handles invalid mask arguments correctly."""
        size = (3, 7)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)

        with self.assertRaises(TypeError):
            mask.erase(invalid_mask, offset)

    def test_erase__invalid_offset_arg(self):
        """Ensure erase handles invalid offset arguments correctly."""
        size = (7, 5)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)

        with self.assertRaises(TypeError):
            mask1.erase(mask2, offset)

    def test_count(self):
        """Ensure a mask's set bits are correctly counted."""
        side = 67
        expected_size = (side, side)
        expected_count = 0
        mask = pygame.mask.Mask(expected_size)

        for i in range(side):
            expected_count += 1
            mask.set_at((i, i))

        count = mask.count()

        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_count__full_mask(self):
        """Ensure a full mask's set bits are correctly counted."""
        width, height = 17, 97
        expected_size = (width, height)
        expected_count = width * height
        mask = pygame.mask.Mask(expected_size, fill=True)

        count = mask.count()

        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_count__empty_mask(self):
        """Ensure an empty mask's set bits are correctly counted."""
        expected_count = 0
        expected_size = (13, 27)
        mask = pygame.mask.Mask(expected_size)

        count = mask.count()

        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def todo_test_centroid(self):
        """Ensure a mask's centroid is correctly calculated."""
        self.fail()

    def test_centroid__empty_mask(self):
        """Ensure an empty mask's centroid is correctly calculated."""
        expected_centroid = (0, 0)
        expected_size = (101, 103)
        mask = pygame.mask.Mask(expected_size)

        centroid = mask.centroid()

        self.assertEqual(centroid, expected_centroid)
        self.assertEqual(mask.get_size(), expected_size)

    def todo_test_angle(self):
        """Ensure a mask's orientation angle is correctly calculated."""
        self.fail()

    def test_angle__empty_mask(self):
        """Ensure an empty mask's angle is correctly calculated."""
        expected_angle = 0.0
        expected_size = (107, 43)
        mask = pygame.mask.Mask(expected_size)

        angle = mask.angle()

        self.assertIsInstance(angle, float)
        self.assertAlmostEqual(angle, expected_angle)
        self.assertEqual(mask.get_size(), expected_size)

    def test_drawing(self):
        """ Test fill, clear, invert, draw, erase
        """
        m = pygame.Mask((100,100))
        self.assertEqual(m.count(), 0)

        m.fill()
        self.assertEqual(m.count(), 10000)

        m2 = pygame.Mask((10, 10), fill=True)
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

        self._assertMaskEqual(m, m.convolve(k))
        self._assertMaskEqual(m, k.convolve(k.convolve(m)))

    def test_convolve__with_output(self):
        """checks that convolution modifies only the correct portion of the output"""

        m = random_mask((10,10))
        k = pygame.Mask((2,2))
        k.set_at((0,0))

        o = pygame.Mask((50,50))
        test = pygame.Mask((50,50))

        m.convolve(k,o)
        test.draw(m,(1,1))
        self._assertMaskEqual(o, test)

        o.clear()
        test.clear()

        m.convolve(k,o, (10,10))
        test.draw(m,(11,11))
        self._assertMaskEqual(o, test)

    def test_convolve__out_of_range(self):
        full = pygame.Mask((2, 2), fill=True)

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

    def _draw_component_pattern_box(self, mask, size, pos, inverse=False):
        # Helper method to create/draw a 'box' pattern for testing.
        #
        # 111
        # 101  3x3 example pattern
        # 111
        pattern = pygame.mask.Mask((size, size), fill=True)
        pattern.set_at((size // 2, size // 2), 0)

        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)

        return pattern

    def _draw_component_pattern_x(self, mask, size, pos, inverse=False):
        # Helper method to create/draw an 'X' pattern for testing.
        #
        # 101
        # 010  3x3 example pattern
        # 101
        pattern = pygame.mask.Mask((size, size))

        ymax = size - 1
        for y in range(size):
            for x in range(size):
                if x == y or x == ymax - y:
                    pattern.set_at((x, y))

        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)

        return pattern

    def _draw_component_pattern_plus(self, mask, size, pos, inverse=False):
        # Helper method to create/draw a '+' pattern for testing.
        #
        # 010
        # 111  3x3 example pattern
        # 010
        pattern = pygame.mask.Mask((size, size))

        xmid = ymid = size // 2
        for y in range(size):
            for x in range(size):
                if x == xmid or y == ymid:
                    pattern.set_at((x, y))

        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)

        return pattern

    def test_connected_component(self):
        """Ensure a mask's connected component is correctly calculated."""
        width, height = 41, 27
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size)
        patterns = []  # Patterns and offsets.

        # Draw some connected patterns on the original mask.
        offset = (0, 0)
        pattern = self._draw_component_pattern_x(original_mask, 3, offset)
        patterns.append((pattern, offset))

        size = 4
        offset = (width - size, 0)
        pattern = self._draw_component_pattern_plus(original_mask, size,
                                                    offset)
        patterns.append((pattern, offset))

        # Make this one the largest connected component.
        offset = (width // 2, height // 2)
        pattern = self._draw_component_pattern_box(original_mask, 7, offset)
        patterns.append((pattern, offset))

        expected_pattern, expected_offset = patterns[-1]
        expected_count = expected_pattern.count()
        original_count = sum(p.count() for p, _ in patterns)

        mask = original_mask.connected_component()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset),
                         expected_count)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)

        for pattern, offset in patterns:
            self.assertEqual(original_mask.overlap_area(pattern, offset),
                             pattern.count())

    def test_connected_component__full_mask(self):
        """Ensure a mask's connected component is correctly calculated
        when the mask is full."""
        expected_size = (23, 31)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        expected_count = original_mask.count()

        mask = original_mask.connected_component()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), expected_count)
        self.assertEqual(original_mask.get_size(), expected_size)

    def test_connected_component__empty_mask(self):
        """Ensure a mask's connected component is correctly calculated
        when the mask is empty."""
        expected_size = (37, 43)
        original_mask = pygame.mask.Mask(expected_size)
        original_count = original_mask.count()
        expected_count = 0

        mask = original_mask.connected_component()

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)

    def test_connected_component__one_set_bit(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of 1 bit.
        """
        width, height = 71, 67
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        xset, yset = width // 2, height // 2
        set_pos = (xset, yset)
        expected_offset = (xset - 1, yset - 1)

        # This isolates the bit at set_pos from all the other bits.
        expected_pattern = self._draw_component_pattern_box(original_mask, 3,
            expected_offset, inverse=True)
        expected_count = 1
        original_count = original_mask.count()

        mask = original_mask.connected_component(set_pos)

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset),
                         expected_count)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.overlap_area(
            expected_pattern, expected_offset), expected_count)

    def test_connected_component__multi_set_bits(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of > 1 bit.
        """
        expected_size = (113, 67)
        original_mask = pygame.mask.Mask(expected_size)
        p_width, p_height = 11, 13
        set_pos = xset, yset = 11, 21
        expected_offset = (xset - 1, yset - 1)
        expected_pattern = pygame.mask.Mask((p_width, p_height), fill=True)

        # Make an unsymmetrical pattern. All the set bits need to be connected
        # in the resulting pattern for this to work properly.
        for y in range(3, p_height):
            for x in range(1, p_width):
                if x == y or x == y - 3 or x == p_width - 4:
                    expected_pattern.set_at((x, y), 0)

        expected_count = expected_pattern.count()
        original_mask.draw(expected_pattern, expected_offset)

        mask = original_mask.connected_component(set_pos)

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset),
                         expected_count)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), expected_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.overlap_area(
            expected_pattern, expected_offset), expected_count)

    def test_connected_component__unset_bit(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is unset.
        """
        width, height = 109, 101
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        unset_pos = (width // 2, height // 2)
        original_mask.set_at(unset_pos, 0)
        original_count = original_mask.count()
        expected_count = 0

        mask = original_mask.connected_component(unset_pos)

        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.get_at(unset_pos), 0)

    def test_connected_component__out_of_bounds(self):
        """Ensure connected_component() checks bounds."""
        width, height = 19, 11
        original_size = (width, height)
        original_mask = pygame.mask.Mask(original_size, fill=True)
        original_count = original_mask.count()

        for pos in ((0, -1), (-1, 0), (0, height + 1), (width + 1, 0)):
            with self.assertRaises(IndexError):
                mask = original_mask.connected_component(pos)

            # Ensure the original mask is unchanged.
            self.assertEqual(original_mask.count(), original_count)
            self.assertEqual(original_mask.get_size(), original_size)

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

    def test_zero_mask(self):
        mask = pygame.mask.Mask((0, 0))
        self.assertEqual(mask.get_size(), (0, 0))

        mask = pygame.mask.Mask((100, 0))
        self.assertEqual(mask.get_size(), (100, 0))

        mask = pygame.mask.Mask((0, 100))
        self.assertEqual(mask.get_size(), (0, 100))

    def test_zero_mask_get_size(self):
        """Ensures get_size correctly handles zero sized masks."""
        for expected_size in ((41, 0), (0, 40), (0, 0)):
            mask = pygame.mask.Mask(expected_size)

            size = mask.get_size()

            self.assertEqual(size, expected_size)

    def test_zero_mask_get_at(self):
        """Ensures get_at correctly handles zero sized masks."""
        for size in ((51, 0), (0, 50), (0, 0)):
            mask = pygame.mask.Mask(size)

            with self.assertRaises(IndexError):
                value = mask.get_at((0, 0))

    def test_zero_mask_set_at(self):
        """Ensures set_at correctly handles zero sized masks."""
        for size in ((31, 0), (0, 30), (0, 0)):
            mask = pygame.mask.Mask(size)

            with self.assertRaises(IndexError):
                mask.set_at((0, 0))

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
            mask = pygame.mask.Mask(size, fill=True)
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
            mask2 = pygame.mask.Mask((100, 100), fill=True)
            before = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            mask.draw(mask2, (0, 0))
            after = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            self.assertEqual(before, after)

    def test_zero_mask_erase(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = pygame.mask.Mask((100, 100), fill=True)
            before = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            mask.erase(mask2, (0, 0))
            after = [mask2.get_at((x, y)) for x in range(100) for y in range(100)]
            self.assertEqual(before, after)

    def test_zero_mask_count(self):
        sizes = ((100, 0), (0, 100), (0, 0))

        for size in sizes:
            mask = pygame.mask.Mask(size, fill=True)
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

    def test_zero_mask_outline(self):
        """Ensures outline correctly handles zero sized masks."""
        expected_points = []

        for size in ((61, 0), (0, 60), (0, 0)):
            mask = pygame.mask.Mask(size)

            points = mask.outline()

            self.assertListEqual(points, expected_points,
                                 'size={}'.format(size))

    def test_zero_mask_outline__with_arg(self):
        """Ensures outline correctly handles zero sized masks
        when using the skip pixels argument."""
        expected_points = []

        for size in ((66, 0), (0, 65), (0, 0)):
            mask = pygame.mask.Mask(size)

            points = mask.outline(10)

            self.assertListEqual(points, expected_points,
                                 'size={}'.format(size))

    @unittest.skip('can cause segmentation fault')
    def test_zero_mask_convolve(self):
        """Ensures convolve correctly handles zero sized masks.

        Tests the different combinations of sized and zero sized masks.
        """
        for size1 in ((17, 13), (71, 0), (0, 70), (0, 0)):
            mask1 = pygame.mask.Mask(size1, fill=True)

            for size2 in ((11, 7), (81, 0), (0, 60), (0, 0)):
                msg = 'sizes={}, {}'.format(size1, size2)
                mask2 = pygame.mask.Mask(size2, fill=True)
                expected_size = (max(0, size1[0] + size2[0] - 1),
                                 max(0, size1[1] + size2[1] - 1))

                mask = mask1.convolve(mask2)

                self.assertIsNot(mask, mask2, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)

    def test_zero_mask_convolve__with_output_mask(self):
        """Ensures convolve correctly handles zero sized masks
        when using an output mask argument.

        Tests the different combinations of sized and zero sized masks.
        """
        for size1 in ((11, 17), (91, 0), (0, 90), (0, 0)):
            mask1 = pygame.mask.Mask(size1, fill=True)

            for size2 in ((13, 11), (83, 0), (0, 62), (0, 0)):
                mask2 = pygame.mask.Mask(size2, fill=True)

                for output_size in ((7, 5), (71, 0), (0, 70), (0, 0)):
                    msg = 'sizes={}, {}, {}'.format(size1, size2, output_size)
                    output_mask = pygame.mask.Mask(output_size)

                    mask = mask1.convolve(mask2, output_mask)

                    self.assertIs(mask, output_mask, msg)
                    self.assertEqual(mask.get_size(), output_size, msg)

    def test_zero_mask_connected_component(self):
        """Ensures connected_component correctly handles zero sized masks."""
        expected_count = 0

        for size in ((81, 0), (0, 80), (0, 0)):
            mask = pygame.mask.Mask(size)

            cc_mask = mask.connected_component()

            self.assertEqual(cc_mask.get_size(), size)
            self.assertEqual(cc_mask.count(), expected_count,
                             'size={}'.format(size))

    def test_zero_mask_connected_component__indexed(self):
        """Ensures connected_component correctly handles zero sized masks
        when using an index argument."""
        for size in ((91, 0), (0, 90), (0, 0)):
            mask = pygame.mask.Mask(size)

            with self.assertRaises(IndexError):
                cc_mask = mask.connected_component((0, 0))

    def test_zero_mask_connected_components(self):
        """Ensures connected_components correctly handles zero sized masks."""
        expected_cc_masks = []

        for size in ((11, 0), (0, 10), (0, 0)):
            mask = pygame.mask.Mask(size)

            cc_masks = mask.connected_components()

            self.assertListEqual(cc_masks, expected_cc_masks,
                                 'size={}'.format(size))

    def test_zero_mask_get_bounding_rects(self):
        """Ensures get_bounding_rects correctly handles zero sized masks."""
        expected_bounding_rects = []

        for size in ((21, 0), (0, 20), (0, 0)):
            mask = pygame.mask.Mask(size)

            bounding_rects = mask.get_bounding_rects()

            self.assertListEqual(bounding_rects, expected_bounding_rects,
                                 'size={}'.format(size))


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
