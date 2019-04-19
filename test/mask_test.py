from collections import OrderedDict
import random
import unittest

import pygame
from pygame.locals import *


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


def create_bounding_rect(points):
    """Creates a bounding rect from the given points."""
    xmin = xmax = points[0][0]
    ymin = ymax = points[0][1]

    for x, y in points[1:]:
        xmin = min(x, xmin)
        xmax = max(x, xmax)
        ymin = min(y, ymin)
        ymax = max(y, ymax)

    return pygame.Rect((xmin, ymin), (xmax - xmin + 1, ymax - ymin + 1))


def zero_size_pairs(width, height):
    """Creates a generator which yields pairs of sizes.

    For each pair of sizes at least one of the sizes will have a 0 in it.
    """
    sizes = ((width, height), (width, 0), (0, height), (0, 0))

    return ((a, b) for a in sizes for b in sizes if 0 in a or 0 in b)


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

        self.assertIsInstance(mask1, pygame.mask.Mask)
        self.assertEqual(mask1.count(), expected_count)
        self.assertEqual(mask1.get_size(), expected_size)

        self.assertIsInstance(mask2, pygame.mask.Mask)
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

            self.assertIsInstance(mask, pygame.mask.Mask, msg)
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

            self.assertIsInstance(mask, pygame.mask.Mask, msg)
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

            self.assertIsInstance(mask1, pygame.mask.Mask, msg)
            self.assertIsInstance(mask2, pygame.mask.Mask, msg)
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

    def test_overlap__offset_boundary(self):
        """Ensures overlap handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((13, 3), fill=True)
        mask2 = pygame.mask.Mask((7, 5), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Check the 4 boundaries.
        offsets = ((mask1_size[0], 0),   # off right
                   (0, mask1_size[1]),   # off bottom
                   (-mask2_size[0], 0),  # off left
                   (0, -mask2_size[1]))  # off top

        for offset in offsets:
            msg = 'offset={}'.format(offset)

            overlap_pos = mask1.overlap(mask2, offset)

            self.assertIsNone(overlap_pos, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

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

    def test_overlap_area__offset_boundary(self):
        """Ensures overlap_area handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((11, 3), fill=True)
        mask2 = pygame.mask.Mask((5, 7), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        expected_count = 0

        # Check the 4 boundaries.
        offsets = ((mask1_size[0], 0),   # off right
                   (0, mask1_size[1]),   # off bottom
                   (-mask2_size[0], 0),  # off left
                   (0, -mask2_size[1]))  # off top

        for offset in offsets:
            msg = 'offset={}'.format(offset)

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

                self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
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

            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), expected_size, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap_mask__offset_boundary(self):
        """Ensures overlap_mask handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((9, 3), fill=True)
        mask2 = pygame.mask.Mask((11, 5), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        expected_count = 0
        expected_size = mask1_size

        # Check the 4 boundaries.
        offsets = ((mask1_size[0], 0),   # off right
                   (0, mask1_size[1]),   # off bottom
                   (-mask2_size[0], 0),  # off left
                   (0, -mask2_size[1]))  # off top

        for offset in offsets:
            msg = 'offset={}'.format(offset)

            overlap_mask = mask1.overlap_mask(mask2, offset)

            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), expected_size, msg)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
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

                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
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

    def test_draw__offset_boundary(self):
        """Ensures draw handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((13, 5))
        mask2 = pygame.mask.Mask((7, 3), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Check the 4 boundaries.
        offsets = ((mask1_size[0], 0),   # off right
                   (0, mask1_size[1]),   # off bottom
                   (-mask2_size[0], 0),  # off left
                   (0, -mask2_size[1]))  # off top

        for offset in offsets:
            msg = 'offset={}'.format(offset)

            mask1.draw(mask2, offset)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
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

    def test_erase__offset_boundary(self):
        """Ensures erase handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((7, 11), fill=True)
        mask2 = pygame.mask.Mask((3, 13), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()

        # Check the 4 boundaries.
        offsets = ((mask1_size[0], 0),   # off right
                   (0, mask1_size[1]),   # off bottom
                   (-mask2_size[0], 0),  # off left
                   (0, -mask2_size[1]))  # off top

        for offset in offsets:
            msg = 'offset={}'.format(offset)

            mask1.erase(mask2, offset)

            # Ensure mask1/mask2 unchanged.
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
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

                self.assertIsInstance(o, pygame.mask.Mask)

                for i in (0,1):
                    self.assertEqual(o.get_size()[i],
                                     m1.get_size()[i] + m2.get_size()[i] - 1)

    def test_convolve__point_identities(self):
        """Convolving with a single point is the identity, while convolving a point with something flips it."""
        m = random_mask((100,100))
        k = pygame.Mask((1,1))
        k.set_at((0,0))

        convolve_mask = m.convolve(k)

        self.assertIsInstance(convolve_mask, pygame.mask.Mask)
        self._assertMaskEqual(m, convolve_mask)

        convolve_mask = k.convolve(k.convolve(m))

        self.assertIsInstance(convolve_mask, pygame.mask.Mask)
        self._assertMaskEqual(m, convolve_mask)

    def test_convolve__with_output(self):
        """checks that convolution modifies only the correct portion of the output"""

        m = random_mask((10,10))
        k = pygame.Mask((2,2))
        k.set_at((0,0))

        o = pygame.Mask((50,50))
        test = pygame.Mask((50,50))

        m.convolve(k,o)
        test.draw(m,(1,1))

        self.assertIsInstance(o, pygame.mask.Mask)
        self._assertMaskEqual(o, test)

        o.clear()
        test.clear()

        m.convolve(k,o, (10,10))
        test.draw(m,(11,11))

        self.assertIsInstance(o, pygame.mask.Mask)
        self._assertMaskEqual(o, test)

    def test_convolve__out_of_range(self):
        full = pygame.Mask((2, 2), fill=True)
        # Tuple of points (out of range) and the expected count for each.
        pts_data = (((0, 3), 0), ((0, 2), 3), ((-2, -2), 1), ((-3, -3), 0))

        for pt, expected_count in pts_data:
            convolve_mask = full.convolve(full, None, pt)

            self.assertIsInstance(convolve_mask, pygame.mask.Mask)
            self.assertEqual(convolve_mask.count(), expected_count)

    def test_convolve(self):
        """Tests the definition of convolution"""
        m1 = random_mask((100,100))
        m2 = random_mask((100,100))
        conv = m1.convolve(m2)

        self.assertIsInstance(conv, pygame.mask.Mask)
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

        self.assertIsInstance(mask, pygame.mask.Mask)
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
        when the mask is full.
        """
        expected_size = (23, 31)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        expected_count = original_mask.count()

        mask = original_mask.connected_component()

        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

        # Ensure the original mask is unchanged.
        self.assertEqual(original_mask.count(), expected_count)
        self.assertEqual(original_mask.get_size(), expected_size)

    def test_connected_component__empty_mask(self):
        """Ensure a mask's connected component is correctly calculated
        when the mask is empty.
        """
        expected_size = (37, 43)
        original_mask = pygame.mask.Mask(expected_size)
        original_count = original_mask.count()
        expected_count = 0

        mask = original_mask.connected_component()

        self.assertIsInstance(mask, pygame.mask.Mask)
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

        self.assertIsInstance(mask, pygame.mask.Mask)
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

        self.assertIsInstance(mask, pygame.mask.Mask)
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

        self.assertIsInstance(mask, pygame.mask.Mask)
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
        m = pygame.Mask((10, 10))

        self.assertListEqual(m.connected_components(), [])

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
        self.assertListEqual(comps3, [])

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

        for mask in comps:
            self.assertIsInstance(mask, pygame.mask.Mask)

    def test_get_bounding_rects(self):
        """Ensures get_bounding_rects works correctly."""
        # Create masks with different set point groups. Each group of
        # connected set points will be contained in its own bounding rect.
        # Diagonal points are considered connected.
        mask_data = [] # [((size), ((rect1_pts), ...)), ...]

        # Mask 1:
        #  |0123456789
        # -+----------
        # 0|1100000000
        # 1|1000000000
        # 2|0000000000
        # 3|1001000000
        # 4|0000000000
        # 5|0000000000
        # 6|0000000000
        # 7|0000000000
        # 8|0000000000
        # 9|0000000000
        mask_data.append(((10, 10), # size
                          # Points to set for the 3 bounding rects.
                          (((0, 0), (1, 0), (0, 1)), # rect1
                           ((0, 3),),   # rect2
                           ((3, 3),)))) # rect3

        # Mask 2:
        #  |0123
        # -+----
        # 0|1100
        # 1|1111
        mask_data.append(((4, 2), # size
                          # Points to set for the 1 bounding rect.
                          (((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))

        # Mask 3:
        #  |01234
        # -+-----
        # 0|00100
        # 1|01110
        # 2|00100
        mask_data.append(((5, 3), # size
                           # Points to set for the 1 bounding rect.
                           (((2, 0), (1, 1), (2, 1), (3, 1), (2, 2)),)))

        # Mask 4:
        #  |01234
        # -+-----
        # 0|00010
        # 1|00100
        # 2|01000
        mask_data.append(((5, 3), # size
                          # Points to set for the 1 bounding rect.
                          (((3, 0), (2, 1), (1, 2)),)))

        # Mask 5:
        #  |01234
        # -+-----
        # 0|00011
        # 1|11111
        mask_data.append(((5, 2), # size
                          # Points to set for the 1 bounding rect.
                          (((3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))

        # Mask 6:
        #  |01234
        # -+-----
        # 0|10001
        # 1|00100
        # 2|10001
        mask_data.append(((5, 3), # size
                          # Points to set for the 5 bounding rects.
                          (((0, 0),),   # rect1
                           ((4, 0),),   # rect2
                           ((2, 1),),   # rect3
                           ((0, 2),),   # rect4
                           ((4, 2),)))) # rect5

        for size, rect_point_tuples in mask_data:
            rects = []
            mask = pygame.Mask(size)

            for rect_points in rect_point_tuples:
                rects.append(create_bounding_rect(rect_points))
                for pt in rect_points:
                    mask.set_at(pt)

            expected_rects = sorted(rects, key=tuple)

            rects = mask.get_bounding_rects()

            self.assertListEqual(sorted(mask.get_bounding_rects(), key=tuple),
                                 expected_rects, 'size={}'.format(size))

    def test_zero_mask(self):
        """Ensures masks can be created with zero sizes."""
        for size in ((100, 0), (0, 100), (0, 0)):
            for fill in (True, False):
                msg = 'size={}, fill={}'.format(size, fill)

                mask = pygame.mask.Mask(size, fill=fill)

                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), size, msg)

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
        """Ensures overlap correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)

        for size1, size2 in zero_size_pairs(51, 42):
            msg = 'size1={}, size2={}'.format(size1, size2)
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)

            overlap_pos = mask1.overlap(mask2, offset)

            self.assertIsNone(overlap_pos, msg)

    def test_zero_mask_overlap_area(self):
        """Ensures overlap_area correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        expected_count = 0

        for size1, size2 in zero_size_pairs(41, 52):
            msg = 'size1={}, size2={}'.format(size1, size2)
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)

            overlap_count = mask1.overlap_area(mask2, offset)

            self.assertEqual(overlap_count, expected_count, msg)

    def test_zero_mask_overlap_mask(self):
        """Ensures overlap_mask correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        expected_count = 0

        for size1, size2 in zero_size_pairs(43, 53):
            msg = 'size1={}, size2={}'.format(size1, size2)
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)

            overlap_mask = mask1.overlap_mask(mask2, offset)

            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), size1, msg)

    def test_zero_mask_fill(self):
        """Ensures fill correctly handles zero sized masks."""
        expected_count = 0

        for size in ((100, 0), (0, 100), (0, 0)):
            mask = pygame.mask.Mask(size)

            mask.fill()

            self.assertEqual(mask.count(), expected_count,
                             'size={}'.format(size))

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

            self.assertIsInstance(mask2, pygame.mask.Mask)
            self.assertEqual(mask2.get_size(), (2, 3))

    def test_zero_mask_draw(self):
        """Ensures draw correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)

        for size1, size2 in zero_size_pairs(31, 37):
            msg = 'size1={}, size2={}'.format(size1, size2)
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            expected_count = mask1.count()

            mask1.draw(mask2, offset)

            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), size1, msg)

    def test_zero_mask_erase(self):
        """Ensures erase correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)

        for size1, size2 in zero_size_pairs(29, 23):
            msg = 'size1={}, size2={}'.format(size1, size2)
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            expected_count = mask1.count()

            mask1.erase(mask2, offset)

            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), size1, msg)

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

                self.assertIsInstance(mask, pygame.mask.Mask, msg)
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

                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
                    self.assertIs(mask, output_mask, msg)
                    self.assertEqual(mask.get_size(), output_size, msg)

    def test_zero_mask_connected_component(self):
        """Ensures connected_component correctly handles zero sized masks."""
        expected_count = 0

        for size in ((81, 0), (0, 80), (0, 0)):
            msg = 'size={}'.format(size)
            mask = pygame.mask.Mask(size)

            cc_mask = mask.connected_component()

            self.assertIsInstance(cc_mask, pygame.mask.Mask, msg)
            self.assertEqual(cc_mask.get_size(), size)
            self.assertEqual(cc_mask.count(), expected_count, msg)

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


class SubMask(pygame.mask.Mask):
    """Subclass of the Mask class to help test subclassing."""
    def __init__(self, *args, **kwargs):
        super(SubMask, self).__init__(*args, **kwargs)
        self.test_attribute = True


class MaskSubclassTest(unittest.TestCase):
    """Test subclassed Masks."""
    def test_subclass_mask(self):
        """Ensures the Mask class can be subclassed."""
        mask = SubMask((5, 3), fill=True)

        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertIsInstance(mask, SubMask)
        self.assertTrue(mask.test_attribute)

    def test_subclass_get_size(self):
        """Ensures get_size works for subclassed Masks."""
        expected_size = (2, 3)
        mask = SubMask(expected_size)

        size = mask.get_size()

        self.assertEqual(size, expected_size)

    def test_subclass_get_at(self):
        """Ensures get_at works for subclassed Masks."""
        expected_bit = 1
        mask = SubMask((3, 2), fill=True)

        bit = mask.get_at((0, 0))

        self.assertEqual(bit, expected_bit)

    def test_subclass_set_at(self):
        """Ensures set_at works for subclassed Masks."""
        expected_bit = 1
        expected_count = 1
        pos = (0, 0)
        mask = SubMask(fill=False, size=(4, 2))

        mask.set_at(pos)

        self.assertEqual(mask.get_at(pos), expected_bit)
        self.assertEqual(mask.count(), expected_count)

    def test_subclass_overlap(self):
        """Ensures overlap works for subclassed Masks."""
        expected_pos = (0, 0)
        mask_size = (2, 3)
        masks = (pygame.mask.Mask(fill=True, size=mask_size),
                 SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size),
                     SubMask(mask_size, True))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_pos = mask.overlap(arg_mask, (0, 0))

                self.assertEqual(overlap_pos, expected_pos)

    def test_subclass_overlap_area(self):
        """Ensures overlap_area works for subclassed Masks."""
        mask_size = (3, 2)
        expected_count = mask_size[0] * mask_size[1]
        masks = (pygame.mask.Mask(fill=True, size=mask_size),
                 SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size),
                     SubMask(mask_size, True))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_count = mask.overlap_area(arg_mask, (0, 0))

                self.assertEqual(overlap_count, expected_count)

    def test_subclass_overlap_mask(self):
        """Ensures overlap_mask works for subclassed Masks."""
        expected_size = (4, 5)
        expected_count = expected_size[0] * expected_size[1]
        masks = (pygame.mask.Mask(fill=True, size=expected_size),
                 SubMask(expected_size, True))
        arg_masks = (pygame.mask.Mask(fill=True, size=expected_size),
                     SubMask(expected_size, True))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in masks:
            for arg_mask in arg_masks:
                overlap_mask = mask.overlap_mask(arg_mask, (0, 0))

                self.assertIsInstance(overlap_mask, pygame.mask.Mask)
                self.assertNotIsInstance(overlap_mask, SubMask)
                self.assertEqual(overlap_mask.count(), expected_count)
                self.assertEqual(overlap_mask.get_size(), expected_size)

    def test_subclass_fill(self):
        """Ensures fill works for subclassed Masks."""
        mask_size = (2, 4)
        expected_count = mask_size[0] * mask_size[1]
        mask = SubMask(fill=False, size=mask_size)

        mask.fill()

        self.assertEqual(mask.count(), expected_count)

    def test_subclass_clear(self):
        """Ensures clear works for subclassed Masks."""
        mask_size = (4, 3)
        expected_count = 0
        mask = SubMask(mask_size, True)

        mask.clear()

        self.assertEqual(mask.count(), expected_count)

    def test_subclass_invert(self):
        """Ensures invert works for subclassed Masks."""
        mask_size = (1, 4)
        expected_count = mask_size[0] * mask_size[1]
        mask = SubMask(fill=False, size=mask_size)

        mask.invert()

        self.assertEqual(mask.count(), expected_count)

    def test_subclass_scale(self):
        """Ensures scale works for subclassed Masks."""
        expected_size = (5, 2)
        mask = SubMask((1, 4))

        scaled_mask = mask.scale(expected_size)

        self.assertIsInstance(scaled_mask, pygame.mask.Mask)
        self.assertNotIsInstance(scaled_mask, SubMask)
        self.assertEqual(scaled_mask.get_size(), expected_size)

    def test_subclass_draw(self):
        """Ensures draw works for subclassed Masks."""
        mask_size = (5, 4)
        expected_count = mask_size[0] * mask_size[1]
        arg_masks = (pygame.mask.Mask(fill=True, size=mask_size),
                     SubMask(mask_size, True))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
            for arg_mask in arg_masks:
                mask.clear() # Clear for each test.

                mask.draw(arg_mask, (0, 0))

                self.assertEqual(mask.count(), expected_count)

    def test_subclass_erase(self):
        """Ensures erase works for subclassed Masks."""
        mask_size = (3, 4)
        expected_count = 0
        masks = (pygame.mask.Mask(mask_size, True), SubMask(mask_size, True))
        arg_masks = (pygame.mask.Mask(mask_size, True),
                     SubMask(mask_size, True))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in masks:
            for arg_mask in arg_masks:
                mask.fill() # Fill for each test.

                mask.erase(arg_mask, (0, 0))

                self.assertEqual(mask.count(), expected_count)

    def test_subclass_count(self):
        """Ensures count works for subclassed Masks."""
        mask_size = (5, 2)
        expected_count = mask_size[0] * mask_size[1] - 1
        mask = SubMask(fill=True, size=mask_size)
        mask.set_at((1, 1), 0)

        count = mask.count()

        self.assertEqual(count, expected_count)

    def test_subclass_centroid(self):
        """Ensures centroid works for subclassed Masks."""
        expected_centroid = (0, 0)
        mask_size = (3, 2)
        mask = SubMask((3, 2))

        centroid = mask.centroid()

        self.assertEqual(centroid, expected_centroid)

    def test_subclass_angle(self):
        """Ensures angle works for subclassed Masks."""
        expected_angle = 0.0
        mask = SubMask(size=(5, 4))

        angle = mask.angle()

        self.assertAlmostEqual(angle, expected_angle)

    def test_subclass_outline(self):
        """Ensures outline works for subclassed Masks."""
        expected_outline = []
        mask = SubMask((3, 4))

        outline = mask.outline()

        self.assertListEqual(outline, expected_outline)

    def test_subclass_convolve(self):
        """Ensures convolve works for subclassed Masks."""
        width, height = 7, 5
        mask_size = (width, height)
        expected_count = 0
        expected_size = (max(0, width * 2 - 1), max(0, height * 2 - 1))

        arg_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))
        output_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))

        # Test different combinations of subclassed and non-subclassed Masks.
        for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
            for arg_mask in arg_masks:
                convolve_mask = mask.convolve(arg_mask)

                self.assertIsInstance(convolve_mask, pygame.mask.Mask)
                self.assertNotIsInstance(convolve_mask, SubMask)
                self.assertEqual(convolve_mask.count(), expected_count)
                self.assertEqual(convolve_mask.get_size(), expected_size)

                # Test subclassed masks for the output_mask as well.
                for output_mask in output_masks:
                    convolve_mask = mask.convolve(arg_mask, output_mask)

                    self.assertIsInstance(convolve_mask, pygame.mask.Mask)
                    self.assertEqual(convolve_mask.count(), expected_count)
                    self.assertEqual(convolve_mask.get_size(), mask_size)

                    if isinstance(output_mask, SubMask):
                        self.assertIsInstance(convolve_mask, SubMask)
                    else:
                        self.assertNotIsInstance(convolve_mask, SubMask)

    def test_subclass_connected_component(self):
        """Ensures connected_component works for subclassed Masks."""
        expected_count = 0
        expected_size = (3, 4)
        mask = SubMask(expected_size)

        cc_mask = mask.connected_component()

        self.assertIsInstance(cc_mask, pygame.mask.Mask)
        self.assertNotIsInstance(cc_mask, SubMask)
        self.assertEqual(cc_mask.count(), expected_count)
        self.assertEqual(cc_mask.get_size(), expected_size)

    def test_subclass_connected_components(self):
        """Ensures connected_components works for subclassed Masks."""
        expected_ccs = []
        mask = SubMask((5, 4))

        ccs = mask.connected_components()

        self.assertListEqual(ccs, expected_ccs)

    def test_subclass_get_bounding_rects(self):
        """Ensures get_bounding_rects works for subclassed Masks."""
        expected_bounding_rects = []
        mask = SubMask((3, 2))

        bounding_rects = mask.get_bounding_rects()

        self.assertListEqual(bounding_rects, expected_bounding_rects)


class MaskModuleTest(unittest.TestCase):
    def test_from_surface(self):
        """Ensures from_surface creates a mask with the correct bits set.

        This test checks the masks created by the from_surface function using
        16 and 32 bit surfaces. Each alpha value (0-255) is tested against
        several different threshold values.
        Note: On 16 bit surface the requested alpha value can differ from what
              is actually set. This test uses the value read from the surface.
        """
        threshold_count = 256
        surface_color = [55, 155, 255, 0]
        expected_size = (11, 9)
        all_set_count = expected_size[0] * expected_size[1]
        none_set_count = 0

        for depth in (16, 32):
            surface = pygame.Surface(expected_size, SRCALPHA, depth)

            for alpha in range(threshold_count):
                surface_color[3] = alpha
                surface.fill(surface_color)

                if depth < 32:
                    # On surfaces with depths < 32 the requested alpha can be
                    # different than what gets set. Use the value read from the
                    # surface.
                    alpha = surface.get_at((0, 0))[3]

                # Test the mask created at threshold values low, high and
                # around alpha.
                threshold_test_values = set(
                    [-1, 0, alpha - 1, alpha, alpha + 1, 255, 256])

                for threshold in threshold_test_values:
                    msg = 'depth={}, alpha={}, threshold={}'.format(
                        depth, alpha, threshold)

                    if alpha > threshold:
                        expected_count = all_set_count
                    else:
                        expected_count = none_set_count

                    mask = pygame.mask.from_surface(surface, threshold)

                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
                    self.assertEqual(mask.get_size(), expected_size, msg)
                    self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__different_alphas_32bit(self):
        """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (32 bits surfaces).

        This test checks the masks created by the from_surface function using
        a 32 bit surface. The surface is created with each pixel having a
        different alpha value (0-255). This surface is tested over a range
        of threshold values (0-255).
        """
        offset = (0, 0)
        threshold_count = 256
        surface_color = [10, 20, 30, 0]
        expected_size = (threshold_count, 1)
        expected_mask = pygame.Mask(expected_size, fill=True)
        surface = pygame.Surface(expected_size, SRCALPHA, 32)

        # Give each pixel a different alpha.
        surface.lock()  # Lock for possible speed up.
        for a in range(threshold_count):
            surface_color[3] = a
            surface.set_at((a, 0), surface_color)
        surface.unlock()

        # Test the mask created for each different alpha threshold.
        for threshold in range(threshold_count):
            msg = 'threshold={}'.format(threshold)
            expected_mask.set_at((threshold, 0), 0)
            expected_count = expected_mask.count()

            mask = pygame.mask.from_surface(surface, threshold)

            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.overlap_area(expected_mask, offset),
                             expected_count, msg)

    def test_from_surface__different_alphas_16bit(self):
        """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (16 bit surfaces).

        This test checks the masks created by the from_surface function using
        a 16 bit surface. Each pixel of the surface is set with a different
        alpha value (0-255), but since this is a 16 bit surface the requested
        alpha value can differ from what is actually set. The resulting surface
        will have groups of alpha values which complicates the test as the
        alpha groups will all be set/unset at a given threshold. The setup
        calculates these groups and an expected mask for each. This test data
        is then used to test each alpha grouping over a range of threshold
        values.
        """
        threshold_count = 256
        surface_color = [110, 120, 130, 0]
        expected_size = (threshold_count, 1)
        surface = pygame.Surface(expected_size, SRCALPHA, 16)

        # Give each pixel a different alpha.
        surface.lock()  # Lock for possible speed up.
        for a in range(threshold_count):
            surface_color[3] = a
            surface.set_at((a, 0), surface_color)
        surface.unlock()

        alpha_thresholds = OrderedDict()
        special_thresholds = set()

        # Create the threshold ranges and identify any thresholds that need
        # special handling.
        for threshold in range(threshold_count):
            # On surfaces with depths < 32 the requested alpha can be different
            # than what gets set. Use the value read from the surface.
            alpha = surface.get_at((threshold, 0))[3]

            if alpha not in alpha_thresholds:
                alpha_thresholds[alpha] = [threshold]
            else:
                alpha_thresholds[alpha].append(threshold)

            if threshold < alpha:
                special_thresholds.add(threshold)

        # Use each threshold group to create an expected mask.
        test_data = []  # [(from_threshold, to_threshold, expected_mask), ...]
        offset = (0, 0)
        erase_mask = pygame.Mask(expected_size)
        exp_mask = pygame.Mask(expected_size, fill=True)

        for thresholds in alpha_thresholds.values():
            for threshold in thresholds:
                if threshold in special_thresholds:
                    # Any special thresholds just reuse previous exp_mask.
                    test_data.append((threshold, threshold + 1, exp_mask))
                else:
                    to_threshold = thresholds[-1] + 1

                    # Make the expected mask by erasing the unset bits.
                    for thres in range(to_threshold):
                        erase_mask.set_at((thres, 0), 1)

                    exp_mask = pygame.Mask(expected_size, fill=True)
                    exp_mask.erase(erase_mask, offset)
                    test_data.append((threshold, to_threshold, exp_mask))
                    break

        # All the setup is done. Now test the masks created over the threshold
        # ranges.
        for from_threshold, to_threshold, expected_mask in test_data:
            expected_count = expected_mask.count()

            for threshold in range(from_threshold, to_threshold):
                msg = 'threshold={}'.format(threshold)

                mask = pygame.mask.from_surface(surface, threshold)

                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)
                self.assertEqual(mask.overlap_area(expected_mask, offset),
                                 expected_count, msg)

    def test_from_surface__with_colorkey_mask_cleared(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with the colorkey color so the resulting masks
        are expected to have no bits set.
        """
        colorkeys = ((0, 0, 0), (1, 2, 3), (50, 100, 200), (255, 255, 255))
        expected_size = (7, 11)
        expected_count = 0

        for depth in (8, 16, 24, 32):
            msg = 'depth={}'.format(depth)
            surface = pygame.Surface(expected_size, 0, depth)

            for colorkey in colorkeys:
                surface.set_colorkey(colorkey)
                # With some depths (i.e. 8 and 16) the actual colorkey can be
                # different than what was requested via the set.
                surface.fill(surface.get_colorkey())

                mask = pygame.mask.from_surface(surface)

                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__with_colorkey_mask_filled(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with a color that is not the colorkey color so
        the resulting masks are expected to have all bits set.
        """
        colorkeys = ((0, 0, 0), (1, 2, 3), (10, 100, 200), (255, 255, 255))
        surface_color = (50, 100, 200)
        expected_size = (11, 7)
        expected_count = expected_size[0] * expected_size[1]

        for depth in (8, 16, 24, 32):
            msg = 'depth={}'.format(depth)
            surface = pygame.Surface(expected_size, 0, depth)
            surface.fill(surface_color)

            for colorkey in colorkeys:
                surface.set_colorkey(colorkey)

                mask = pygame.mask.from_surface(surface)

                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__with_colorkey_mask_pattern(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with alternating pixels of colorkey and
        non-colorkey colors, so the resulting masks are expected to have
        alternating bits set.
        """
        def alternate(func, set_value, unset_value, width, height):
            # Helper function to set alternating values.
            setbit = False
            for pos in ((x, y) for x in range(width) for y in range(height)):
                func(pos, set_value if setbit else unset_value)
                setbit = not setbit

        surface_color = (5, 10, 20)
        colorkey = (50, 60, 70)
        expected_size = (11, 2)
        expected_mask = pygame.mask.Mask(expected_size)
        alternate(expected_mask.set_at, 1, 0, *expected_size)
        expected_count = expected_mask.count()
        offset = (0, 0)

        for depth in (8, 16, 24, 32):
            msg = 'depth={}'.format(depth)
            surface = pygame.Surface(expected_size, 0, depth)
            # Fill the surface with alternating colors.
            alternate(surface.set_at, surface_color, colorkey, *expected_size)
            surface.set_colorkey(colorkey)

            mask = pygame.mask.from_surface(surface)

            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.overlap_area(expected_mask, offset),
                             expected_count, msg)

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

            self.assertIsInstance(mask, pygame.mask.Mask)
            self.assertEqual(mask.count(), 100)
            self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((40,40,10,10))])

    def test_zero_size_from_surface(self):
        """Ensures from_surface can create masks from zero sized surfaces."""
        for size in ((100, 0), (0, 100), (0, 0)):
            mask = pygame.mask.from_surface(pygame.Surface(size))

            self.assertIsInstance(mask, pygame.mask.MaskType,
                                  'size={}'.format(size))
            self.assertEqual(mask.get_size(), size)

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

                self.assertIsInstance(mask, pygame.mask.Mask)
                self.assertEqual(mask.count(), 0)

                rects = mask.get_bounding_rects()
                self.assertEqual(rects, [])

if __name__ == '__main__':
    unittest.main()
