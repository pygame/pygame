import os

import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
    example_path,
    AssertRaisesRegexMixin,
    SurfaceSubclass,
)

try:
    from pygame.tests.test_utils.arrinter import *
except (ImportError, NameError):
    pass

import pygame
from pygame.locals import *
from pygame.compat import xrange_, as_bytes, as_unicode
from pygame.bufferproxy import BufferProxy

import platform
import gc
import weakref
import ctypes

IS_PYPY = "PyPy" == platform.python_implementation()


def intify(i):
    """If i is a long, cast to an int while preserving the bits"""
    if 0x80000000 & i:
        return int((0xFFFFFFFF & i))
    return i


def longify(i):
    """If i is an int, cast to a long while preserving the bits"""
    if i < 0:
        return 0xFFFFFFFF & i
    return long(i)


class SurfaceTypeTest(AssertRaisesRegexMixin, unittest.TestCase):
    def test_surface__pixel_format_as_surface_subclass(self):
        """Ensure a subclassed surface can be used for pixel format
        when creating a new surface."""
        expected_depth = 16
        expected_flags = SRCALPHA
        expected_size = (13, 37)
        depth_surface = SurfaceSubclass((11, 21), expected_flags, expected_depth)

        surface = pygame.Surface(expected_size, 0, depth_surface)

        self.assertIsNot(surface, depth_surface)
        self.assertIsInstance(surface, pygame.Surface)
        self.assertNotIsInstance(surface, SurfaceSubclass)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_flags(), expected_flags)
        self.assertEqual(surface.get_bitsize(), expected_depth)

    def test_set_clip(self):
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
        surf = pygame.Surface((70, 70), 0, 32)
        self.assertEqual(repr(surf), "<Surface(70x70x32 SW)>")

    def test_keyword_arguments(self):
        surf = pygame.Surface((70, 70), flags=SRCALPHA, depth=32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)
        self.assertEqual(surf.get_bitsize(), 32)

        # sanity check to make sure the check below is valid
        surf_16 = pygame.Surface((70, 70), 0, 16)
        self.assertEqual(surf_16.get_bytesize(), 2)

        # try again with an argument list
        surf_16 = pygame.Surface((70, 70), depth=16)
        self.assertEqual(surf_16.get_bytesize(), 2)

    def test_set_at(self):

        # 24bit surfaces
        s = pygame.Surface((100, 100), 0, 24)
        s.fill((0, 0, 0))

        # set it with a tuple.
        s.set_at((0, 0), (10, 10, 10, 255))
        r = s.get_at((0, 0))
        self.assertIsInstance(r, pygame.Color)
        self.assertEqual(r, (10, 10, 10, 255))

        # try setting a color with a single integer.
        s.fill((0, 0, 0, 255))
        s.set_at((10, 1), 0x0000FF)
        r = s.get_at((10, 1))
        self.assertEqual(r, (0, 0, 255, 255))

    def test_SRCALPHA(self):
        # has the flag been passed in ok?
        surf = pygame.Surface((70, 70), SRCALPHA, 32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)

        # 24bit surfaces can not have SRCALPHA.
        self.assertRaises(ValueError, pygame.Surface, (100, 100), pygame.SRCALPHA, 24)

        # if we have a 32 bit surface, the SRCALPHA should have worked too.
        surf2 = pygame.Surface((70, 70), SRCALPHA)
        if surf2.get_bitsize() == 32:
            self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)

    def test_masks(self):
        def make_surf(bpp, flags, masks):
            pygame.Surface((10, 10), flags, bpp, masks)

        # With some masks SDL_CreateRGBSurface does not work properly.
        masks = (0xFF000000, 0xFF0000, 0xFF00, 0)
        self.assertEqual(make_surf(32, 0, masks), None)
        # For 24 and 32 bit surfaces Pygame assumes no losses.
        masks = (0x7F0000, 0xFF00, 0xFF, 0)
        self.assertRaises(ValueError, make_surf, 24, 0, masks)
        self.assertRaises(ValueError, make_surf, 32, 0, masks)
        # What contiguous bits in a mask.
        masks = (0x6F0000, 0xFF00, 0xFF, 0)
        self.assertRaises(ValueError, make_surf, 32, 0, masks)

    def test_get_bounding_rect(self):
        surf = pygame.Surface((70, 70), SRCALPHA, 32)
        surf.fill((0, 0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30, 30), (255, 255, 255, 1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((29, 29), (255, 255, 255, 1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 29)
        self.assertEqual(bound_rect.top, 29)
        self.assertEqual(bound_rect.width, 2)
        self.assertEqual(bound_rect.height, 2)

        surf = pygame.Surface((70, 70), 0, 24)
        surf.fill((0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, surf.get_width())
        self.assertEqual(bound_rect.height, surf.get_height())

        surf.set_colorkey((0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30, 30), (255, 255, 255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((60, 60), (255, 255, 255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 31)
        self.assertEqual(bound_rect.height, 31)

        # Issue #180
        pygame.display.init()
        try:
            surf = pygame.Surface((4, 1), 0, 8)
            surf.fill((255, 255, 255))
            surf.get_bounding_rect()  # Segfault.
        finally:
            pygame.display.quit()

    def test_copy(self):
        """Ensure a surface can be copied."""
        color = (25, 25, 25, 25)
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s1.fill(color)

        s2 = s1.copy()

        s1rect = s1.get_rect()
        s2rect = s2.get_rect()

        self.assertEqual(s1rect.size, s2rect.size)
        self.assertEqual(s2.get_at((10, 10)), color)

    def test_fill(self):
        """Ensure a surface can be filled."""
        color = (25, 25, 25, 25)
        fill_rect = pygame.Rect(0, 0, 16, 16)
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s1.fill(color, fill_rect)

        for pt in test_utils.rect_area_pts(fill_rect):
            self.assertEqual(s1.get_at(pt), color)

        for pt in test_utils.rect_outer_bounds(fill_rect):
            self.assertNotEqual(s1.get_at(pt), color)

    def test_fill_rle(self):
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)
        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255,0,255), pygame.RLEACCEL)
        s0.blit(s1, (0,0))
        s1.fill(color)
        self.assertTrue(s1.get_flags()&pygame.RLEACCEL)

    @unittest.expectedFailure
    def test_copy_rle(self):
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)
        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255,0,255), pygame.RLEACCEL)
        s0.blit(s1, (0,0))
        s1.copy()
        self.assertTrue(s1.get_flags()&pygame.RLEACCEL)

    def test_subsurface_rle(self):
        """Ensure an RLE sub-surface works independently of its parent."""
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)
        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255,0,255), pygame.RLEACCEL)
        s1.fill(color)
        s2 = s1.subsurface(sub_rect)
        s2.fill(color2)
        s0.blit(s1, (0,0))
        self.assertTrue(s1.get_flags()&pygame.RLEACCEL)
        self.assertTrue(not s2.get_flags()&pygame.RLEACCEL)

    @unittest.expectedFailure
    def test_subsurface_rle2(self):
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)

        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255,0,255), pygame.RLEACCEL)
        s1.fill(color)
        s2 = s1.subsurface(sub_rect)
        s2.fill(color2)
        s0.blit(s2, (0,0))
        self.assertTrue(s1.get_flags()&pygame.RLEACCEL)
        self.assertTrue(not s2.get_flags()&pygame.RLEACCEL)

    def test_fill_negative_coordinates(self):

        # negative coordinates should be clipped by fill, and not draw outside the surface.
        color = (25, 25, 25, 25)
        color2 = (20, 20, 20, 25)
        fill_rect = pygame.Rect(-10, -10, 16, 16)

        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        r1 = s1.fill(color, fill_rect)
        c = s1.get_at((0, 0))
        self.assertEqual(c, color)

        # make subsurface in the middle to test it doesn't over write.
        s2 = s1.subsurface((5, 5, 5, 5))
        r2 = s2.fill(color2, (-3, -3, 5, 5))
        c2 = s1.get_at((4, 4))
        self.assertEqual(c, color)

        # rect returns the area we actually fill.
        r3 = s2.fill(color2, (-30, -30, 5, 5))
        # since we are using negative coords, it should be an zero sized rect.
        self.assertEqual(tuple(r3), (0, 0, 0, 0))

    def test_fill_keyword_args(self):
        """Ensure fill() accepts keyword arguments."""
        color = (1, 2, 3, 255)
        area = (1, 1, 2, 2)
        s1 = pygame.Surface((4, 4), 0, 32)
        s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)

        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)

    ########################################################################

    def test_get_alpha(self):
        """Ensure a surface's alpha value can be retrieved."""
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)

        self.assertEqual(s1.get_alpha(), 255)

        for alpha in (0, 32, 127, 255):
            s1.set_alpha(alpha)
            for t in range(4):
                s1.set_alpha(s1.get_alpha())

            self.assertEqual(s1.get_alpha(), alpha)

    ########################################################################

    def test_get_bytesize(self):
        """Ensure a surface's bit and byte sizes can be retrieved."""
        depth = 32
        depth_bytes = 4
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, depth)

        self.assertEqual(s1.get_bytesize(), depth_bytes)
        self.assertEqual(s1.get_bitsize(), depth)

    ########################################################################

    def test_get_flags(self):
        """Ensure a surface's flags can be retrieved."""
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)

        self.assertEqual(s1.get_flags(), pygame.SRCALPHA)

    ########################################################################

    def test_get_parent(self):
        """Ensure a surface's parent can be retrieved."""
        parent = pygame.Surface((16, 16))
        child = parent.subsurface((0, 0, 5, 5))

        self.assertIs(child.get_parent(), parent)

    ########################################################################

    def test_get_rect(self):
        """Ensure a surface's rect can be retrieved."""
        size = (16, 16)
        surf = pygame.Surface(size)
        rect = surf.get_rect()

        self.assertEqual(rect.size, size)

    ########################################################################

    def test_get_width__size_and_height(self):
        """Ensure a surface's size, width and height can be retrieved."""
        for w in xrange_(0, 255, 32):
            for h in xrange_(0, 127, 15):
                s = pygame.Surface((w, h))
                self.assertEqual(s.get_width(), w)
                self.assertEqual(s.get_height(), h)
                self.assertEqual(s.get_size(), (w, h))

    def test_get_view(self):
        """Ensure a buffer view of the surface's pixels can be retrieved."""
        # Check that BufferProxys are returned when array depth is supported,
        # ValueErrors returned otherwise.
        Error = ValueError
        s = pygame.Surface((5, 7), 0, 8)
        v2 = s.get_view("2")

        self.assertRaises(Error, s.get_view, "0")
        self.assertRaises(Error, s.get_view, "1")
        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, "3")

        s = pygame.Surface((8, 7), 0, 8)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view("0")
        v1 = s.get_view("1")

        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)

        s = pygame.Surface((5, 7), 0, 16)
        v2 = s.get_view("2")

        self.assertRaises(Error, s.get_view, "0")
        self.assertRaises(Error, s.get_view, "1")
        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, "3")

        s = pygame.Surface((8, 7), 0, 16)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view("0")
        v1 = s.get_view("1")

        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)

        s = pygame.Surface((5, 7), pygame.SRCALPHA, 16)
        v2 = s.get_view("2")

        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, "3")

        s = pygame.Surface((5, 7), 0, 24)
        v2 = s.get_view("2")
        v3 = s.get_view("3")

        self.assertRaises(Error, s.get_view, "0")
        self.assertRaises(Error, s.get_view, "1")
        self.assertIsInstance(v2, BufferProxy)
        self.assertIsInstance(v3, BufferProxy)

        s = pygame.Surface((8, 7), 0, 24)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view("0")
        v1 = s.get_view("1")

        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)

        s = pygame.Surface((5, 7), 0, 32)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view("0")
        v1 = s.get_view("1")
        v2 = s.get_view("2")
        v3 = s.get_view("3")

        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)
        self.assertIsInstance(v2, BufferProxy)
        self.assertIsInstance(v3, BufferProxy)

        s2 = s.subsurface((0, 0, 4, 7))

        self.assertRaises(Error, s2.get_view, "0")
        self.assertRaises(Error, s2.get_view, "1")

        s2 = None
        s = pygame.Surface((5, 7), pygame.SRCALPHA, 32)

        for kind in ("2", "3", "a", "A", "r", "R", "g", "G", "b", "B"):
            self.assertIsInstance(s.get_view(kind), BufferProxy)

        # Check default argument value: '2'
        s = pygame.Surface((2, 4), 0, 32)
        v = s.get_view()
        if not IS_PYPY:
            ai = ArrayInterface(v)
            self.assertEqual(ai.nd, 2)

        # Check locking.
        s = pygame.Surface((2, 4), 0, 32)

        self.assertFalse(s.get_locked())

        v = s.get_view("2")

        self.assertFalse(s.get_locked())

        c = v.__array_interface__

        self.assertTrue(s.get_locked())

        c = None
        gc.collect()

        self.assertTrue(s.get_locked())

        v = None
        gc.collect()

        self.assertFalse(s.get_locked())

        # Check invalid view kind values.
        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        self.assertRaises(TypeError, s.get_view, "")
        self.assertRaises(TypeError, s.get_view, "9")
        self.assertRaises(TypeError, s.get_view, "RGBA")
        self.assertRaises(TypeError, s.get_view, 2)

        # Both unicode and bytes strings are allowed for kind.
        s = pygame.Surface((2, 4), 0, 32)
        s.get_view(as_unicode("2"))
        s.get_view(as_bytes("2"))

        # Garbage collection
        s = pygame.Surface((2, 4), 0, 32)
        weak_s = weakref.ref(s)
        v = s.get_view("3")
        weak_v = weakref.ref(v)
        gc.collect()
        self.assertTrue(weak_s() is s)
        self.assertTrue(weak_v() is v)
        del v
        gc.collect()
        self.assertTrue(weak_s() is s)
        self.assertTrue(weak_v() is None)
        del s
        gc.collect()
        self.assertTrue(weak_s() is None)

    def test_get_buffer(self):
        # Check that get_buffer works for all pixel sizes and for a subsurface.

        # Check for all pixel sizes
        for bitsize in [8, 16, 24, 32]:
            s = pygame.Surface((5, 7), 0, bitsize)
            length = s.get_pitch() * s.get_height()
            v = s.get_buffer()

            self.assertIsInstance(v, BufferProxy)
            self.assertEqual(v.length, length)
            self.assertEqual(repr(v), "<BufferProxy(" + str(length) + ")>")

        # Check for a subsurface (not contiguous)
        s = pygame.Surface((7, 10), 0, 32)
        s2 = s.subsurface((1, 2, 5, 7))
        length = s2.get_pitch() * s2.get_height()
        v = s2.get_buffer()

        self.assertIsInstance(v, BufferProxy)
        self.assertEqual(v.length, length)

        # Check locking.
        s = pygame.Surface((2, 4), 0, 32)
        v = s.get_buffer()
        self.assertTrue(s.get_locked())
        v = None
        gc.collect()
        self.assertFalse(s.get_locked())

    OLDBUF = hasattr(pygame.bufferproxy, "get_segcount")

    @unittest.skipIf(not OLDBUF, "old buffer not available")
    def test_get_buffer_oldbuf(self):
        from pygame.bufferproxy import get_segcount, get_write_buffer

        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        v = s.get_buffer()
        segcount, buflen = get_segcount(v)
        self.assertEqual(segcount, 1)
        self.assertEqual(buflen, s.get_pitch() * s.get_height())
        seglen, segaddr = get_write_buffer(v, 0)
        self.assertEqual(segaddr, s._pixels_address)
        self.assertEqual(seglen, buflen)

    @unittest.skipIf(not OLDBUF, "old buffer not available")
    def test_get_view_oldbuf(self):
        from pygame.bufferproxy import get_segcount, get_write_buffer

        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        v = s.get_view("1")
        segcount, buflen = get_segcount(v)
        self.assertEqual(segcount, 8)
        self.assertEqual(buflen, s.get_pitch() * s.get_height())
        seglen, segaddr = get_write_buffer(v, 7)
        self.assertEqual(segaddr, s._pixels_address + s.get_bytesize() * 7)
        self.assertEqual(seglen, s.get_bytesize())

    def test_set_colorkey(self):

        # __doc__ (as of 2008-06-25) for pygame.surface.Surface.set_colorkey:

        # Surface.set_colorkey(Color, flags=0): return None
        # Surface.set_colorkey(None): return None
        # Set the transparent colorkey

        s = pygame.Surface((16, 16), pygame.SRCALPHA, 32)

        colorkeys = ((20, 189, 20, 255), (128, 50, 50, 255), (23, 21, 255, 255))

        for colorkey in colorkeys:
            s.set_colorkey(colorkey)

            for t in range(4):
                s.set_colorkey(s.get_colorkey())

            self.assertEqual(s.get_colorkey(), colorkey)

    def test_set_masks(self):
        s = pygame.Surface((32, 32))
        r, g, b, a = s.get_masks()
        if pygame.get_sdl_version()[0] == 1:
            s.set_masks((b, g, r, a))
            r2, g2, b2, a2 = s.get_masks()
            self.assertEqual((r, g, b, a), (b2, g2, r2, a2))
        else:
            self.assertRaises(TypeError, s.set_masks, (b, g, r, a))

    def test_set_shifts(self):
        s = pygame.Surface((32, 32))
        r, g, b, a = s.get_shifts()
        if pygame.get_sdl_version()[0] == 1:
            s.set_shifts((b, g, r, a))
            r2, g2, b2, a2 = s.get_shifts()
            self.assertEqual((r, g, b, a), (b2, g2, r2, a2))
        else:
            self.assertRaises(TypeError, s.set_shifts, (b, g, r, a))

    def test_blit_keyword_args(self):
        color = (1, 2, 3, 255)
        s1 = pygame.Surface((4, 4), 0, 32)
        s2 = pygame.Surface((2, 2), 0, 32)
        s2.fill((1, 2, 3))
        s1.blit(special_flags=BLEND_ADD, source=s2, dest=(1, 1), area=s2.get_rect())
        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)

    def todo_test_blit(self):
        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.blit:

        # Surface.blit(source, dest, area=None, special_flags = 0): return Rect
        # draw one image onto another
        #
        # Draws a source Surface onto this Surface. The draw can be positioned
        # with the dest argument. Dest can either be pair of coordinates
        # representing the upper left corner of the source. A Rect can also be
        # passed as the destination and the topleft corner of the rectangle
        # will be used as the position for the blit. The size of the
        # destination rectangle does not effect the blit.
        #
        # An optional area rectangle can be passed as well. This represents a
        # smaller portion of the source Surface to draw.
        #
        # An optional special flags is for passing in new in 1.8.0: BLEND_ADD,
        # BLEND_SUB, BLEND_MULT, BLEND_MIN, BLEND_MAX new in 1.8.1:
        # BLEND_RGBA_ADD, BLEND_RGBA_SUB, BLEND_RGBA_MULT, BLEND_RGBA_MIN,
        # BLEND_RGBA_MAX BLEND_RGB_ADD, BLEND_RGB_SUB, BLEND_RGB_MULT,
        # BLEND_RGB_MIN, BLEND_RGB_MAX With other special blitting flags
        # perhaps added in the future.
        #
        # The return rectangle is the area of the affected pixels, excluding
        # any pixels outside the destination Surface, or outside the clipping
        # area.
        #
        # Pixel alphas will be ignored when blitting to an 8 bit Surface.
        # special_flags new in pygame 1.8.

        self.fail()

    def test_blit__SRCALPHA_opaque_source(self):
        src = pygame.Surface((256, 256), SRCALPHA, 32)
        dst = src.copy()

        for i, j in test_utils.rect_area_pts(src.get_rect()):
            dst.set_at((i, j), (i, 0, 0, j))
            src.set_at((i, j), (0, i, 0, 255))

        dst.blit(src, (0, 0))

        for pt in test_utils.rect_area_pts(src.get_rect()):
            self.assertEqual(dst.get_at(pt)[1], src.get_at(pt)[1])

    def todo_test_blit__blit_to_self(self):  # TODO
        src = pygame.Surface((256, 256), SRCALPHA, 32)
        rect = src.get_rect()

        for pt, color in test_utils.gradient(rect.width, rect.height):
            src.set_at(pt, color)

        src.blit(src, (0, 0))

    def todo_test_blit__SRCALPHA_to_SRCALPHA_non_zero(self):  # TODO
        # " There is no unit test for blitting a SRCALPHA source with non-zero
        #   alpha to a SRCALPHA destination with non-zero alpha " LL

        w, h = size = 32, 32

        s = pygame.Surface(size, pygame.SRCALPHA, 32)
        s2 = s.copy()

        s.fill((32, 32, 32, 111))
        s2.fill((32, 32, 32, 31))

        s.blit(s2, (0, 0))

        # TODO:
        # what is the correct behaviour ?? should it blend? what algorithm?

        self.assertEqual(s.get_at((0, 0)), (32, 32, 32, 31))

    def test_blit__SRCALPHA32_to_8(self):
        # Bug: fatal
        # SDL_DisplayConvert segfaults when video is uninitialized.
        target = pygame.Surface((11, 8), 0, 8)
        color = target.get_palette_at(2)
        source = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        source.set_at((0, 0), color)
        target.blit(source, (0, 0))

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'requires a non-"dummy" SDL_VIDEODRIVER',
    )
    def test_image_convert_bug_131(self):
        # Bitbucket bug #131: Unable to Surface.convert(32) some 1-bit images.
        # https://bitbucket.org/pygame/pygame/issue/131/unable-to-surfaceconvert-32-some-1-bit

        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))

            im = pygame.image.load(example_path(os.path.join("data", "city.png")))
            im2 = pygame.image.load(example_path(os.path.join("data", "brick.png")))

            self.assertEqual(im.get_palette(), ((0, 0, 0, 255), (255, 255, 255, 255)))
            self.assertEqual(im2.get_palette(), ((0, 0, 0, 255), (0, 0, 0, 255)))

            self.assertEqual(repr(im.convert(32)), "<Surface(24x24x32 SW)>")
            self.assertEqual(repr(im2.convert(32)), "<Surface(469x137x32 SW)>")

            # Ensure a palette format to palette format works.
            im3 = im.convert(8)
            self.assertEqual(repr(im3), "<Surface(24x24x8 SW)>")
            self.assertEqual(im3.get_palette(), im.get_palette())

        finally:
            pygame.display.quit()

    def test_convert_init(self):
        """ Ensure initialization exceptions are raised
            for surf.convert()."""
        pygame.display.quit()
        surf = pygame.Surface((1, 1))

        self.assertRaisesRegex(pygame.error, "display initialized", surf.convert)

        pygame.display.init()
        try:
            if os.environ.get("SDL_VIDEODRIVER") != "dummy":
                try:
                    surf.convert(32)
                    surf.convert(pygame.Surface((1, 1)))
                except pygame.error:
                    self.fail("convert() should not raise an exception here.")

            self.assertRaisesRegex(pygame.error, "No video mode", surf.convert)

            pygame.display.set_mode((640, 480))
            try:
                surf.convert()
            except pygame.error:
                self.fail("convert() should not raise an exception here.")
        finally:
            pygame.display.quit()

    def test_convert_alpha_init(self):
        """ Ensure initialization exceptions are raised
            for surf.convert_alpha()."""
        pygame.display.quit()
        surf = pygame.Surface((1, 1))

        self.assertRaisesRegex(pygame.error, "display initialized", surf.convert_alpha)

        pygame.display.init()
        try:
            self.assertRaisesRegex(pygame.error, "No video mode", surf.convert_alpha)

            pygame.display.set_mode((640, 480))
            try:
                surf.convert_alpha()
            except pygame.error:
                self.fail("convert_alpha() should not raise an exception here.")
        finally:
            pygame.display.quit()

    def test_convert_alpha_SRCALPHA(self):
        """ Ensure that the surface returned by surf.convert_alpha()
            has alpha blending enabled"""
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))

            s1 = pygame.Surface((100, 100), 0, 32)
            # s2=pygame.Surface((100,100), pygame.SRCALPHA, 32)
            s1_alpha = s1.convert_alpha()
            self.assertEqual(s1_alpha.get_flags() & SRCALPHA, SRCALPHA)
            self.assertEqual(s1_alpha.get_alpha(), 255)
        finally:
            pygame.display.quit()

    @unittest.skip("causes failures in other tests if run, so skip")
    def test_src_alpha_issue_1289(self):
        """ blit should be white.
        """
        surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf1.fill((255, 255, 255, 100))

        surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 0))
        surf2.blit(surf1, (0, 0))

        self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
        self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 100))

    def todo_test_convert(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.convert:

        # Surface.convert(Surface): return Surface
        # Surface.convert(depth, flags=0): return Surface
        # Surface.convert(masks, flags=0): return Surface
        # Surface.convert(): return Surface
        # change the pixel format of an image
        #
        # Creates a new copy of the Surface with the pixel format changed. The
        # new pixel format can be determined from another existing Surface.
        # Otherwise depth, flags, and masks arguments can be used, similar to
        # the pygame.Surface() call.
        #
        # If no arguments are passed the new Surface will have the same pixel
        # format as the display Surface. This is always the fastest format for
        # blitting. It is a good idea to convert all Surfaces before they are
        # blitted many times.
        #
        # The converted Surface will have no pixel alphas. They will be
        # stripped if the original had them. See Surface.convert_alpha() for
        # preserving or creating per-pixel alphas.
        #

        self.fail()

    def test_convert__pixel_format_as_surface_subclass(self):
        """Ensure convert accepts a Surface subclass argument."""
        expected_size = (23, 17)
        convert_surface = SurfaceSubclass(expected_size, 0, 32)
        depth_surface = SurfaceSubclass((31, 61), 0, 32)

        pygame.display.init()
        try:
            surface = convert_surface.convert(depth_surface)

            self.assertIsNot(surface, depth_surface)
            self.assertIsNot(surface, convert_surface)
            self.assertIsInstance(surface, pygame.Surface)
            self.assertIsInstance(surface, SurfaceSubclass)
            self.assertEqual(surface.get_size(), expected_size)
        finally:
            pygame.display.quit()

    def todo_test_convert_alpha(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.convert_alpha:

        # Surface.convert_alpha(Surface): return Surface
        # Surface.convert_alpha(): return Surface
        # change the pixel format of an image including per pixel alphas
        #
        # Creates a new copy of the surface with the desired pixel format. The
        # new surface will be in a format suited for quick blitting to the
        # given format with per pixel alpha. If no surface is given, the new
        # surface will be optimized for blitting to the current display.
        #
        # Unlike the Surface.convert() method, the pixel format for the new
        # image will not be exactly the same as the requested source, but it
        # will be optimized for fast alpha blitting to the destination.
        #

        self.fail()

    def test_convert_alpha__pixel_format_as_surface_subclass(self):
        """Ensure convert_alpha accepts a Surface subclass argument."""
        expected_size = (23, 17)
        convert_surface = SurfaceSubclass(expected_size, SRCALPHA, 32)
        depth_surface = SurfaceSubclass((31, 57), SRCALPHA, 32)

        pygame.display.init()
        try:
            pygame.display.set_mode((60, 60))

            # This is accepted as an argument, but its values are ignored.
            # See issue #599.
            surface = convert_surface.convert_alpha(depth_surface)

            self.assertIsNot(surface, depth_surface)
            self.assertIsNot(surface, convert_surface)
            self.assertIsInstance(surface, pygame.Surface)
            self.assertIsInstance(surface, SurfaceSubclass)
            self.assertEqual(surface.get_size(), expected_size)
        finally:
            pygame.display.quit()

    def todo_test_get_abs_offset(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_abs_offset:

        # Surface.get_abs_offset(): return (x, y)
        # find the absolute position of a child subsurface inside its top level parent
        #
        # Get the offset position of a child subsurface inside of its top
        # level parent Surface. If the Surface is not a subsurface this will
        # return (0, 0).
        #

        self.fail()

    def todo_test_get_abs_parent(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_abs_parent:

        # Surface.get_abs_parent(): return Surface
        # find the top level parent of a subsurface
        #
        # Returns the parent Surface of a subsurface. If this is not a
        # subsurface then this surface will be returned.
        #

        self.fail()

    def test_get_at(self):
        surf = pygame.Surface((2, 2), 0, 24)
        c00 = pygame.Color(1, 2, 3)
        c01 = pygame.Color(5, 10, 15)
        c10 = pygame.Color(100, 50, 0)
        c11 = pygame.Color(4, 5, 6)
        surf.set_at((0, 0), c00)
        surf.set_at((0, 1), c01)
        surf.set_at((1, 0), c10)
        surf.set_at((1, 1), c11)
        c = surf.get_at((0, 0))
        self.assertIsInstance(c, pygame.Color)
        self.assertEqual(c, c00)
        self.assertEqual(surf.get_at((0, 1)), c01)
        self.assertEqual(surf.get_at((1, 0)), c10)
        self.assertEqual(surf.get_at((1, 1)), c11)
        for p in [(-1, 0), (0, -1), (2, 0), (0, 2)]:
            self.assertRaises(IndexError, surf.get_at, p)

    def test_get_at_mapped(self):
        color = pygame.Color(10, 20, 30)
        for bitsize in [8, 16, 24, 32]:
            surf = pygame.Surface((2, 2), 0, bitsize)
            surf.fill(color)
            pixel = surf.get_at_mapped((0, 0))
            self.assertEqual(
                pixel,
                surf.map_rgb(color),
                "%i != %i, bitsize: %i" % (pixel, surf.map_rgb(color), bitsize),
            )

    def todo_test_get_bitsize(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_bitsize:

        # Surface.get_bitsize(): return int
        # get the bit depth of the Surface pixel format
        #
        # Returns the number of bits used to represent each pixel. This value
        # may not exactly fill the number of bytes used per pixel. For example
        # a 15 bit Surface still requires a full 2 bytes.
        #

        self.fail()

    def todo_test_get_clip(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_clip:

        # Surface.get_clip(): return Rect
        # get the current clipping area of the Surface
        #
        # Return a rectangle of the current clipping area. The Surface will
        # always return a valid rectangle that will never be outside the
        # bounds of the image. If the Surface has had None set for the
        # clipping area, the Surface will return a rectangle with the full
        # area of the Surface.
        #

        self.fail()

    def todo_test_get_colorkey(self):
        surf = pygame.surface((2, 2), 0, 24)
        self.assertIsNone(surf.get_colorykey())
        colorkey = pygame.Color(20, 40, 60)
        surf.set_colorkey(colorkey)
        ck = surf.get_colorkey()
        self.assertIsInstance(ck, pygame.Color)
        self.assertEqual(ck, colorkey)

    def todo_test_get_height(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_height:

        # Surface.get_height(): return height
        # get the height of the Surface
        #
        # Return the height of the Surface in pixels.

        self.fail()

    def todo_test_get_locked(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_locked:

        # Surface.get_locked(): return bool
        # test if the Surface is current locked
        #
        # Returns True when the Surface is locked. It doesn't matter how many
        # times the Surface is locked.
        #

        self.fail()

    def todo_test_get_locks(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_locks:

        # Surface.get_locks(): return tuple
        # Gets the locks for the Surface
        #
        # Returns the currently existing locks for the Surface.

        self.fail()

    def todo_test_get_losses(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_losses:

        # Surface.get_losses(): return (R, G, B, A)
        # the significant bits used to convert between a color and a mapped integer
        #
        # Return the least significant number of bits stripped from each color
        # in a mapped integer.
        #
        # This value is not needed for normal Pygame usage.

        self.fail()

    def todo_test_get_masks(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_masks:

        # Surface.get_masks(): return (R, G, B, A)
        # the bitmasks needed to convert between a color and a mapped integer
        #
        # Returns the bitmasks used to isolate each color in a mapped integer.
        # This value is not needed for normal Pygame usage.

        self.fail()

    def todo_test_get_offset(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_offset:

        # Surface.get_offset(): return (x, y)
        # find the position of a child subsurface inside a parent
        #
        # Get the offset position of a child subsurface inside of a parent. If
        # the Surface is not a subsurface this will return (0, 0).
        #

        self.fail()

    def test_get_palette(self):
        pygame.display.init()
        try:
            palette = [Color(i, i, i) for i in range(256)]
            pygame.display.set_mode((100, 50))
            surf = pygame.Surface((2, 2), 0, 8)
            surf.set_palette(palette)
            palette2 = surf.get_palette()
            r, g, b = palette2[0]

            self.assertEqual(len(palette2), len(palette))
            for c2, c in zip(palette2, palette):
                self.assertEqual(c2, c)
            for c in palette2:
                self.assertIsInstance(c, pygame.Color)
        finally:
            pygame.display.quit()

    def test_get_palette_at(self):
        # See also test_get_palette
        pygame.display.init()
        try:
            pygame.display.set_mode((100, 50))
            surf = pygame.Surface((2, 2), 0, 8)
            color = pygame.Color(1, 2, 3, 255)
            surf.set_palette_at(0, color)
            color2 = surf.get_palette_at(0)
            self.assertIsInstance(color2, pygame.Color)
            self.assertEqual(color2, color)
            self.assertRaises(IndexError, surf.get_palette_at, -1)
            self.assertRaises(IndexError, surf.get_palette_at, 256)
        finally:
            pygame.display.quit()

    def todo_test_get_pitch(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_pitch:

        # Surface.get_pitch(): return int
        # get the number of bytes used per Surface row
        #
        # Return the number of bytes separating each row in the Surface.
        # Surfaces in video memory are not always linearly packed. Subsurfaces
        # will also have a larger pitch than their real width.
        #
        # This value is not needed for normal Pygame usage.

        self.fail()

    def todo_test_get_shifts(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_shifts:

        # Surface.get_shifts(): return (R, G, B, A)
        # the bit shifts needed to convert between a color and a mapped integer
        #
        # Returns the pixel shifts need to convert between each color and a
        # mapped integer.
        #
        # This value is not needed for normal Pygame usage.

        self.fail()

    def todo_test_get_size(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.get_size:

        # Surface.get_size(): return (width, height)
        # get the dimensions of the Surface
        #
        # Return the width and height of the Surface in pixels.

        self.fail()

    def todo_test_lock(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.lock:

        # Surface.lock(): return None
        # lock the Surface memory for pixel access
        #
        # Lock the pixel data of a Surface for access. On accelerated
        # Surfaces, the pixel data may be stored in volatile video memory or
        # nonlinear compressed forms. When a Surface is locked the pixel
        # memory becomes available to access by regular software. Code that
        # reads or writes pixel values will need the Surface to be locked.
        #
        # Surfaces should not remain locked for more than necessary. A locked
        # Surface can often not be displayed or managed by Pygame.
        #
        # Not all Surfaces require locking. The Surface.mustlock() method can
        # determine if it is actually required. There is no performance
        # penalty for locking and unlocking a Surface that does not need it.
        #
        # All pygame functions will automatically lock and unlock the Surface
        # data as needed. If a section of code is going to make calls that
        # will repeatedly lock and unlock the Surface many times, it can be
        # helpful to wrap the block inside a lock and unlock pair.
        #
        # It is safe to nest locking and unlocking calls. The surface will
        # only be unlocked after the final lock is released.
        #

        self.fail()

    def test_map_rgb(self):
        color = Color(0, 128, 255, 64)
        surf = pygame.Surface((5, 5), SRCALPHA, 32)
        c = surf.map_rgb(color)
        self.assertEqual(surf.unmap_rgb(c), color)

        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
        surf.fill(c)
        self.assertEqual(surf.get_at((0, 0)), color)

        surf.fill((0, 0, 0, 0))
        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
        surf.set_at((0, 0), c)
        self.assertEqual(surf.get_at((0, 0)), color)

    def todo_test_mustlock(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.mustlock:

        # Surface.mustlock(): return bool
        # test if the Surface requires locking
        #
        # Returns True if the Surface is required to be locked to access pixel
        # data. Usually pure software Surfaces do not require locking. This
        # method is rarely needed, since it is safe and quickest to just lock
        # all Surfaces as needed.
        #
        # All pygame functions will automatically lock and unlock the Surface
        # data as needed. If a section of code is going to make calls that
        # will repeatedly lock and unlock the Surface many times, it can be
        # helpful to wrap the block inside a lock and unlock pair.
        #

        self.fail()

    def test_set_alpha_none(self):
        """surf.set_alpha(None) disables blending"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((0, 255, 0, 128))
        s.set_alpha(None)
        self.assertEqual(None, s.get_alpha())

        s2 = pygame.Surface((1, 1), SRCALPHA, 32)
        s2.fill((255, 0, 0, 255))
        s2.blit(s, (0, 0))
        self.assertEqual(s2.get_at((0, 0))[0], 0, "the red component should be 0")

    def test_set_alpha_value(self):
        """surf.set_alpha(x), where x != None, enables blending"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((0, 255, 0, 128))
        s.set_alpha(255)

        s2 = pygame.Surface((1, 1), SRCALPHA, 32)
        s2.fill((255, 0, 0, 255))
        s2.blit(s, (0, 0))
        self.assertGreater(
            s2.get_at((0, 0))[0], 0, "the red component should be above 0"
        )

    def test_palette_colorkey(self):
        """ test bug discovered by robertpfeiffer
        https://github.com/pygame/pygame/issues/721
        """
        surf = pygame.image.load(example_path(os.path.join("data", "alien2.png")))
        key = surf.get_colorkey()
        self.assertEqual(surf.get_palette()[surf.map_rgb(key)], key)

    def test_palette_colorkey_set_px(self):
        surf = pygame.image.load(example_path(os.path.join("data", "alien2.png")))
        key = surf.get_colorkey()
        surf.set_at((0, 0), key)
        self.assertEqual(surf.get_at((0, 0)), key)

    def test_palette_colorkey_fill(self):
        surf = pygame.image.load(example_path(os.path.join("data", "alien2.png")))
        key = surf.get_colorkey()
        surf.fill(key)
        self.assertEqual(surf.get_at((0, 0)), key)

    def test_set_palette(self):
        palette = [pygame.Color(i, i, i) for i in range(256)]
        palette[10] = tuple(palette[10])  # 4 element tuple
        palette[11] = tuple(palette[11])[0:3]  # 3 element tuple

        surf = pygame.Surface((2, 2), 0, 8)
        pygame.display.init()
        try:
            pygame.display.set_mode((100, 50))
            surf.set_palette(palette)
            for i in range(256):
                self.assertEqual(surf.map_rgb(palette[i]), i, "palette color %i" % (i,))
                c = palette[i]
                surf.fill(c)
                self.assertEqual(surf.get_at((0, 0)), c, "palette color %i" % (i,))
            for i in range(10):
                palette[i] = pygame.Color(255 - i, 0, 0)
            surf.set_palette(palette[0:10])
            for i in range(256):
                self.assertEqual(surf.map_rgb(palette[i]), i, "palette color %i" % (i,))
                c = palette[i]
                surf.fill(c)
                self.assertEqual(surf.get_at((0, 0)), c, "palette color %i" % (i,))
            self.assertRaises(ValueError, surf.set_palette, [Color(1, 2, 3, 254)])
            self.assertRaises(ValueError, surf.set_palette, (1, 2, 3, 254))
        finally:
            pygame.display.quit()

    def test_set_palette__fail(self):
        pygame.init()
        palette = 256 * [(10, 20, 30)]
        surf = pygame.Surface((2, 2), 0, 32)
        self.assertRaises(pygame.error, surf.set_palette, palette)
        pygame.quit()

    def test_set_palette_at(self):
        pygame.display.init()
        try:
            pygame.display.set_mode((100, 50))
            surf = pygame.Surface((2, 2), 0, 8)
            original = surf.get_palette_at(10)
            replacement = Color(1, 1, 1, 255)
            if replacement == original:
                replacement = Color(2, 2, 2, 255)
            surf.set_palette_at(10, replacement)
            self.assertEqual(surf.get_palette_at(10), replacement)
            next = tuple(original)
            surf.set_palette_at(10, next)
            self.assertEqual(surf.get_palette_at(10), next)
            next = tuple(original)[0:3]
            surf.set_palette_at(10, next)
            self.assertEqual(surf.get_palette_at(10), next)
            self.assertRaises(IndexError, surf.set_palette_at, 256, replacement)
            self.assertRaises(IndexError, surf.set_palette_at, -1, replacement)
        finally:
            pygame.display.quit()

    def test_subsurface(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.subsurface:

        # Surface.subsurface(Rect): return Surface
        # create a new surface that references its parent
        #
        # Returns a new Surface that shares its pixels with its new parent.
        # The new Surface is considered a child of the original. Modifications
        # to either Surface pixels will effect each other. Surface information
        # like clipping area and color keys are unique to each Surface.
        #
        # The new Surface will inherit the palette, color key, and alpha
        # settings from its parent.
        #
        # It is possible to have any number of subsurfaces and subsubsurfaces
        # on the parent. It is also possible to subsurface the display Surface
        # if the display mode is not hardware accelerated.
        #
        # See the Surface.get_offset(), Surface.get_parent() to learn more
        # about the state of a subsurface.
        #

        surf = pygame.Surface((16, 16))
        s = surf.subsurface(0, 0, 1, 1)
        s = surf.subsurface((0, 0, 1, 1))

        # s = surf.subsurface((0,0,1,1), 1)
        # This form is not acceptable.
        # s = surf.subsurface(0,0,10,10, 1)

        self.assertRaises(ValueError, surf.subsurface, (0, 0, 1, 1, 666))

        self.assertEqual(s.get_shifts(), surf.get_shifts())
        self.assertEqual(s.get_masks(), surf.get_masks())
        self.assertEqual(s.get_losses(), surf.get_losses())

        # Issue 2 at Bitbucket.org/pygame/pygame
        surf = pygame.Surface.__new__(pygame.Surface)
        self.assertRaises(pygame.error, surf.subsurface, (0, 0, 0, 0))

    def todo_test_unlock(self):

        # __doc__ (as of 2008-08-02) for pygame.surface.Surface.unlock:

        # Surface.unlock(): return None
        # unlock the Surface memory from pixel access
        #
        # Unlock the Surface pixel data after it has been locked. The unlocked
        # Surface can once again be drawn and managed by Pygame. See the
        # Surface.lock() documentation for more details.
        #
        # All pygame functions will automatically lock and unlock the Surface
        # data as needed. If a section of code is going to make calls that
        # will repeatedly lock and unlock the Surface many times, it can be
        # helpful to wrap the block inside a lock and unlock pair.
        #
        # It is safe to nest locking and unlocking calls. The surface will
        # only be unlocked after the final lock is released.
        #

        self.fail()

    def test_unmap_rgb(self):
        # Special case, 8 bit-per-pixel surface (has a palette).
        surf = pygame.Surface((2, 2), 0, 8)
        c = (1, 1, 1)  # Unlikely to be in a default palette.
        i = 67
        pygame.display.init()
        try:
            pygame.display.set_mode((100, 50))
            surf.set_palette_at(i, c)
            unmapped_c = surf.unmap_rgb(i)
            self.assertEqual(unmapped_c, c)
            # Confirm it is a Color instance
            self.assertIsInstance(unmapped_c, pygame.Color)
        finally:
            pygame.display.quit()

        # Remaining, non-pallete, cases.
        c = (128, 64, 12, 255)
        formats = [(0, 16), (0, 24), (0, 32), (SRCALPHA, 16), (SRCALPHA, 32)]
        for flags, bitsize in formats:
            surf = pygame.Surface((2, 2), flags, bitsize)
            unmapped_c = surf.unmap_rgb(surf.map_rgb(c))
            surf.fill(c)
            comparison_c = surf.get_at((0, 0))
            self.assertEqual(
                unmapped_c,
                comparison_c,
                "%s != %s, flags: %i, bitsize: %i"
                % (unmapped_c, comparison_c, flags, bitsize),
            )
            # Confirm it is a Color instance
            self.assertIsInstance(unmapped_c, pygame.Color)

    def test_scroll(self):
        scrolls = [
            (8, 2, 3),
            (16, 2, 3),
            (24, 2, 3),
            (32, 2, 3),
            (32, -1, -3),
            (32, 0, 0),
            (32, 11, 0),
            (32, 0, 11),
            (32, -11, 0),
            (32, 0, -11),
            (32, -11, 2),
            (32, 2, -11),
        ]
        for bitsize, dx, dy in scrolls:
            surf = pygame.Surface((10, 10), 0, bitsize)
            surf.fill((255, 0, 0))
            surf.fill((0, 255, 0), (2, 2, 2, 2))
            comp = surf.copy()
            comp.blit(surf, (dx, dy))
            surf.scroll(dx, dy)
            w, h = surf.get_size()
            for x in range(w):
                for y in range(h):
                    self.assertEqual(
                        surf.get_at((x, y)),
                        comp.get_at((x, y)),
                        "%s != %s, bpp:, %i, x: %i, y: %i"
                        % (surf.get_at((x, y)), comp.get_at((x, y)), bitsize, dx, dy),
                    )
        # Confirm clip rect containment
        surf = pygame.Surface((20, 13), 0, 32)
        surf.fill((255, 0, 0))
        surf.fill((0, 255, 0), (7, 1, 6, 6))
        comp = surf.copy()
        clip = Rect(3, 1, 8, 14)
        surf.set_clip(clip)
        comp.set_clip(clip)
        comp.blit(surf, (clip.x + 2, clip.y + 3), surf.get_clip())
        surf.scroll(2, 3)
        w, h = surf.get_size()
        for x in range(w):
            for y in range(h):
                self.assertEqual(surf.get_at((x, y)), comp.get_at((x, y)))
        # Confirm keyword arguments and per-pixel alpha
        spot_color = (0, 255, 0, 128)
        surf = pygame.Surface((4, 4), pygame.SRCALPHA, 32)
        surf.fill((255, 0, 0, 255))
        surf.set_at((1, 1), spot_color)
        surf.scroll(dx=1)
        self.assertEqual(surf.get_at((2, 1)), spot_color)
        surf.scroll(dy=1)
        self.assertEqual(surf.get_at((2, 2)), spot_color)
        surf.scroll(dy=1, dx=1)
        self.assertEqual(surf.get_at((3, 3)), spot_color)
        surf.scroll(dx=-3, dy=-3)
        self.assertEqual(surf.get_at((0, 0)), spot_color)


class SurfaceSubtypeTest(unittest.TestCase):
    """Issue #280: Methods that return a new Surface preserve subclasses"""

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_copy(self):
        """Ensure method copy() preserves the surface's class

        When Surface is subclassed, the inherited copy() method will return
        instances of the subclass. Non Surface fields are uncopied, however.
        This includes instance attributes.
        """
        expected_size = (32, 32)
        ms1 = SurfaceSubclass(expected_size, SRCALPHA, 32)
        ms2 = ms1.copy()

        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, "test_attribute")
        self.assertEqual(ms2.get_size(), expected_size)

    def test_convert(self):
        """Ensure method convert() preserves the surface's class

        When Surface is subclassed, the inherited convert() method will return
        instances of the subclass. Non Surface fields are omitted, however.
        This includes instance attributes.
        """
        expected_size = (32, 32)
        ms1 = SurfaceSubclass(expected_size, 0, 24)
        ms2 = ms1.convert(24)

        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, "test_attribute")
        self.assertEqual(ms2.get_size(), expected_size)

    def test_convert_alpha(self):
        """Ensure method convert_alpha() preserves the surface's class

        When Surface is subclassed, the inherited convert_alpha() method will
        return instances of the subclass. Non Surface fields are omitted,
        however. This includes instance attributes.
        """
        pygame.display.set_mode((40, 40))
        expected_size = (32, 32)
        s = pygame.Surface(expected_size, SRCALPHA, 16)
        ms1 = SurfaceSubclass(expected_size, SRCALPHA, 32)
        ms2 = ms1.convert_alpha(s)

        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, "test_attribute")
        self.assertEqual(ms2.get_size(), expected_size)

    def test_subsurface(self):
        """Ensure method subsurface() preserves the surface's class

        When Surface is subclassed, the inherited subsurface() method will
        return instances of the subclass. Non Surface fields are uncopied,
        however. This includes instance attributes.
        """
        expected_size = (10, 12)
        ms1 = SurfaceSubclass((32, 32), SRCALPHA, 32)
        ms2 = ms1.subsurface((4, 5), expected_size)

        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, "test_attribute")
        self.assertEqual(ms2.get_size(), expected_size)


class SurfaceGetBufferTest(unittest.TestCase):

    # These tests requires ctypes. They are disabled if ctypes
    # is not installed.
    #
    try:
        ArrayInterface
    except NameError:
        __tags__ = ("ignore", "subprocess_ignore")

    lilendian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN

    def _check_interface_2D(self, s):
        s_w, s_h = s.get_size()
        s_bytesize = s.get_bytesize()
        s_pitch = s.get_pitch()
        s_pixels = s._pixels_address

        # check the array interface structure fields.
        v = s.get_view("2")
        if not IS_PYPY:
            flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
            if s.get_pitch() == s_w * s_bytesize:
                flags |= PAI_FORTRAN

            inter = ArrayInterface(v)

            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 2)
            self.assertEqual(inter.typekind, "u")
            self.assertEqual(inter.itemsize, s_bytesize)
            self.assertEqual(inter.shape[0], s_w)
            self.assertEqual(inter.shape[1], s_h)
            self.assertEqual(inter.strides[0], s_bytesize)
            self.assertEqual(inter.strides[1], s_pitch)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.data, s_pixels)

    def _check_interface_3D(self, s):
        s_w, s_h = s.get_size()
        s_bytesize = s.get_bytesize()
        s_pitch = s.get_pitch()
        s_pixels = s._pixels_address
        s_shifts = list(s.get_shifts())

        # Check for RGB or BGR surface.
        if s_shifts[0:3] == [0, 8, 16]:
            if self.lilendian:
                # RGB
                offset = 0
                step = 1
            else:
                # BGR
                offset = s_bytesize - 1
                step = -1
        elif s_shifts[0:3] == [8, 16, 24]:
            if self.lilendian:
                # xRGB
                offset = 1
                step = 1
            else:
                # BGRx
                offset = s_bytesize - 2
                step = -1
        elif s_shifts[0:3] == [16, 8, 0]:
            if self.lilendian:
                # BGR
                offset = 2
                step = -1
            else:
                # RGB
                offset = s_bytesize - 3
                step = 1
        elif s_shifts[0:3] == [24, 16, 8]:
            if self.lilendian:
                # BGRx
                offset = 2
                step = -1
            else:
                # RGBx
                offset = s_bytesize - 4
                step = -1
        else:
            return

        # check the array interface structure fields.
        v = s.get_view("3")
        if not IS_PYPY:
            inter = ArrayInterface(v)
            flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 3)
            self.assertEqual(inter.typekind, "u")
            self.assertEqual(inter.itemsize, 1)
            self.assertEqual(inter.shape[0], s_w)
            self.assertEqual(inter.shape[1], s_h)
            self.assertEqual(inter.shape[2], 3)
            self.assertEqual(inter.strides[0], s_bytesize)
            self.assertEqual(inter.strides[1], s_pitch)
            self.assertEqual(inter.strides[2], step)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.data, s_pixels + offset)

    def _check_interface_rgba(self, s, plane):
        s_w, s_h = s.get_size()
        s_bytesize = s.get_bytesize()
        s_pitch = s.get_pitch()
        s_pixels = s._pixels_address
        s_shifts = s.get_shifts()
        s_masks = s.get_masks()

        # Find the color plane position within the pixel.
        if not s_masks[plane]:
            return
        alpha_shift = s_shifts[plane]
        offset = alpha_shift // 8
        if not self.lilendian:
            offset = s_bytesize - offset - 1

        # check the array interface structure fields.
        v = s.get_view("rgba"[plane])
        if not IS_PYPY:
            inter = ArrayInterface(v)
            flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 2)
            self.assertEqual(inter.typekind, "u")
            self.assertEqual(inter.itemsize, 1)
            self.assertEqual(inter.shape[0], s_w)
            self.assertEqual(inter.shape[1], s_h)
            self.assertEqual(inter.strides[0], s_bytesize)
            self.assertEqual(inter.strides[1], s_pitch)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.data, s_pixels + offset)

    def test_array_interface(self):
        self._check_interface_2D(pygame.Surface((5, 7), 0, 8))
        self._check_interface_2D(pygame.Surface((5, 7), 0, 16))
        self._check_interface_2D(pygame.Surface((5, 7), pygame.SRCALPHA, 16))
        self._check_interface_3D(pygame.Surface((5, 7), 0, 24))
        self._check_interface_3D(pygame.Surface((8, 4), 0, 24))  # No gaps
        self._check_interface_2D(pygame.Surface((5, 7), 0, 32))
        self._check_interface_3D(pygame.Surface((5, 7), 0, 32))
        self._check_interface_2D(pygame.Surface((5, 7), pygame.SRCALPHA, 32))
        self._check_interface_3D(pygame.Surface((5, 7), pygame.SRCALPHA, 32))

    def test_array_interface_masks(self):
        """Test non-default color byte orders on 3D views"""

        sz = (5, 7)
        # Reversed RGB byte order
        s = pygame.Surface(sz, 0, 32)
        s_masks = list(s.get_masks())
        masks = [0xFF, 0xFF00, 0xFF0000]
        if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
            masks = s_masks[2::-1] + s_masks[3:4]
            self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))
        s = pygame.Surface(sz, 0, 24)
        s_masks = list(s.get_masks())
        masks = [0xFF, 0xFF00, 0xFF0000]
        if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
            masks = s_masks[2::-1] + s_masks[3:4]
            self._check_interface_3D(pygame.Surface(sz, 0, 24, masks))

        masks = [0xFF00, 0xFF0000, 0xFF000000, 0]
        self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))

        # Unsupported RGB byte orders
        if pygame.get_sdl_version()[0] == 1:
            # Invalid mask values with SDL2
            masks = [0xFF00, 0xFF, 0xFF0000, 0]
            self.assertRaises(
                ValueError, pygame.Surface(sz, 0, 24, masks).get_view, "3"
            )

    def test_array_interface_alpha(self):
        for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
            masks = [0xFF << s for s in shifts]
            s = pygame.Surface((4, 2), pygame.SRCALPHA, 32, masks)
            self._check_interface_rgba(s, 3)

    def test_array_interface_rgb(self):
        for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
            masks = [0xFF << s for s in shifts]
            masks[3] = 0
            for plane in range(3):
                s = pygame.Surface((4, 2), 0, 24)
                self._check_interface_rgba(s, plane)
                s = pygame.Surface((4, 2), 0, 32)
                self._check_interface_rgba(s, plane)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_bytes(self):
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_buffer()
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertFalse(b.readonly)
        b = Importer(a, buftools.PyBUF_FORMAT)
        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.format, "B")
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, (a.length,))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.strides, (1,))
        s2 = s.subsurface((1, 1, 7, 4))  # Not contiguous
        a = s2.get_buffer()
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s2._pixels_address)
        b = Importer(a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))
        b = Importer(a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))
        b = Importer(a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_0D(self):
        # This is the same handler as used by get_buffer(), so just
        # confirm that it succeeds for one case.
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view("0")
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_1D(self):
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view("1")
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertFalse(b.readonly)
        b = Importer(a, buftools.PyBUF_FORMAT)
        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.format, "=I")
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertEqual(b.shape, (s.get_width() * s.get_height(),))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.strides, (s.get_bytesize(),))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_2D(self):
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view("2")
        # Non dimensional requests, no PyDEF_ND, are handled by the
        # 1D surface buffer code, so only need to confirm a success.
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        # Uniquely 2D
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertEqual(b.shape, s.get_size())
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, "=I")
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, "=I")
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, None)
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, None)
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        s2 = s.subsurface((1, 1, 7, 4))  # Not contiguous
        a = s2.get_view("2")
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s2.get_bytesize())
        self.assertEqual(b.shape, s2.get_size())
        self.assertEqual(b.strides, (s2.get_bytesize(), s.get_pitch()))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s2._pixels_address)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, "=I")
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_3D(self):
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((12, 6), 0, 24)
        rmask, gmask, bmask, amask = s.get_masks()
        if self.lilendian:
            if rmask == 0x0000FF:
                color_step = 1
                addr_offset = 0
            else:
                color_step = -1
                addr_offset = 2
        else:
            if rmask == 0xFF0000:
                color_step = 1
                addr_offset = 0
            else:
                color_step = -1
                addr_offset = 2
        a = s.get_view("3")
        b = Importer(a, buftools.PyBUF_STRIDES)
        w, h = s.get_size()
        shape = w, h, 3
        strides = 3, s.get_pitch(), color_step
        self.assertEqual(b.ndim, 3)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, shape)
        self.assertEqual(b.strides, strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address + addr_offset)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, "B")
        self.assertEqual(b.strides, strides)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, "B")
        self.assertEqual(b.strides, strides)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    def test_newbuf_PyBUF_flags_rgba(self):
        # All color plane views are handled by the same routine,
        # so only one plane need be checked.
        from pygame.tests.test_utils import buftools

        Importer = buftools.Importer
        s = pygame.Surface((12, 6), 0, 24)
        rmask, gmask, bmask, amask = s.get_masks()
        if self.lilendian:
            if rmask == 0x0000FF:
                addr_offset = 0
            else:
                addr_offset = 2
        else:
            if rmask == 0xFF0000:
                addr_offset = 0
            else:
                addr_offset = 2
        a = s.get_view("R")
        b = Importer(a, buftools.PyBUF_STRIDES)
        w, h = s.get_size()
        shape = w, h
        strides = s.get_bytesize(), s.get_pitch()
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, shape)
        self.assertEqual(b.strides, strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address + addr_offset)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, "B")
        self.assertEqual(b.strides, strides)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, "B")
        self.assertEqual(b.strides, strides)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)


class SurfaceBlendTest(unittest.TestCase):
    def setUp(self):
        # Needed for 8 bits-per-pixel color palette surface tests.
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    _test_palette = [
        (0, 0, 0, 255),
        (10, 30, 60, 0),
        (25, 75, 100, 128),
        (200, 150, 100, 200),
        (0, 100, 200, 255),
    ]
    surf_size = (10, 12)
    _test_points = [
        ((0, 0), 1),
        ((4, 5), 1),
        ((9, 0), 2),
        ((5, 5), 2),
        ((0, 11), 3),
        ((4, 6), 3),
        ((9, 11), 4),
        ((5, 6), 4),
    ]

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self._test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self._test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def _assert_surface(self, surf, palette=None, msg=""):
        if palette is None:
            palette = self._test_palette
        if surf.get_bitsize() == 16:
            palette = [surf.unmap_rgb(surf.map_rgb(c)) for c in palette]
        for posn, i in self._test_points:
            self.assertEqual(
                surf.get_at(posn),
                palette[i],
                "%s != %s: flags: %i, bpp: %i, posn: %s%s"
                % (
                    surf.get_at(posn),
                    palette[i],
                    surf.get_flags(),
                    surf.get_bitsize(),
                    posn,
                    msg,
                ),
            )

    def test_blit_blend(self):
        sources = [
            self._make_src_surface(8),
            self._make_src_surface(16),
            self._make_src_surface(16, srcalpha=True),
            self._make_src_surface(24),
            self._make_src_surface(32),
            self._make_src_surface(32, srcalpha=True),
        ]
        destinations = [
            self._make_surface(8),
            self._make_surface(16),
            self._make_surface(16, srcalpha=True),
            self._make_surface(24),
            self._make_surface(32),
            self._make_surface(32, srcalpha=True),
        ]
        blend = [
            ("BLEND_ADD", (0, 25, 100, 255), lambda a, b: min(a + b, 255)),
            ("BLEND_SUB", (100, 25, 0, 100), lambda a, b: max(a - b, 0)),
            ("BLEND_MULT", (100, 200, 0, 0), lambda a, b: (a * b) // 256),
            ("BLEND_MIN", (255, 0, 0, 255), min),
            ("BLEND_MAX", (0, 255, 0, 255), max),
        ]

        for src in sources:
            src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
            for dst in destinations:
                for blend_name, dst_color, op in blend:
                    dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                    p = []
                    for sc in src_palette:
                        c = [op(dc[i], sc[i]) for i in range(3)]
                        if dst.get_masks()[3]:
                            c.append(dc[3])
                        else:
                            c.append(255)
                        c = dst.unmap_rgb(dst.map_rgb(c))
                        p.append(c)
                    dst.fill(dst_color)
                    dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                    self._assert_surface(
                        dst,
                        p,
                        (
                            ", op: %s, src bpp: %i"
                            ", src flags: %i"
                            % (blend_name, src.get_bitsize(), src.get_flags())
                        ),
                    )

        src = self._make_src_surface(32)
        masks = src.get_masks()
        dst = pygame.Surface(
            src.get_size(), 0, 32, [masks[2], masks[1], masks[0], masks[3]]
        )
        for blend_name, dst_color, op in blend:
            p = []
            for src_color in self._test_palette:
                c = [op(dst_color[i], src_color[i]) for i in range(3)]
                c.append(255)
                p.append(tuple(c))
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, ", %s" % blend_name)

        # Blend blits are special cased for 32 to 32 bit surfaces.
        #
        # Confirm that it works when the rgb bytes are not the
        # least significant bytes.
        pat = self._make_src_surface(32)
        masks = pat.get_masks()
        if min(masks) == intify(0xFF000000):
            masks = [longify(m) >> 8 for m in masks]
        else:
            masks = [intify(m << 8) for m in masks]
        src = pygame.Surface(pat.get_size(), 0, 32, masks)
        self._fill_surface(src)
        dst = pygame.Surface(src.get_size(), 0, 32, masks)
        for blend_name, dst_color, op in blend:
            p = []
            for src_color in self._test_palette:
                c = [op(dst_color[i], src_color[i]) for i in range(3)]
                c.append(255)
                p.append(tuple(c))
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, ", %s" % blend_name)

    def test_blit_blend_rgba(self):
        sources = [
            self._make_src_surface(8),
            self._make_src_surface(16),
            self._make_src_surface(16, srcalpha=True),
            self._make_src_surface(24),
            self._make_src_surface(32),
            self._make_src_surface(32, srcalpha=True),
        ]
        destinations = [
            self._make_surface(8),
            self._make_surface(16),
            self._make_surface(16, srcalpha=True),
            self._make_surface(24),
            self._make_surface(32),
            self._make_surface(32, srcalpha=True),
        ]
        blend = [
            ("BLEND_RGBA_ADD", (0, 25, 100, 255), lambda a, b: min(a + b, 255)),
            ("BLEND_RGBA_SUB", (0, 25, 100, 255), lambda a, b: max(a - b, 0)),
            ("BLEND_RGBA_MULT", (0, 7, 100, 255), lambda a, b: (a * b) // 256),
            ("BLEND_RGBA_MIN", (0, 255, 0, 255), min),
            ("BLEND_RGBA_MAX", (0, 255, 0, 255), max),
        ]

        for src in sources:
            src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
            for dst in destinations:
                for blend_name, dst_color, op in blend:
                    dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                    p = []
                    for sc in src_palette:
                        c = [op(dc[i], sc[i]) for i in range(4)]
                        if not dst.get_masks()[3]:
                            c[3] = 255
                        c = dst.unmap_rgb(dst.map_rgb(c))
                        p.append(c)
                    dst.fill(dst_color)
                    dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                    self._assert_surface(
                        dst,
                        p,
                        (
                            ", op: %s, src bpp: %i"
                            ", src flags: %i"
                            % (blend_name, src.get_bitsize(), src.get_flags())
                        ),
                    )

        # Blend blits are special cased for 32 to 32 bit surfaces
        # with per-pixel alpha.
        #
        # Confirm the general case is used instead when the formats differ.
        src = self._make_src_surface(32, srcalpha=True)
        masks = src.get_masks()
        dst = pygame.Surface(
            src.get_size(), SRCALPHA, 32, (masks[2], masks[1], masks[0], masks[3])
        )
        for blend_name, dst_color, op in blend:
            p = [
                tuple([op(dst_color[i], src_color[i]) for i in range(4)])
                for src_color in self._test_palette
            ]
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, ", %s" % blend_name)

        # Confirm this special case handles subsurfaces.
        src = pygame.Surface((8, 10), SRCALPHA, 32)
        dst = pygame.Surface((8, 10), SRCALPHA, 32)
        tst = pygame.Surface((8, 10), SRCALPHA, 32)
        src.fill((1, 2, 3, 4))
        dst.fill((40, 30, 20, 10))
        subsrc = src.subsurface((2, 3, 4, 4))
        subdst = dst.subsurface((2, 3, 4, 4))
        subdst.blit(subsrc, (0, 0), special_flags=BLEND_RGBA_ADD)
        tst.fill((40, 30, 20, 10))
        tst.fill((41, 32, 23, 14), (2, 3, 4, 4))
        for x in range(8):
            for y in range(10):
                self.assertEqual(
                    dst.get_at((x, y)),
                    tst.get_at((x, y)),
                    "%s != %s at (%i, %i)"
                    % (dst.get_at((x, y)), tst.get_at((x, y)), x, y),
                )

    def test_blit_blend_big_rect(self):
        """ test that an oversized rect works ok.
        """
        color = (1, 2, 3, 255)
        area = (1, 1, 30, 30)
        s1 = pygame.Surface((4, 4), 0, 32)
        r = s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)

        self.assertEqual(pygame.Rect((1, 1, 3, 3)), r)
        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)

        black = pygame.Color("black")
        red = pygame.Color("red")
        self.assertNotEqual(black, red)

        surf = pygame.Surface((10, 10), 0, 32)
        surf.fill(black)
        subsurf = surf.subsurface(pygame.Rect(0, 1, 10, 8))
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)

        subsurf.fill(red, (0, -1, 10, 1), pygame.BLEND_RGB_ADD)
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)

        subsurf.fill(red, (0, 8, 10, 1), pygame.BLEND_RGB_ADD)
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)

    def test_GET_PIXELVALS(self):
        # surface.h GET_PIXELVALS bug regarding whether of not
        # a surface has per-pixel alpha. Looking at the Amask
        # is not enough. The surface's SRCALPHA flag must also
        # be considered. Fix rev. 1923.
        src = self._make_surface(32, srcalpha=True)
        src.fill((0, 0, 0, 128))
        src.set_alpha(None)  # Clear SRCALPHA flag.
        dst = self._make_surface(32, srcalpha=True)
        dst.blit(src, (0, 0), special_flags=BLEND_RGBA_ADD)
        self.assertEqual(dst.get_at((0, 0)), (0, 0, 0, 255))

    def test_fill_blend(self):
        destinations = [
            self._make_surface(8),
            self._make_surface(16),
            self._make_surface(16, srcalpha=True),
            self._make_surface(24),
            self._make_surface(32),
            self._make_surface(32, srcalpha=True),
        ]
        blend = [
            ("BLEND_ADD", (0, 25, 100, 255), lambda a, b: min(a + b, 255)),
            ("BLEND_SUB", (0, 25, 100, 255), lambda a, b: max(a - b, 0)),
            ("BLEND_MULT", (0, 7, 100, 255), lambda a, b: (a * b) // 256),
            ("BLEND_MIN", (0, 255, 0, 255), min),
            ("BLEND_MAX", (0, 255, 0, 255), max),
        ]

        for dst in destinations:
            dst_palette = [dst.unmap_rgb(dst.map_rgb(c)) for c in self._test_palette]
            for blend_name, fill_color, op in blend:
                fc = dst.unmap_rgb(dst.map_rgb(fill_color))
                self._fill_surface(dst)
                p = []
                for dc in dst_palette:
                    c = [op(dc[i], fc[i]) for i in range(3)]
                    if dst.get_masks()[3]:
                        c.append(dc[3])
                    else:
                        c.append(255)
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(fill_color, special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, ", %s" % blend_name)

    def test_fill_blend_rgba(self):
        destinations = [
            self._make_surface(8),
            self._make_surface(16),
            self._make_surface(16, srcalpha=True),
            self._make_surface(24),
            self._make_surface(32),
            self._make_surface(32, srcalpha=True),
        ]
        blend = [
            ("BLEND_RGBA_ADD", (0, 25, 100, 255), lambda a, b: min(a + b, 255)),
            ("BLEND_RGBA_SUB", (0, 25, 100, 255), lambda a, b: max(a - b, 0)),
            ("BLEND_RGBA_MULT", (0, 7, 100, 255), lambda a, b: (a * b) // 256),
            ("BLEND_RGBA_MIN", (0, 255, 0, 255), min),
            ("BLEND_RGBA_MAX", (0, 255, 0, 255), max),
        ]

        for dst in destinations:
            dst_palette = [dst.unmap_rgb(dst.map_rgb(c)) for c in self._test_palette]
            for blend_name, fill_color, op in blend:
                fc = dst.unmap_rgb(dst.map_rgb(fill_color))
                self._fill_surface(dst)
                p = []
                for dc in dst_palette:
                    c = [op(dc[i], fc[i]) for i in range(4)]
                    if not dst.get_masks()[3]:
                        c[3] = 255
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(fill_color, special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, ", %s" % blend_name)


class SurfaceSelfBlitTest(unittest.TestCase):
    """Blit to self tests.

    This test case is in response to MotherHamster Bugzilla Bug 19.
    """

    def setUp(self):
        # Needed for 8 bits-per-pixel color palette surface tests.
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    _test_palette = [(0, 0, 0, 255), (255, 0, 0, 0), (0, 255, 0, 255)]
    surf_size = (9, 6)

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self._test_palette
        surf.fill(palette[1])
        surf.fill(palette[2], (1, 2, 1, 2))

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self._test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        self._fill_surface(surf, palette)
        return surf

    def _assert_same(self, a, b):
        w, h = a.get_size()
        for x in range(w):
            for y in range(h):
                self.assertEqual(
                    a.get_at((x, y)),
                    b.get_at((x, y)),
                    (
                        "%s != %s, bpp: %i"
                        % (a.get_at((x, y)), b.get_at((x, y)), a.get_bitsize())
                    ),
                )

    def test_overlap_check(self):
        # Ensure overlapping blits are properly detected. There are two
        # places where this is done, within SoftBlitPyGame() in alphablit.c
        # and PySurface_Blit() in surface.c. SoftBlitPyGame should catch the
        # per-pixel alpha surface, PySurface_Blit the colorkey and blanket
        # alpha surface. per-pixel alpha and blanket alpha self blits are
        # not properly handled by SDL 1.2.13, so Pygame does them.
        bgc = (0, 0, 0, 255)
        rectc_left = (128, 64, 32, 255)
        rectc_right = (255, 255, 255, 255)
        colors = [(255, 255, 255, 255), (128, 64, 32, 255)]
        overlaps = [
            (0, 0, 1, 0, (50, 0)),
            (0, 0, 49, 1, (98, 2)),
            (0, 0, 49, 49, (98, 98)),
            (49, 0, 0, 1, (0, 2)),
            (49, 0, 0, 49, (0, 98)),
        ]
        surfs = [pygame.Surface((100, 100), SRCALPHA, 32)]
        surf = pygame.Surface((100, 100), 0, 32)
        surf.set_alpha(255)
        surfs.append(surf)
        surf = pygame.Surface((100, 100), 0, 32)
        surf.set_colorkey((0, 1, 0))
        surfs.append(surf)
        for surf in surfs:
            for s_x, s_y, d_x, d_y, test_posn in overlaps:
                surf.fill(bgc)
                surf.fill(rectc_right, (25, 0, 25, 50))
                surf.fill(rectc_left, (0, 0, 25, 50))
                surf.blit(surf, (d_x, d_y), (s_x, s_y, 50, 50))
                self.assertEqual(surf.get_at(test_posn), rectc_right)

    # https://github.com/pygame/pygame/issues/370#issuecomment-364625291
    @unittest.skipIf("ppc64le" in platform.uname(), "known ppc64le issue")
    def test_colorkey(self):
        # Check a workaround for an SDL 1.2.13 surface self-blit problem
        # (MotherHamster Bugzilla bug 19).
        pygame.display.set_mode((100, 50))  # Needed for 8bit surface
        bitsizes = [8, 16, 24, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            surf.set_colorkey(self._test_palette[1])
            surf.blit(surf, (3, 0))
            p = []
            for c in self._test_palette:
                c = surf.unmap_rgb(surf.map_rgb(c))
                p.append(c)
            p[1] = (p[1][0], p[1][1], p[1][2], 0)
            tmp = self._make_surface(32, srcalpha=True, palette=p)
            tmp.blit(tmp, (3, 0))
            tmp.set_alpha(None)
            comp = self._make_surface(bitsize)
            comp.blit(tmp, (0, 0))
            self._assert_same(surf, comp)

    # https://github.com/pygame/pygame/issues/370#issuecomment-364625291
    @unittest.skipIf("ppc64le" in platform.uname(), "known ppc64le issue")
    def test_blanket_alpha(self):
        # Check a workaround for an SDL 1.2.13 surface self-blit problem
        # (MotherHamster Bugzilla bug 19).
        pygame.display.set_mode((100, 50))  # Needed for 8bit surface
        bitsizes = [8, 16, 24, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            surf.set_alpha(128)
            surf.blit(surf, (3, 0))
            p = []
            for c in self._test_palette:
                c = surf.unmap_rgb(surf.map_rgb(c))
                p.append((c[0], c[1], c[2], 128))
            tmp = self._make_surface(32, srcalpha=True, palette=p)
            tmp.blit(tmp, (3, 0))
            tmp.set_alpha(None)
            comp = self._make_surface(bitsize)
            comp.blit(tmp, (0, 0))
            self._assert_same(surf, comp)

    def test_pixel_alpha(self):
        bitsizes = [16, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize, srcalpha=True)
            comp = self._make_surface(bitsize, srcalpha=True)
            comp.blit(surf, (3, 0))
            surf.blit(surf, (3, 0))
            self._assert_same(surf, comp)

    def test_blend(self):
        bitsizes = [8, 16, 24, 32]
        blends = ["BLEND_ADD", "BLEND_SUB", "BLEND_MULT", "BLEND_MIN", "BLEND_MAX"]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            comp = self._make_surface(bitsize)
            for blend in blends:
                self._fill_surface(surf)
                self._fill_surface(comp)
                comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                self._assert_same(surf, comp)

    def test_blend_rgba(self):
        bitsizes = [16, 32]
        blends = [
            "BLEND_RGBA_ADD",
            "BLEND_RGBA_SUB",
            "BLEND_RGBA_MULT",
            "BLEND_RGBA_MIN",
            "BLEND_RGBA_MAX",
        ]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize, srcalpha=True)
            comp = self._make_surface(bitsize, srcalpha=True)
            for blend in blends:
                self._fill_surface(surf)
                self._fill_surface(comp)
                comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                self._assert_same(surf, comp)

    def test_subsurface(self):
        # Blitting a surface to its subsurface is allowed.
        surf = self._make_surface(32, srcalpha=True)
        comp = surf.copy()
        comp.blit(surf, (3, 0))
        sub = surf.subsurface((3, 0, 6, 6))
        sub.blit(surf, (0, 0))
        del sub
        self._assert_same(surf, comp)
        # Blitting a subsurface to its owner is forbidden because of
        # lock conficts. This limitation allows the overlap check
        # in PySurface_Blit of alphablit.c to be simplified.
        def do_blit(d, s):
            d.blit(s, (0, 0))

        sub = surf.subsurface((1, 1, 2, 2))
        self.assertRaises(pygame.error, do_blit, surf, sub)

    def test_copy_alpha(self):
        """issue 581: alpha of surface copy with SRCALPHA is set to 0."""
        surf = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
        self.assertEqual(surf.get_alpha(), 255)
        surf2 = surf.copy()
        self.assertEqual(surf2.get_alpha(), 255)


class SurfaceFillTest(unittest.TestCase):
    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_fill(self):
        screen = pygame.display.set_mode((640, 480))

        # Green and blue test pattern
        screen.fill((0, 255, 0), (0, 0, 320, 240))
        screen.fill((0, 255, 0), (320, 240, 320, 240))
        screen.fill((0, 0, 255), (320, 0, 320, 240))
        screen.fill((0, 0, 255), (0, 240, 320, 240))

        # Now apply a clip rect, such that only the left side of the
        # screen should be effected by blit operations.
        screen.set_clip((0, 0, 320, 480))

        # Test fills with each special flag, and additionally without any.
        screen.fill((255, 0, 0, 127), (160, 0, 320, 30), 0)
        screen.fill((255, 0, 0, 127), (160, 30, 320, 30), pygame.BLEND_ADD)
        screen.fill((0, 127, 127, 127), (160, 60, 320, 30), pygame.BLEND_SUB)
        screen.fill((0, 63, 63, 127), (160, 90, 320, 30), pygame.BLEND_MULT)
        screen.fill((0, 127, 127, 127), (160, 120, 320, 30), pygame.BLEND_MIN)
        screen.fill((127, 0, 0, 127), (160, 150, 320, 30), pygame.BLEND_MAX)
        screen.fill((255, 0, 0, 127), (160, 180, 320, 30), pygame.BLEND_RGBA_ADD)
        screen.fill((0, 127, 127, 127), (160, 210, 320, 30), pygame.BLEND_RGBA_SUB)
        screen.fill((0, 63, 63, 127), (160, 240, 320, 30), pygame.BLEND_RGBA_MULT)
        screen.fill((0, 127, 127, 127), (160, 270, 320, 30), pygame.BLEND_RGBA_MIN)
        screen.fill((127, 0, 0, 127), (160, 300, 320, 30), pygame.BLEND_RGBA_MAX)
        screen.fill((255, 0, 0, 127), (160, 330, 320, 30), pygame.BLEND_RGB_ADD)
        screen.fill((0, 127, 127, 127), (160, 360, 320, 30), pygame.BLEND_RGB_SUB)
        screen.fill((0, 63, 63, 127), (160, 390, 320, 30), pygame.BLEND_RGB_MULT)
        screen.fill((0, 127, 127, 127), (160, 420, 320, 30), pygame.BLEND_RGB_MIN)
        screen.fill((255, 0, 0, 127), (160, 450, 320, 30), pygame.BLEND_RGB_MAX)

        # Update the display so we can see the results
        pygame.display.flip()

        # Compare colors on both sides of window
        for y in range(5, 480, 10):
            self.assertEqual(screen.get_at((10, y)), screen.get_at((330, 480 - y)))


if __name__ == "__main__":
    unittest.main()
