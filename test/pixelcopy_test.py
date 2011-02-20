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

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest
import pygame
from pygame.locals import *


from pygame.pixelcopy import surface_to_array, map_array

def unsigned32(i):
    """cast signed 32 bit integer to an unsigned integer"""
    return i & 0xFFFFFFFF

class PixelcopyModuleTest (unittest.TestCase):

    bitsizes = [8, 16, 32]

    test_palette = [(0, 0, 0, 255),
                    (10, 30, 60, 255),
                    (25, 75, 100, 255),
                    (100, 150, 200, 255),
                    (0, 100, 200, 255)]

    surf_size = (10, 12)
    test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2),
                   ((5, 5), 2), ((0, 11), 3), ((4, 6), 3),
                   ((9, 11), 4), ((5, 6), 4)]

    def __init__(self, *args, **kwds):
        pygame.display.init()
        try:
            unittest.TestCase.__init__(self, *args, **kwds)
            self.sources = [self._make_src_surface(8),
                            self._make_src_surface(16),
                            self._make_src_surface(16, srcalpha=True),
                            self._make_src_surface(24),
                            self._make_src_surface(32),
                            self._make_src_surface(32, srcalpha=True)]
        finally:
            pygame.display.quit()

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self.test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self.test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_surface_to_array_2d(self):
        alpha_color = (0, 0, 0, 128)

        for surf in self.sources:
            src_bitsize = surf.get_bitsize()
            for dst_bitsize in self.bitsizes:
                dst = pygame.Surface(surf.get_size(), 0, dst_bitsize)
                dst.fill((0, 0, 0, 0))
                view = dst.get_view('2')
                self.assertFalse(surf.get_locked())
                if dst_bitsize < src_bitsize:
                    self.assertRaises(ValueError, surface_to_array, view, surf)
                    self.assertFalse(surf.get_locked())
                    continue
                surface_to_array(view, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    sp = surf.get_at_mapped(posn)
                    dp = dst.get_at_mapped(posn)
                    self.assertEqual(dp, sp,
                                     "%s != %s: flags: %i"
                                     ", bpp: %i, posn: %s" %
                                     (dp, sp,
                                      surf.get_flags(), surf.get_bitsize(),
                                      posn))
                del view
                    
                if surf.get_masks()[3]:
                    dst.fill((0, 0, 0, 0))
                    view = dst.get_view('2')
                    posn = (2, 1)
                    surf.set_at(posn, alpha_color)
                    self.assertFalse(surf.get_locked())
                    surface_to_array(view, surf)
                    self.assertFalse(surf.get_locked())
                    sp = surf.get_at_mapped(posn)
                    dp = dst.get_at_mapped(posn)
                    self.assertEqual(dp, sp,
                                     "%s != %s: bpp: %i" %
                                     (dp, sp, surf.get_bitsize()))

    def test_surface_to_array_3d(self):
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            masks = (0xff, 0xff00, 0xff0000, 0)
        else:
            masks = (0xff000000, 0xff0000, 0xff00, 0)
        dst = pygame.Surface(self.surf_size, 0, 24, masks=masks)
        for surf in self.sources:
            dst.fill((0, 0, 0, 0))
            src_bitsize = surf.get_bitsize()
            view = dst.get_view('3')
            self.assertFalse(surf.get_locked())
            surface_to_array(view, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                sc = surf.get_at(posn)[0:3]
                dc = dst.get_at(posn)[0:3]
                self.assertEqual(dc, sc,
                                 "%s != %s: flags: %i"
                                 ", bpp: %i, posn: %s" %
                                 (dc, sc,
                                  surf.get_flags(), surf.get_bitsize(),
                                  posn))
            del view

    def todo_test_array_to_surface(self):
        # target surfaces
        targets = [_make_surface(8),
                   _make_surface(16),
                   _make_surface(16, srcalpha=True),
                   _make_surface(24),
                   _make_surface(32),
                   _make_surface(32, srcalpha=True),
                   ]
        
        # source arrays
        arrays3d = []
        dtypes = [(8, uint8), (16, uint16), (32, uint32)]
        try:
            dtypes.append((64, uint64))
        except NameError:
            pass
        arrays3d = [(self._make_src_array3d(dtype), None)
                    for __, dtype in dtypes]
        for bitsize in [8, 16, 24, 32]:
            palette = None
            if bitsize == 16:
                s = pygame.Surface((1,1), 0, 16)
                palette = [s.unmap_rgb(s.map_rgb(c))
                           for c in self.test_palette]
            if self.pixels3d[bitsize]:
                surf = self._make_src_surface(bitsize)
                arr = pygame.surfarray.pixels3d(surf)
                arrays3d.append((arr, palette))
            if self.array3d[bitsize]:
                surf = self._make_src_surface(bitsize)
                arr = pygame.surfarray.array3d(surf)
                arrays3d.append((arr, palette))
                for sz, dtype in dtypes:
                    arrays3d.append((arr.astype(dtype), palette))

        # tests on arrays
        def do_blit(surf, arr):
            pygame.surfarray.blit_array(surf, arr)

        for surf in targets:
            bitsize = surf.get_bitsize()
            for arr, palette in arrays3d:
                surf.fill((0, 0, 0, 0))
                if bitsize == 8:
                    self.failUnlessRaises(ValueError, do_blit, surf, arr)
                else:
                    pygame.surfarray.blit_array(surf, arr)
                    self._assert_surface(surf, palette)

            if self.pixels2d[bitsize]:
                surf.fill((0, 0, 0, 0))
                s = self._make_src_surface(bitsize, surf.get_flags() & SRCALPHA)
                arr = pygame.surfarray.pixels2d(s)
                pygame.surfarray.blit_array(surf, arr)
                self._assert_surface(surf)

            if self.array2d[bitsize]:
                s = self._make_src_surface(bitsize, surf.get_flags() & SRCALPHA)
                arr = pygame.surfarray.array2d(s)
                for sz, dtype in dtypes:
                    surf.fill((0, 0, 0, 0))
                    if sz >= bitsize:
                        pygame.surfarray.blit_array(surf, arr.astype(dtype))
                        self._assert_surface(surf)
                    else:
                        self.failUnlessRaises(ValueError, do_blit,
                                              surf, self._make_array2d(dtype))

        # Check alpha for 2D arrays
        surf = self._make_surface(16, srcalpha=True)
        arr = zeros(surf.get_size(), uint16)
        arr[...] = surf.map_rgb((0, 128, 255, 64))
        color = surf.unmap_rgb(arr[0, 0])
        pygame.surfarray.blit_array(surf, arr)
        self.assertEqual(surf.get_at((5, 5)), color)

        surf = self._make_surface(32, srcalpha=True)
        arr = zeros(surf.get_size(), uint32)
        color = (0, 111, 255, 63)
        arr[...] = surf.map_rgb(color)
        pygame.surfarray.blit_array(surf, arr)
        self.assertEqual(surf.get_at((5, 5)), color)

        # Check shifts
        arr3d = self._make_src_array3d(uint8)

        shift_tests = [(16,
                        [12, 0, 8, 4],
                        [0xf000, 0xf, 0xf00, 0xf0]),
                       (24,
                        [16, 0, 8, 0],
                        [0xff0000, 0xff, 0xff00, 0]),
                       (32,
                        [0, 16, 24, 8],
                        [0xff, 0xff0000, 0xff000000, 0xff00])]

        for bitsize, shifts, masks in shift_tests:
            surf = self._make_surface(bitsize, srcalpha=(shifts[3] != 0))
            palette = None
            if bitsize == 16:
                palette = [surf.unmap_rgb(surf.map_rgb(c))
                           for c in self.test_palette]
            surf.set_shifts(shifts)
            surf.set_masks(masks)
            pygame.surfarray.blit_array(surf, arr3d)
            self._assert_surface(surf, palette)

        # Invalid arrays
        surf = pygame.Surface((1,1), 0, 32)
        t = 'abcd'
        self.failUnlessRaises(ValueError, do_blit, surf, t)

        surf_size = self.surf_size
        surf = pygame.Surface(surf_size, 0, 32)
        arr = zeros([surf_size[0], surf_size[1] + 1, 3], uint32)
        self.failUnlessRaises(ValueError, do_blit, surf, arr)
        arr = zeros([surf_size[0] + 1, surf_size[1], 3], uint32)
        self.failUnlessRaises(ValueError, do_blit, surf, arr)

        surf = pygame.Surface((1, 4), 0, 32)
        arr = zeros((4,), uint32)
        self.failUnlessRaises(ValueError, do_blit, surf, arr)
        arr.shape = (1, 1, 1, 4)
        self.failUnlessRaises(ValueError, do_blit, surf, arr)

        arr = zeros((10, 10), float64)
        surf = pygame.Surface((10, 10), 0, 32)
        self.failUnlessRaises(ValueError, do_blit, surf, arr)

class PixelCopyTestWithArray(unittest.TestCase):
    global numpy
    try:
        import numpy
    except ImportError:
        __tags__ = ['ignore', 'subprocess_ignore']

    bitsizes = [8, 16, 32]

    test_palette = [(0, 0, 0, 255),
                    (10, 30, 60, 255),
                    (25, 75, 100, 255),
                    (100, 150, 200, 255),
                    (0, 100, 200, 255)]

    surf_size = (10, 12)
    test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2),
                   ((5, 5), 2), ((0, 11), 3), ((4, 6), 3),
                   ((9, 11), 4), ((5, 6), 4)]

    try:
        dst_types = [numpy.uint8, numpy.uint16, numpy.uint32]
        try:    
            dst_types.append(numpy.uint64) 
        except AttributeError:
            pass
    except NameError:
        pass

    def __init__(self, *args, **kwds):
        pygame.display.init()
        try:
            unittest.TestCase.__init__(self, *args, **kwds)
            self.sources = [self._make_src_surface(8),
                            self._make_src_surface(16),
                            self._make_src_surface(16, srcalpha=True),
                            self._make_src_surface(24),
                            self._make_src_surface(32),
                            self._make_src_surface(32, srcalpha=True)]
        finally:
            pygame.display.quit()

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self.test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self.test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_surface_to_array_2d(self):
        try:
            from numpy import empty, dtype
        except ImportError:
            return

        palette = self.test_palette
        alpha_color = (0, 0, 0, 128)

        dst_dims = self.surf_size
        destinations = [empty(dst_dims, t) for t in self.dst_types]
        if (pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN):
            swapped_dst = empty(dst_dims, dtype('>u4'))
        else:
            swapped_dst = empty(dst_dims, dtype('<u4'))

        for surf in self.sources:
            src_bytesize = surf.get_bytesize()
            for dst in destinations:
                if dst.itemsize < src_bytesize:
                    self.assertRaises(ValueError, surface_to_array, dst, surf)
                    continue
                dst[...] = 0
                self.assertFalse(surf.get_locked())
                surface_to_array(dst, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    sp = unsigned32(surf.get_at_mapped(posn))
                    dp = dst[posn]
                    self.assertEqual(dp, sp,
                                     "%s != %s: flags: %i"
                                     ", bpp: %i, dtype: %s,  posn: %s" %
                                     (dp, sp,
                                      surf.get_flags(), surf.get_bitsize(),
                                      dst.dtype,
                                      posn))
                    
                if surf.get_masks()[3]:
                    posn = (2, 1)
                    surf.set_at(posn, alpha_color)
                    surface_to_array(dst, surf)
                    sp = unsigned32(surf.get_at_mapped(posn))
                    dp = dst[posn]
                    self.assertEqual(dp, sp, "%s != %s: bpp: %i" %
                                     (dp, sp, surf.get_bitsize()))

            swapped_dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(swapped_dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = swapped_dst[posn]
                self.assertEqual(dp, sp,
                                 "%s != %s: flags: %i"
                                 ", bpp: %i, dtype: %s,  posn: %s" %
                                 (dp, sp,
                                  surf.get_flags(), surf.get_bitsize(),
                                  dst.dtype,
                                  posn))
                
            if surf.get_masks()[3]:
                posn = (2, 1)
                surf.set_at(posn, alpha_color)
                self.assertFalse(surf.get_locked())
                surface_to_array(swapped_dst, surf)
                self.assertFalse(surf.get_locked())
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = swapped_dst[posn]
                self.assertEqual(dp, sp, "%s != %s: bpp: %i" %
                                     (dp, sp, surf.get_bitsize()))

    def test_surface_to_array_3d(self):
        try:
            from numpy import empty, dtype
        except ImportError:
            return

        palette = self.test_palette

        dst_dims = self.surf_size + (3,)
        destinations = [empty(dst_dims, t) for t in self.dst_types]
        if (pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN):
            swapped_dst = empty(dst_dims, dtype('>u4'))
        else:
            swapped_dst = empty(dst_dims, dtype('<u4'))

        for surf in self.sources:
            src_bitsize = surf.get_bitsize()
            for dst in destinations:
                dst[...] = 0
                self.assertFalse(surf.get_locked())
                surface_to_array(dst, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
                    r_arr, g_arr, b_arr = dst[posn]
                    self.assertEqual(r_arr, r_surf,
                                     "%i != %i, color: red, flags: %i"
                                     ", bpp: %i, posn: %s" %
                                     (r_arr, r_surf,
                                      surf.get_flags(), surf.get_bitsize(),
                                      posn))
                    self.assertEqual(g_arr, g_surf,
                                     "%i != %i, color: green, flags: %i"
                                     ", bpp: %i, posn: %s" %
                                     (r_arr, r_surf,
                                      surf.get_flags(), surf.get_bitsize(),
                                      posn))
                    self.assertEqual(b_arr, b_surf,
                                     "%i != %i, color: blue, flags: %i"
                                     ", bpp: %i, posn: %s" %
                                     (r_arr, r_surf,
                                      surf.get_flags(), surf.get_bitsize(),
                                      posn))
        
            swapped_dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(swapped_dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
                r_arr, g_arr, b_arr = swapped_dst[posn]
                self.assertEqual(r_arr, r_surf,
                                 "%i != %i, color: red, flags: %i"
                                 ", bpp: %i, posn: %s" %
                                 (r_arr, r_surf,
                                  surf.get_flags(), surf.get_bitsize(),
                                  posn))
                self.assertEqual(g_arr, g_surf,
                                 "%i != %i, color: green, flags: %i"
                                 ", bpp: %i, posn: %s" %
                                 (r_arr, r_surf,
                                  surf.get_flags(), surf.get_bitsize(),
                                  posn))
                self.assertEqual(b_arr, b_surf,
                                 "%i != %i, color: blue, flags: %i"
                                 ", bpp: %i, posn: %s" %
                                 (r_arr, r_surf,
                                  surf.get_flags(), surf.get_bitsize(),
                                  posn))
        
    def test_map_array(self):
        try:
            from numpy import array, zeros, uint8, int32, alltrue
        except ImportError:
            return

        surf = pygame.Surface((1, 1), 0, 32)

        # color fill
        color = array([11, 17, 59], uint8)
        target = zeros((5, 7), int32)
        map_array(target, color, surf)
        self.assert_(alltrue(target == surf.map_rgb(color)))

        # array row stripes
        stripe = array([[2, 5, 7], [11, 19, 23], [37, 53, 101]], uint8)
        target = zeros((4, 3), int32)
        map_array(target, stripe, surf)
        target_stripe = array([surf.map_rgb(c) for c in stripe], int32)
        self.assert_(alltrue(target[...] == target_stripe))

if __name__ == '__main__':
    unittest.main()
