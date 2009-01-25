__tags__ = ['array']

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

pygame.init()


arraytype = ""
try:
    import pygame.surfarray
except ImportError:
    pass
else:
    arraytype = pygame.surfarray.get_arraytype()
    if arraytype == 'numpy':
        from numpy import \
             uint8, uint16, uint32, uint64, zeros, float64
    elif arraytype == 'numeric':
        from Numeric import \
             UInt8 as uint8, UInt16 as uint16, UInt32 as uint32, zeros, \
             Float64 as float64
    else:
        print ("Unknown array type %s; tests skipped" %
               pygame.surfarray.get_arraytype())
        arraytype = ""

class SurfarrayModuleTest (unittest.TestCase):

    pixels2d = {8: True, 16: True, 24: False, 32: True}
    pixels3d = {8: False, 16: False, 24: True, 32: True}
    array2d = {8: True, 16: True, 24: True, 32: True}
    array3d = {8: False, 16: False, 24: True, 32: True}

    test_palette = [(0, 0, 0, 255),
                    (10, 30, 60, 255),
                    (25, 75, 100, 255),
                    (100, 150, 200, 255),
                    (0, 100, 200, 255)]
    surf_size = (10, 12)
    test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2),
                   ((5, 5), 2), ((0, 11), 3), ((4, 6), 3),
                   ((9, 11), 4), ((5, 6), 4)]

    def _make_surface(self, bitsize, srcalpha=False):
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in self.test_palette])
        return surf

    def _fill_surface(self, surf):
        surf.fill(self.test_palette[1], (0, 0, 5, 6))
        surf.fill(self.test_palette[2], (5, 0, 5, 6))
        surf.fill(self.test_palette[3], (0, 6, 5, 6))
        surf.fill(self.test_palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False):
        surf = self._make_surface(bitsize, srcalpha)
        self._fill_surface(surf)
        return surf

    def _assert_surface(self, surf, palette=None, msg=""):
        if palette is None:
            palette = self.test_palette
        if surf.get_bitsize() == 16:
            palette = [surf.unmap_rgb(surf.map_rgb(c)) for c in palette]
        for posn, i in self.test_points:
            self.failUnlessEqual(surf.get_at(posn), palette[i],
                                 "%s != %s: flags: %i, bpp: %i, posn: %s%s" %
                                 (surf.get_at(posn),
                                  palette[i], surf.get_flags(),
                                  surf.get_bitsize(), posn, msg))

    def _make_array3d(self, dtype):
        return zeros((self.surf_size[0], self.surf_size[1], 3), dtype)

    def _fill_array2d(self, arr, surf):
        palette = self.test_palette
        arr[:5,:6] = surf.map_rgb(palette[1])
        arr[5:,:6] = surf.map_rgb(palette[2])
        arr[:5,6:] = surf.map_rgb(palette[3])
        arr[5:,6:] = surf.map_rgb(palette[4])

    def _fill_array3d(self, arr):
        palette = self.test_palette
        arr[:5,:6] = palette[1][:3]
        arr[5:,:6] = palette[2][:3]
        arr[:5,6:] = palette[3][:3]
        arr[5:,6:] = palette[4][:3]

    def _make_src_array3d(self, dtype):
        arr = self._make_array3d(dtype)
        self._fill_array3d(arr)
        return arr

    def _make_array2d(self, dtype):
        return zeros(self.surf_size, dtype)

    def _assert_locking(self, func):
        surf = self._make_surface(32)

        arr = func(surf)
        self.failUnless(surf.get_locked())

        # Numpy uses the surface's buffer.
        if arraytype == "numeric":
            self.failUnlessEqual(sf.get_locks(), (ar,))

        surf.unlock()
        self.failUnless(surf.get_locked())

        del arr
        self.failUnless(surf.get_locked())
        self.failUnlessEqual(surf.get_locks(), ())

    def setUp(self):
        # Needed for 8 bits-per-pixel color palette surface tests.
        pygame.init()

        # Makes sure the same array package is used each time.
        if arraytype:
            pygame.surfarray.use_arraytype(arraytype)

    def tearDown(self):
        pygame.quit()

    def test_array2d(self):
        if not arraytype:
            self.fail("no array package installed")
        if arraytype == 'numeric':
            # This is known to fail with Numeric
            return

        sources = [self._make_src_surface(8),
                   self._make_src_surface(16),
                   self._make_src_surface(16, srcalpha=True),
                   self._make_src_surface(24),
                   self._make_src_surface(32),
                   self._make_src_surface(32, srcalpha=True)]
        palette = self.test_palette
        alpha_color = (0, 0, 0, 128)

        for surf in sources:
            arr = pygame.surfarray.array2d(surf)
            map_rgb = surf.map_rgb
            for (x, y), i in self.test_points:
                self.failUnlessEqual(arr[x, y], map_rgb(palette[i]),
                                     "%s != %s: flags: %i, bpp: %i, posn: %s" %
                                     (arr[x, y],
                                      map_rgb(palette[i]),
                                      surf.get_flags(), surf.get_bitsize(),
                                      (x, y)))

            if surf.get_flags() & SRCALPHA:
                surf.fill(alpha_color)
                arr = pygame.surfarray.array2d(surf)
                self.failUnlessEqual(arr[0, 0], map_rgb(alpha_color),
                                     "%s != %s: bpp: %i" %
                                     (arr[0, 0],
                                      map_rgb(alpha_color),
                                      surf.get_bitsize()))

    def test_array3d(self):
        if not arraytype:
            self.fail("no array package installed")

        sources = [self._make_src_surface(16),
                   self._make_src_surface(16, srcalpha=True),
                   self._make_src_surface(24),
                   self._make_src_surface(32),
                   self._make_src_surface(32, srcalpha=True)]
        palette = self.test_palette

        for surf in sources:
            arr = pygame.surfarray.array3d(surf)
            map_rgb = surf.map_rgb
            unmap_rgb = surf.unmap_rgb
            def same_color(ac, sc):
                sc = unmap_rgb(map_rgb(sc))
                return (ac[0] == sc[0] and
                        ac[1] == sc[1] and
                        ac[2] == sc[2])
            for (x, y), i in self.test_points:
                self.failUnless(same_color(arr[x, y], palette[i]),
                                "%s != %s: flags: %i, bpp: %i, posn: %s" %
                                (tuple(arr[x, y]),
                                 unmap_rgb(map_rgb(palette[i])),
                                 surf.get_flags(), surf.get_bitsize(),
                                 (x, y)))

    def todo_test_array_alpha(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.array_alpha:

          # pygame.surfarray.array_alpha (Surface): return array
          # 
          # Copy pixel alphas into a 2d array.
          # 
          # Copy the pixel alpha values (degree of transparency) from a Surface
          # into a 2D array. This will work for any type of Surface
          # format. Surfaces without a pixel alpha will return an array with all
          # opaque values.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 
          # Copy the pixel alpha values (degree of transparency) from a Surface
          # into a 2D array. This will work for any type of Surface format.
          # Surfaces without a pixel alpha will return an array with all opaque
          # values.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 

        self.fail() 

    def todo_test_array_colorkey(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.array_colorkey:

          # pygame.surfarray.array_colorkey (Surface): return array
          # 
          # Copy the colorkey values into a 2d array.
          # 
          # Create a new array with the colorkey transparency value from each
          # pixel. If the pixel matches the colorkey it will be fully
          # tranparent; otherwise it will be fully opaque.
          # 
          # This will work on any type of Surface format. If the image has no
          # colorkey a solid opaque array will be returned.
          # 
          # This function will temporarily lock the Surface as pixels are
          # copied.
          # 
          # Create a new array with the colorkey transparency value from each
          # pixel. If the pixel matches the colorkey it will be fully
          # tranparent; otherwise it will be fully opaque.
          # 
          # This will work on any type of Surface format. If the image has no
          # colorkey a solid opaque array will be returned.
          # 
          # This function will temporarily lock the Surface as pixels are copied. 

        self.fail() 

    def test_blit_array(self):
        if not arraytype:
            self.fail("no array package installed")

        # bug 24 at http://pygame.motherhamster.org/bugzilla/
        if 'numpy' in pygame.surfarray.get_arraytypes():
            prev = pygame.surfarray.get_arraytype()
            # This would raise exception:
            #  File "[...]\pygame\_numpysurfarray.py", line 381, in blit_array
            #    (array[:,:,1::3] >> losses[1] << shifts[1]) | \
            # TypeError: unsupported operand type(s) for >>: 'float' and 'int'
            pygame.surfarray.use_arraytype('numpy')
            s = pygame.Surface((10,10), 0, 24)
            a = pygame.surfarray.array3d(s)
            pygame.surfarray.blit_array(s, a)
            prev = pygame.surfarray.use_arraytype(prev)

        # target surfaces
        targets = [self._make_surface(8),
                   self._make_surface(16),
                   self._make_surface(16, srcalpha=True),
                   self._make_surface(24),
                   self._make_surface(32),
                   self._make_surface(32, srcalpha=True),
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
                s = self._make_src_surface(bitsize, surf.get_flags())
                arr = pygame.surfarray.pixels2d(s)
                pygame.surfarray.blit_array(surf, arr)
                self._assert_surface(surf)

            if self.array2d[bitsize]:
                s = self._make_src_surface(bitsize, surf.get_flags())
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
        self.failUnlessEqual(surf.get_at((5, 5)), color)

        surf = self._make_surface(32, srcalpha=True)
        arr = zeros(surf.get_size(), uint32)
        color = (0, 111, 255, 63)
        arr[...] = surf.map_rgb(color)
        pygame.surfarray.blit_array(surf, arr)
        self.failUnlessEqual(surf.get_at((5, 5)), color)

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
        
    def test_get_arraytype(self):
        if not arraytype:
            self.fail("no array package installed")

        self.failUnless((pygame.surfarray.get_arraytype() in
                         ['numpy', 'numeric']),
                        ("unknown array type %s" %
                         pygame.surfarray.get_arraytype()))

    def test_get_arraytypes(self):
        if not arraytype:
            self.fail("no array package installed")

        arraytypes = pygame.surfarray.get_arraytypes()
        try:
            import numpy
        except ImportError:
            self.failIf('numpy' in arraytypes)
        else:
            self.failUnless('numpy' in arraytypes)

        try:
            import Numeric
        except ImportError:
            self.failIf('numeric' in arraytypes)
        else:
            self.failUnless('numeric' in arraytypes)

        for atype in arraytypes:
            self.failUnless(atype in ['numpy', 'numeric'],
                            "unknown array type %s" % atype)

    def todo_test_make_surface(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.make_surface:

          # pygame.surfarray.make_surface (array): return Surface
          # 
          # Copy an array to a new surface.
          # 
          # Create a new Surface that best resembles the data and format on the
          # array. The array can be 2D or 3D with any sized integer values.
          # 
          # Create a new Surface that best resembles the data and format on the
          # array. The array can be 2D or 3D with any sized integer values.
          # 

        self.fail() 

    def todo_test_map_array(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.map_array:

          # pygame.surfarray.map_array (Surface, array3d): return array2d
          # 
          # Map a 3D array into a 2D array.
          # 
          # Convert a 3D array into a 2D array. This will use the given Surface
          # format to control the conversion. Palette surface formats are not
          # supported.
          # 
          # Convert a 3D array into a 2D array. This will use the given Surface
          # format to control the conversion. Palette surface formats are not
          # supported.
          # 

        self.fail() 

    def test_pixels2d(self):
        if not arraytype:
            self.fail("no array package installed")
        if arraytype == 'numeric':
            # This is known to fail with Numeric.
            return

        sources = [self._make_surface(8),
                   self._make_surface(16, srcalpha=True),
                   self._make_surface(32, srcalpha=True)]

        for surf in sources:
            self.failIf(surf.get_locked())
            arr = pygame.surfarray.pixels2d(surf)
            self.failUnless(surf.get_locked())
            # Numpy uses the surface's buffer.
            if arraytype == "numeric":
                self.failUnlessEqual(surf.get_locks(), (ar,))
            self._fill_array2d(arr, surf)
            surf.unlock()
            self.failUnless(surf.get_locked())
            del arr
            self.failIf(surf.get_locked())
            self.failUnlessEqual(surf.get_locks(), ())
            self._assert_surface(surf)

        # Error checks
        def do_pixels2d(surf):
            pygame.surfarray.pixels2d(surf)

        self.failUnlessRaises(ValueError,
                              do_pixels2d,
                              self._make_surface(24))

    def test_pixels3d(self):
        if not arraytype:
            self.fail("no array package installed")

        sources = [self._make_surface(24),
                   self._make_surface(32)]

        for surf in sources:
            self.failIf(surf.get_locked())
            arr = pygame.surfarray.pixels3d(surf)
            self.failUnless(surf.get_locked())
            # Numpy uses the surface's buffer.
            if arraytype == "numeric":
                self.failUnlessEqual(surf.get_locks(), (ar,))
            self._fill_array3d(arr)
            surf.unlock()
            self.failUnless(surf.get_locked())
            del arr
            self.failIf(surf.get_locked())
            self.failUnlessEqual(surf.get_locks(), ())
            self._assert_surface(surf)

        # Alpha check
        color = (1, 2, 3, 0)
        surf = self._make_surface(32, srcalpha=True)
        arr = pygame.surfarray.pixels3d(surf)
        arr[0,0] = color[:3]
        self.failUnlessEqual(surf.get_at((0, 0)), color)

        # Error checks
        def do_pixels3d(surf):
            pygame.surfarray.pixels3d(surf)

        self.failUnlessRaises(ValueError,
                              do_pixels3d,
                              self._make_surface(8))
        self.failUnlessRaises(ValueError,
                              do_pixels3d,
                              self._make_surface(16))

    def todo_test_pixels_alpha(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.pixels_alpha:

          # pygame.surfarray.pixels_alpha (Surface): return array
          # 
          # Reference pixel alphas into a 2d array.
          # 
          # Create a new 2D array that directly references the alpha values
          # (degree of transparency) in a Surface. Any changes to the array will
          # affect the pixels in the Surface. This is a fast operation since no
          # data is copied.
          # 
          # This can only work on 32-bit Surfaces with a per-pixel alpha value.
          # 
          # The Surface this array references will remain locked for the
          # lifetime of the array.
          # 
          # Create a new 2D array that directly references the alpha values
          # (degree of transparency) in a Surface. Any changes to the array will
          # affect the pixels in the Surface. This is a fast operation since no
          # data is copied.
          # 
          # This can only work on 32-bit Surfaces with a per-pixel alpha value. 
          # The Surface this array references will remain locked for the
          # lifetime of the array.
          # 

        self.fail() 

    def test_use_arraytype(self):
        if not arraytype:
            self.fail("no array package installed")

        def do_use_arraytype(atype):
            pygame.surfarray.use_arraytype(atype)

        try:
            import numpy
        except ImportError:
            self.failUnlessRaises(ValueError, do_use_arraytype, 'numpy')
            self.failIfEqual(pygame.surfarray.get_arraytype(), 'numpy')
        else:
            pygame.surfarray.use_arraytype('numpy')
            self.failUnlessEqual(pygame.surfarray.get_arraytype(), 'numpy')

        try:
            import Numeric
        except ImportError:
            self.failUnlessRaises(ValueError, do_use_arraytype, 'numeric')
            self.failIfEqual(pygame.surfarray.get_arraytype(), 'numeric')
        else:
            pygame.surfarray.use_arraytype('numeric')
            self.failUnlessEqual(pygame.surfarray.get_arraytype(), 'numeric')

        self.failUnlessRaises(ValueError, do_use_arraytype, 'not an option')

    def test_surf_lock (self):
        if not arraytype:
            self.fail("no array package installed")

        sf = pygame.Surface ((5, 5), 0, 32)
        for atype in pygame.surfarray.get_arraytypes ():
            pygame.surfarray.use_arraytype (atype)
            
            ar = pygame.surfarray.pixels2d (sf)
            self.assertEquals (sf.get_locked (), True)

            # Numpy uses the Surface's buffer.
            if atype == "numeric":
                self.assertEquals (sf.get_locks (), (ar,))
                
            sf.unlock ()
            self.assertEquals (sf.get_locked (), True)
                
            del ar
            self.assertEquals (sf.get_locked (), False)
            self.assertEquals (sf.get_locks (), ())

        #print "test_surf_lock - end"


if __name__ == '__main__':
    if not arraytype:
        print "No array package is installed. Cannot run unit tests."
    else:
        unittest.main()
