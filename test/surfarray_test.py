__tags__ = ['surfarray']

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

# Needed for 8 bits-per-pixel color palette surface tests
pygame.init()


skip_tests = False
try:
    from numpy import \
         uint8, uint16, uint32, uint64, zeros, float64
except ImportError:
    try:
        from Numeric import \
             UInt8 as uint8, UInt16 as uint16, UInt32 as uint32, zeros, \
             Float64 as float64
    except ImportError:
        skip_tests = True

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

    def test_import(self):
        'does it import'
        if skip_tests:
            return
        import pygame.surfarray

    def todo_test_array2d(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.array2d:

          # pygame.surfarray.array2d (Surface): return array
          # 
          # Copy pixels into a 2d array.
          # 
          # Copy the pixels from a Surface into a 2D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 
          # Copy the pixels from a Surface into a 2D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 

        self.fail() 

    def todo_test_array3d(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.array3d:

          # pygame.surfarray.array3d (Surface): return array
          # 
          # Copy pixels into a 3d array.
          # 
          # Copy the pixels from a Surface into a 3D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 
          # Copy the pixels from a Surface into a 3D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).
          # 

        self.fail() 

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
        if skip_tests:
            return

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
            surf = self._make_surface(bitsize, srcalpha=(bitsize != 24))
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
        
    def todo_test_get_arraytype(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.get_arraytype:

          # pygame.surfarray.get_arraytype (): return str
          # 
          # Gets the currently active array type.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.
          # 
          # New in pygame 1.8 

        self.fail() 

    def todo_test_get_arraytypes(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.get_arraytypes:

          # pygame.surfarray.get_arraytypes (): return tuple
          # 
          # Gets the array system types currently supported.
          # 
          # Checks, which array system types are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the use_arraytype () method.
          # 
          # If no supported array system could be found, None will be returned.
          # 
          # Checks, which array systems are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the pygame.surfarray.use_arraytype () method. If no supported array
          # system could be found, None will be returned.
          # 
          # New in pygame 1.8. 

        self.fail() 

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

    def todo_test_pixels2d(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.pixels2d:

          # pygame.surfarray.pixels2d (Surface): return array
          # 
          # Reference pixels into a 2d array.
          # 
          # Create a new 2D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # Pixels from a 24-bit Surface cannot be referenced, but all other
          # Surface bit depths can.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).
          # 
          # Create a new 2D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # Pixels from a 24-bit Surface cannot be referenced, but all other
          # Surface bit depths can.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).
          # 

        self.fail() 

    def todo_test_pixels3d(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.pixels3d:

          # pygame.surfarray.pixels3d (Surface): return array
          # 
          # Reference pixels into a 3d array.
          # 
          # Create a new 3D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # This will only work on Surfaces that have 24-bit or 32-bit
          # formats. Lower pixel formats cannot be referenced.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).
          # 
          # Create a new 3D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # This will only work on Surfaces that have 24-bit or 32-bit formats.
          # Lower pixel formats cannot be referenced.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).
          # 

        self.fail() 

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

    def todo_test_use_arraytype(self):

        # __doc__ (as of 2008-08-02) for pygame.surfarray.use_arraytype:

          # pygame.surfarray.use_arraytype (arraytype): return None
          # 
          # Sets the array system to be used for surface arrays.
          # 
          # Uses the requested array type for the module functions.
          # Currently supported array types are:
          # 
          #   numeric 
          #   numpy
          # 
          # If the requested type is not available, a ValueError will be raised.
          # 
          # Uses the requested array type for the module functions. Currently
          # supported array types are:
          # 
          #   numeric
          #   numpy
          # If the requested type is not available, a ValueError will be raised. 
          # New in pygame 1.8. 

        self.fail() 

if __name__ == '__main__':
    if skip_tests:
        print "No array package is installed. Cannot run unit tests."
    else:
        unittest.main()
